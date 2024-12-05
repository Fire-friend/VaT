from functools import partial
from typing import Optional

import numpy as np
import ray
import torch
from KDEpy.bw_selection import (
    improved_sheather_jones,
    scotts_rule,
    silvermans_rule,
)
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y
from tqdm.auto import tqdm

from .utils import (
    HNSWActor,
    compute_centroids,
    optimal_bandwidth,
    parse_param,
    predict_log_proba_batch, to_iterator,
    # to_iterator,
)


class NuqClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            log_pN=0.0,
            kernel_type="RBF",
            n_neighbors=20,
            tune_bandwidth="classification",
            use_centroids=False,
            sparse=False,
            verbose=False,
            batch_size=1000,
            random_seed=42,
    ):
        """Constructs an instance of NuqClassifier.

        Parameters
        ----------
        log_pN : float, optional
            smoothing parameter, check the whitepaper, by default 0.0
        kernel_type : str, optional
            kernel to use, by default "RBF"
        n_neighbors : int, optional
            number of nearest neighbors, by default 20
        tune_bandwidth : str, optional
            bandwidth selection method, given by parameter string;
            for example, "classification:n_points=5;n_folds=10;n_samples=3",
            by default "classification"
        use_centroids : bool, optional
            whether to represent each class as a centroid, by default False
        sparse : bool, optional
            return sparse probability matrix, by default True
        verbose : bool, optional
            print logs and fitting/inference progress, by default False
        batch_size : int, optional
            number of samples in each batch, by default 1000
        random_seed : int, optional
            random seed to use, by default 42
        """
        self.log_pN = log_pN
        self.tune_bandwidth = tune_bandwidth
        self.kernel_type = kernel_type
        self.n_neighbors = n_neighbors
        self.use_centroids = use_centroids
        self.sparse = sparse
        self.verbose = verbose
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.ray = True

    def _tune_kernel(self, X, y, strategy="isj"):
        standard_strategies = {
            "isj": improved_sheather_jones,
            "silverman": silvermans_rule,
            "scott": scotts_rule,
        }
        if strategy in standard_strategies:
            method = standard_strategies[strategy]
            bandwidth = np.apply_along_axis(
                lambda x: method(x[..., None]), axis=0, arr=X
            )
            print(f"  [{strategy.upper()}] {bandwidth = }")
        elif "classification" in strategy:
            _, kwargs = parse_param(strategy)
            kwargs = dict(map(lambda x: (x[0], int(x[1])), kwargs.items()))
            bandwidth, score = optimal_bandwidth(
                self, X, y, mode="classification", **kwargs
            )
            print(f"  [{strategy.upper()}] {bandwidth = } ({score = })")
        else:
            raise ValueError("No such strategy")
        return bandwidth

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            bandwidth: Optional[np.ndarray] = None,
    ):
        """Prepares the model for inference:
          1. Computes centroid for each class [optional].
             This allows to reduce the number of points
             to the number of classes;
          2. Constructs kNN index;
          3. Tunes kernel bandwidth [optional].
             Uses `self.strategy` to find the optimal bandwidth.

        Parameters
        ----------
        X : np.ndarray
            embeddings
        y : np.ndarray
            labels
        bandwidth : np.ndarray, optional
            bandwidth for each embedding, by default None;
            should an array broadcastable with X
            https://numpy.org/devdocs/user/basics.broadcasting.html

        Returns
        -------
        self : NuqClassifier
            fitted model
        """

        X, y = check_X_y(X, y)
        self.label_encoder_ = LabelEncoder()
        y = self.label_encoder_.fit_transform(y)

        # 1. Compute centroid for each class
        if self.use_centroids:
            X, y = compute_centroids(embeddings=X, labels=y)
        self.n_classes_ = y.max() + 1

        # Get prior class weights
        _, counts = np.unique(y, return_counts=True)

        log_prior = np.log(counts) - np.log(len(y))
        self.class_default_ = log_prior.argmax()
        self.log_prior_default_ = log_prior[self.class_default_] + self.log_pN

        # Move log prior vector to shared memory
        if self.ray:
            self.log_prior_ref_ = ray.put(log_prior)
        else:
            self.log_prior_ref_ = log_prior

        # 2. Tune kernel bandwidth
        if self.tune_bandwidth is not None:
            bandwidth = self._tune_kernel(X, y, strategy=self.tune_bandwidth)
        bandwidth = np.array(bandwidth)
        if len(bandwidth.shape) not in [0, 1, 2]:
            raise ValueError(
                "Expected `bandwidth` to be a scalar, 1D or 2D array, "
                f"but got shape {bandwidth.shape} instead."
            )
        if self.ray:
            self.bandwidth_ref_ = ray.put(bandwidth)
        else:
            self.bandwidth_ref_ = bandwidth

        # 3. Construct kNN index on remote

        # Move preprocessed input to shared memory
        if self.ray:
            self.X_ref_, self.y_ref_ = ray.put(X), ray.put(y)
            self.index_ = HNSWActor.remote(
                self.X_ref_,
                n_neighbors=self.n_neighbors,
                random_seed=self.random_seed,
            )
        else:
            self.X_ref_, self.y_ref_ = X, y
            self.index_ = HNSWActor(
                self.X_ref_,
                n_neighbors=self.n_neighbors,
                random_seed=self.random_seed,
            )

        return self

    def predict_proba(
            self,
            X: np.ndarray,
            return_uncertainty: Optional[str] = None,
            batch_size: Optional[int] = None,
            keep_grad=False
    ):
        """Predicts probability distribution between classes for each input X.

        Parameters
        ----------
        X : np.ndarray
            array of shape [N, d]; each row is a vector to predict probabilities for
        return_uncertainty : str, optional
            whether to return probabilities with or without uncertainty
              - None: no uncertainty is returned
              - "aleatoric": log (1 - p_pred)
              - "epistemic": log (tau^2) without the normalizing factor, check NUQ paper

        batch_size : int, optional
            number of samples per batch, by default None

        Returns
        -------
        probs : Any[np.ndarray, scipy.sparse.csr_matrix]
            sparse/dense matrix of probabilities per sample
        uncertainty : np.ndarray [optional]
            corresponding uncertainties if `return_uncertainty` is not None
        """
        allowed = ["aleatoric", "epistemic"]
        if not (return_uncertainty is None or return_uncertainty in allowed):
            raise ValueError(
                "Unsupported `return_uncertainty` value. Expected one of"
                f"[None, 'aleatoric', 'epistemic'], but received {return_uncertainty}"
            )

        if batch_size is None:
            batch_size = self.batch_size

        # Move query to the shared memory
        if self.ray:
            if keep_grad:
                X_ref_knn = ray.put(X.detach().cpu().numpy())
            else:
                X_ref_knn = ray.put(X)
            idx_ref = self.index_.knn_query.remote(X_ref_knn, return_dist=False)
            X_ref = ray.put(X)
            predict_log_proba_handle = partial(
                predict_log_proba_batch.remote,
                self.X_ref_,
                self.y_ref_,
                self.bandwidth_ref_,
                X_ref,
                idx_ref,
                batch_size=batch_size,
                kernel_type=self.kernel_type,
                log_prior=self.log_prior_ref_,
                log_pN=self.log_pN,
                log_prior_default=self.log_prior_default_,
                class_default=self.class_default_,
                return_uncertainty=(return_uncertainty == "epistemic"),
                keep_grad=keep_grad,
            )
        else:
            X_ref = X
            idx_ref = self.index_.knn_query(X_ref, return_dist=False)
            predict_log_proba_handle = partial(
                predict_log_proba_batch,
                self.X_ref_,
                self.y_ref_,
                self.bandwidth_ref_,
                X_ref,
                idx_ref,
                batch_size=batch_size,
                kernel_type=self.kernel_type,
                log_prior=self.log_prior_ref_,
                log_pN=self.log_pN,
                log_prior_default=self.log_prior_default_,
                class_default=self.class_default_,
                return_uncertainty=(return_uncertainty == "epistemic"),
            )

        res_refs = []
        for i in range(0, X.shape[0], batch_size):
            res_refs.append(predict_log_proba_handle(i))

        res = []
        order = []
        for x in tqdm(
                to_iterator(res_refs),
                total=len(res_refs),
                disable=(not self.verbose),
        ):
            order.append(x[0])
            res.append(x[1:])

        # Preserve the batch ordering
        res = [res[i] for i in np.argsort(order)]
        if keep_grad:
            pred, log_proba, log_unc = map(torch.cat, zip(*res))
        else:
            pred, log_proba, log_unc = map(np.concatenate, zip(*res))

        indptr = np.arange(pred.shape[0] + 1, dtype=np.int64)
        if keep_grad:
            indices = pred.detach().cpu().numpy()
            data = np.exp(log_proba.detach().cpu().numpy())
        else:
            indices = pred
            data = np.exp(log_proba)

        probs = csr_matrix(
            (data, indices, indptr), shape=(X.shape[0], self.n_classes_)
        )
        if not self.sparse:
            probs = probs.toarray()

        if return_uncertainty is None:
            return probs
        elif return_uncertainty == "epistemic":
            return probs, log_unc
        elif return_uncertainty == "aleatoric":
            return probs, np.log1p(-np.exp(log_proba))

    def predict(self, X: np.ndarray, return_uncertainty: Optional[str] = None, keep_grad=False):
        """Predicts class for each entry and corresponding
        uncertainties (optional).

        Parameters
        ----------
        X : np.ndarray
            array of shape [N, d]; each row is a vector to predict probabilities for
        return_uncertainty : bool, optional
            whether to return probabilities with or without uncertainty

        Returns
        -------
        probs : Any[np.ndarray, scipy.sparse.csr_matrix]
            sparse/dense matrix of probabilities per sample
        return_uncertainty : str, optional
            whether to return probabilities with or without uncertainty
              - None: no uncertainty is returned
              - "aleatoric": log (1 - p_pred)
              - "epistemic": log (tau^2) without the normalizing factor, check NUQ paper
        """

        probs = self.predict_proba(X, return_uncertainty=return_uncertainty, keep_grad=keep_grad)
        if return_uncertainty:
            probs, uncertainty = probs
        probs = self.label_encoder_.inverse_transform(
            np.array(probs.argmax(axis=1)).ravel()
        )

        if return_uncertainty is None:
            return probs
        else:
            return probs, uncertainty
