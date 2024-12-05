from functools import partial
from typing import Optional

import numpy as np
import ray
from KDEpy.bw_selection import (
    improved_sheather_jones,
    scotts_rule,
    silvermans_rule,
)
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y
from tqdm.auto import tqdm

from .utils import (
    HNSWActor,
    optimal_bandwidth,
    parse_param,
    predict_value_batch,
    to_iterator,
)


class NuqRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        log_pN=0.0,
        kernel_type="RBF",
        n_neighbors=20,
        tune_bandwidth="regression:n_folds=10;n_samples=3;verbose=1",
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
            for example, "regression:n_points=5;n_folds=10;n_samples=3",
            by default "regression"
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
        self.verbose = verbose
        self.batch_size = batch_size
        self.random_seed = random_seed

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
        elif "regression" in strategy:
            _, kwargs = parse_param(strategy)
            kwargs = dict(map(lambda x: (x[0], int(x[1])), kwargs.items()))
            bandwidth, score = optimal_bandwidth(
                self, X, y, mode="regression", **kwargs
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
          1. Constructs kNN index;
          2. Tunes kernel bandwidth [optional].
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

        # Get prior y and y^2 estimates
        self.y_mean_ = y.mean()
        self.y2_mean_ = (y ** 2).mean()

        # 1. Tune kernel bandwidth
        if self.tune_bandwidth is not None:
            bandwidth = self._tune_kernel(X, y, strategy=self.tune_bandwidth)
        bandwidth = np.array(bandwidth)
        if len(bandwidth.shape) not in [0, 1, 2]:
            raise ValueError(
                "Expected `bandwidth` to be a scalar, 1D or 2D array, "
                f"but got shape {bandwidth.shape} instead."
            )
        self.bandwidth_ref_ = ray.put(bandwidth)

        # 2. Construct kNN index on remote

        # Move preprocessed input to shared memory
        self.X_ref_, self.y_ref_ = ray.put(X), ray.put(y)

        self.index_ = HNSWActor.remote(
            self.X_ref_,
            n_neighbors=self.n_neighbors,
            random_seed=self.random_seed,
        )

        return self

    def predict(
        self,
        X: np.ndarray,
        return_uncertainty: Optional[str] = None,
        batch_size: Optional[int] = None,
    ):
        """Predicts probability distribution between classes for each input X.

        Parameters
        ----------
        X : np.ndarray
            array of shape [N, d]; each row is a vector to predict probabilities for
        return_uncertainty : str, optional
            whether to return probabilities with or without uncertainty
              - None: no uncertainty is returned
              - "aleatoric": log(var(x))
              - "epistemic": log(tau^2) without the normalizing factor, check NUQ paper

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
        X_ref = ray.put(X)
        idx_ref = self.index_.knn_query.remote(X_ref, return_dist=False)

        predict_value_handle = partial(
            predict_value_batch.remote,
            self.X_ref_,
            self.y_ref_,
            self.bandwidth_ref_,
            X_ref,
            idx_ref,
            batch_size=batch_size,
            kernel_type=self.kernel_type,
            log_pN=self.log_pN,
            y_mean=self.y_mean_,
            y2_mean=self.y2_mean_,
            return_uncertainty=(return_uncertainty is not None),
        )

        res_refs = []
        for i in range(0, X.shape[0], batch_size):
            res_refs.append(predict_value_handle(i))

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
        y_pr, log_aleatoric, log_epistemic = map(np.concatenate, zip(*res))

        if return_uncertainty is None:
            return y_pr
        elif return_uncertainty == "aleatoric":
            return y_pr, log_aleatoric
        elif return_uncertainty == "epistemic":
            return y_pr, log_epistemic
