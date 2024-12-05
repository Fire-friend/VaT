from collections import defaultdict
from functools import partial
from typing import Callable, Dict, Optional, Tuple, Union

import hnswlib
import numpy as np
import pandas as pd
import ray
from scipy.special import logsumexp
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm.auto import tqdm

from . import kernels


def to_iterator(obj_ids):
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])


@ray.remote
class HNSWActor:
    """Ray Actor interface for HNSW index to leverage the shared storage of `X_base` (vector index)."""

    def __init__(
        self, X_base, M=16, ef_construction=200, random_seed=42, n_neighbors=20
    ):
        self.M = M
        self.ef_construction = ef_construction
        self.random_seed = random_seed
        self.n_neighbors = n_neighbors

        N, dim = X_base.shape
        self.index = hnswlib.Index(space="l2", dim=dim)
        self.index.init_index(
            max_elements=N, M=16, ef_construction=200, random_seed=42
        )
        self.index.add_items(X_base)

    def knn_query(self, X, sl=np.s_[:, :], return_dist=True):
        idx, dist = self.index.knn_query(X[sl], k=self.n_neighbors)
        if return_dist:
            return idx, dist
        else:
            return idx


def predict_log_proba_single(
    X_base: np.ndarray,
    y_base: np.ndarray,
    X_query: np.ndarray,
    log_kernel: Callable,
    log_prior: np.ndarray,
    log_pN: float,
    log_prior_default: float,
    class_default: int,
    return_uncertainty: bool = False,
) -> Tuple[int, float, Optional[float]]:
    """Predict uncertainty for a single entry `X_query` given its
    neighbors (`base`) and their labels.

    Parameters
    ----------
        X_base : np.ndarray[k, dim] of floats
            Neighbors of the query point
        y_base : np.ndarray[k]
            Corresponding labels
        X_query : np.ndarray[dim]
            A single point to make prediction for
        log_kernel : Callable
            Kernel function to be called: `log_kernel(X_base, X_query)`
        log_prior : np.ndarray[n_classes]
            Prior distribution on the classes
        log_pN : float
            Prior importance hyperparameter
        log_prior_default : float
            Highest prior probability
        class_default : int
            Class with highest prior probability
        return_uncertainty : bool, optional
            Whether to compute uncertainty, by default False

    Returns
    -------
        pred : int
            Class with top probability
        log_proba : float
            Predicted class log probability
        log_unc : float
            Optional log uncertainty, -1. by default
    """
    # Create list of all present classes and their corresponding positions
    classes_cur, encoded = np.unique(y_base, return_inverse=True)

    # Compute kernel values for each pair of points
    log_kernel_vals = log_kernel(X_base, X_query)

    # Get positions for each class
    indices = defaultdict(list)
    for i, v in enumerate(encoded):
        indices[v].append(i)

    log_ps_cur = np.array(
        [
            logsumexp(log_kernel_vals[indices[k]])
            for k in range(len(classes_cur))
        ]
    )

    # Compute numerator for each class
    log_ps_total_cur = logsumexp(
        np.c_[
            log_ps_cur,
            log_pN + log_prior[classes_cur],
        ],
        axis=1,
    )

    # Compute denominator (it is the same for all classes)
    log_denominator = logsumexp(
        np.r_[
            log_ps_cur,
            log_pN,
        ]
    )

    # Select class with top probability
    idx_max = np.argmax(log_ps_total_cur)
    # If max probability is greater than all prior probabilities,
    # predict the corresponding class
    if log_ps_total_cur[idx_max] > log_prior_default:
        class_pred = classes_cur[idx_max]
        log_numerator_p = log_ps_total_cur[idx_max]
    # If max probability is still less than any prior probability
    # then just predict the top prior class
    else:
        class_pred = class_default
        log_numerator_p = log_prior_default

    # Compute the Nadaraya-Watson estimator
    log_ps_pred = log_numerator_p - log_denominator

    # In case we don't need the uncertainty just return None
    if not return_uncertainty:
        return class_pred, log_ps_pred, -1.0

    # Uncertainty prediction has the same two cases as probability
    # prediction. By default, the numerator is given by p*(1-p)
    # For convenience, (1-p) is denoted with _1mp
    if log_ps_total_cur[idx_max] > log_prior_default:
        log_numerator_1mp = logsumexp(
            np.r_[
                log_ps_cur[:idx_max],
                log_ps_cur[idx_max + 1 :],
                log_pN + np.log1p(-log_prior[idx_max]),
            ]
        )
    else:
        log_numerator_1mp = logsumexp(
            np.r_[
                log_ps_cur,
                log_pN + np.log1p(-log_prior_default),
            ]
        )

    log_uncertainty_total = (
        log_numerator_p + log_numerator_1mp - 3 * log_denominator
    )

    return class_pred, log_ps_pred, log_uncertainty_total


@ray.remote
def predict_log_proba_batch(
    X_base: np.ndarray,
    y_base: np.ndarray,
    bandwidth: np.ndarray,
    X_query: np.ndarray,
    idx_query: np.ndarray,
    i: int,
    batch_size: int,
    kernel_type: str,
    log_prior: np.ndarray,
    log_pN: float,
    log_prior_default: float,
    class_default: int,
    return_uncertainty: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs prediction pipeline on a single batch.

    Parameters
    ----------
    X_base : np.ndarray[N, dim] of floats
        Base data points (index database)
    y_base : np.ndarray[N]
        Corresponding labels of base data points
    bandwidth: np.ndarray[N] or scalar
        Array of bandwidthes or a single bandwidth
    X_query : np.ndarray[n, dim]
        Query of points to make prediction for
    idx_query : np.ndarray[n, k]
        Indices of neighbors for each `X_query` entry
    i : int
        Batch start position
    batch_size : int
        Batch size, `X_query[i: i + batch_size]` are taken
    kernel_type : str
        Kernel to be used: `log_kernel(X_base, X_query)`, see `kernels.py`
    log_prior : np.ndarray[n_classes]
        Prior distribution on the classes
    log_pN : float
        Prior importance hyperparameter
    log_prior_default : float
        Highest prior probability
    class_default : int
        Class with highest prior probability
    return_uncertainty : bool, optional
        Whether to compute uncertainty, by default False

    Returns
    -------
        i : int
            Batch start position, for order recovery
        pred : np.ndarray[int]
            Predicted class for each entry
        log_proba : np.ndarray[float]
            Predicted log probability for each entry
        log_unc : np.ndarray[float]
            Optional log uncertainty for each entry, -1. by default
    """
    predict_log_proba = partial(
        predict_log_proba_single,
        log_prior=log_prior,
        log_pN=log_pN,
        log_prior_default=log_prior_default,
        class_default=class_default,
        return_uncertainty=return_uncertainty,
    )
    sl = np.s_[i : i + batch_size]
    size = X_query[sl].shape[0]

    preds = np.empty(size, dtype=np.int64)
    log_probas = np.empty(size, dtype=np.float32)
    log_uncs = np.empty(size, dtype=np.float32)

    for j, (X, idx) in enumerate(zip(X_query[sl], idx_query[sl])):
        log_kernel = get_log_kernel(
            kernel_type,
            bandwidth[idx] if len(bandwidth.shape) == 2 else bandwidth,
        )
        preds[j], log_probas[j], log_uncs[j] = predict_log_proba(
            X_base[idx], y_base[idx], X, log_kernel
        )

    return i, preds, log_probas, log_uncs


def compute_centroids(
    embeddings: np.ndarray, labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes centroids for each label group

    Parameters
    ----------
    embeddings : np.ndarray
        embeddings
    labels : np.ndarray
        corresponding labels

    Returns
    -------
    centers : np.ndarray
        center of each centroid
    centers : np.ndarray
        corresponding labels
    """
    centers, labels_unique = [], []
    for c in np.unique(labels):
        mask = labels == c
        centers.append(np.mean(embeddings[mask], axis=0))
        labels_unique.append(c)
    return np.array(centers), np.array(labels_unique)


def optimal_bandwidth(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_points: int = 5,
    n_folds: int = 10,
    n_samples: int = 3,
    verbose: int = 0,
    mode: str = "classification",
) -> Tuple[float, float]:
    """Selects an optimal bandwidth via cross validation.

    Parameters
    ----------
    model
        model to optimize
    X : np.ndarray
        data points
    y : np.ndarray
        corresponding labels
    n_points : int, optional
        number of bandwidth values to check, by default 5
    n_folds : int, optional
        number of folds to create, by default 10
    n_samples : int, optional
        number of folds to compute score on, by default 3
    verbose : int, optional
        level of verbosity (show progressbar), by default 0
    mode : str, optional
        used to determine the cross-validation splitter type,
        by default "classification"
    Returns
    -------
    best_bandwidth : float
    best_score : float
    """
    ModelClass = type(model)
    allowed = ["classification", "regression"]
    if mode not in allowed:
        raise ValueError(
            "Unsupported `mode` value. Expected one of"
            f"['classification', 'regression'], but received {mode}"
        )
    elif mode == "classification":
        splitter = StratifiedKFold
    elif mode == "regression":
        splitter = KFold
    skf = splitter(
        n_splits=n_folds, shuffle=True, random_state=model.random_seed
    )
    it = skf.split(X, y)

    grid = None
    results = {}
    for i, (train_idx, val_idx) in tqdm(
        zip(range(n_samples), it),
        total=n_samples,
        disable=(not verbose),
        desc="Tuning bandwidth",
    ):
        # Prepare a new Nuq instance with disabled `tune_bandwidth`
        params = model.get_params()
        params.update(tune_bandwidth=None, verbose=False)
        nuq = ModelClass(**params)

        # Build kNN index
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        nuq.fit(X_train, y_train)

        # If grid is not set yet, initialize it
        if grid is None:
            # Compute squared distances
            _, squared_dists = ray.get(
                nuq.index_.knn_query.remote(nuq.X_ref_, return_dist=True)
            )
            # Exclude zero distances between point and itself
            dists = np.sqrt(squared_dists)[:, 1:]
            min_dists = dists[:, 0]
            left, right = min_dists[min_dists!=0].min(), dists.max()

            grid = np.logspace(np.log10(left), np.log10(right), n_points)

        # Compute score for every bandwidth
        scores = []
        for bandwidth in grid:
            nuq.bandwidth_ref_ = ray.put(np.array(bandwidth))
            scores.append(nuq.score(X_val, y_val))
        results[f"fold_{i}"] = scores
        del nuq

    results = pd.DataFrame(results)
    scores_mean = results.mean(axis=1)
    best_idx = scores_mean.argmax()

    return grid[best_idx], scores_mean[best_idx]


def get_log_kernel(
    name: str = "RBF", bandwidth: Union[float, np.ndarray] = 1.0
) -> Callable:
    """Constructs log kernel function with signature:
    f(X: [N, d], Y: [N, d]) -> values [N]

    Parameters
    ----------
    name : str, optional
        kernel name, check `nuq.kernels`, by default "RBF"
    bandwidth : Union[float, np.ndarray], optional
        bandwidth, by default 1.0

    Returns
    -------
    Callable
        log kernel function

    Raises
    ------
    ValueError
        unknown kernel
    """

    try:
        return getattr(kernels, name.lower())(bandwidth)
    except AttributeError:
        raise ValueError(f"Unsupported kernel: {name}")


def parse_param(param_string: str) -> Tuple[str, Dict[str, str]]:
    """Parses a string of form "name" or "name:value=1"
    or "name:value1=1;value2=2" and similar. Used to pass parameters
    without messing with dictionaries.

    Parameters
    ----------
    param_string : str
        string of the above form

    Returns
    -------
    Tuple[str, Dict[str, str]]
        either `name`, {} or `name` with dictionary of parameters
    """
    res = param_string.split(":", maxsplit=1)
    if len(res) == 1:
        return res[0], {}
    name, vals_string = res
    vals_strings = filter(None, vals_string.split(";"))
    vals = [v.split("=", maxsplit=1) for v in vals_strings]
    return name, dict(vals)


def predict_value_single(
    X_base: np.ndarray,
    y_base: np.ndarray,
    X_query: np.ndarray,
    log_kernel: Callable,
    log_pN: float,
    y_mean: float,
    y2_mean: float,
    return_uncertainty: bool = False,
) -> Tuple[int, float, float]:
    """Predict uncertainty for a single entry `X_query` given its
    neighbors (`base`) and their labels.

    Parameters
    ----------
        X_base : np.ndarray[k, dim] of floats
            Neighbors of the query point
        y_base : np.ndarray[k]
            Corresponding labels
        X_query : np.ndarray[dim]
            A single point to make prediction for
        log_kernel : Callable
            Kernel function to be called: `log_kernel(X_base, X_query)`
        log_pN : float
            Prior importance hyperparameter
        y_mean : float
            point estimate of value
        y2_mean : float
            point estimate of the squared value
        return_uncertainty : bool, optional
            Whether to compute uncertainty, by default False

    Returns
    -------
        pred : int
            Class with top probability
        log_aleatoric : float
            Optional log aleatoric uncertainty, -1. by default
        log_epistemic : float
            Optional log epistemic uncertainty, -1. by default
    """
    # Compute kernel values for each pair of points
    log_kernel_vals = log_kernel(X_base, X_query)

    # Compute denominator
    log_denominator = logsumexp(
        np.r_[
            log_kernel_vals,
            log_pN,
        ]
    )

    # Compute the weights
    weights = np.exp(log_kernel_vals - log_denominator)
    weights_bias = np.exp(log_pN - log_denominator)

    # Apply the weights to compute the estimates
    y_pr = (y_base * weights).sum() + y_mean * weights_bias
    if not return_uncertainty:
        return y_pr, -1.0, -1.0

    # Compute the aleatoric and epistemic uncertainties
    y2_pr = (y_base ** 2 * weights).sum() + y2_mean * weights_bias
    log_variance = np.log(y2_pr - y_pr ** 2)
    log_epistemic = log_variance - log_denominator

    return y_pr, log_variance, log_epistemic


@ray.remote
def predict_value_batch(
    X_base: np.ndarray,
    y_base: np.ndarray,
    bandwidth: np.ndarray,
    X_query: np.ndarray,
    idx_query: np.ndarray,
    i: int,
    batch_size: int,
    kernel_type: str,
    log_pN: float,
    y_mean: float,
    y2_mean: float,
    return_uncertainty: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs prediction pipeline on a single batch.

    Parameters
    ----------
    X_base : np.ndarray[N, dim] of floats
        Base data points (index database)
    y_base : np.ndarray[N]
        Corresponding labels of base data points
    bandwidth: np.ndarray[N] or scalar
        Array of bandwidthes or a single bandwidth
    X_query : np.ndarray[n, dim]
        Query of points to make prediction for
    idx_query : np.ndarray[n, k]
        Indices of neighbors for each `X_query` entry
    i : int
        Batch start position
    batch_size : int
        Batch size, `X_query[i: i + batch_size]` are taken
    kernel_type : str
        Kernel to be used: `log_kernel(X_base, X_query)`, see `kernels.py`
    log_pN : float
        Prior importance hyperparameter
    y_mean : float
        point estimate of value
    y2_mean : float
        point estimate of the squared value
    return_uncertainty : bool, optional
        Whether to compute uncertainty, by default False

    Returns
    -------
        i : int
            Batch start position, for order recovery
        pred : np.ndarray[int]
            Predicted class for each entry
        log_aleatoric : np.ndarray[float]
            Optional log aleatoric uncertainty for each entry, -1. by default
        log_epistemic : np.ndarray[float]
            Optional log epistemic uncertainty for each entry, -1. by default
    """
    predict_value = partial(
        predict_value_single,
        log_pN=log_pN,
        y_mean=y_mean,
        y2_mean=y2_mean,
        return_uncertainty=return_uncertainty,
    )
    sl = np.s_[i : i + batch_size]
    size = X_query[sl].shape[0]

    preds = np.empty(size, dtype=np.float32)
    log_aleatoric = np.empty(size, dtype=np.float32)
    log_epistemic = np.empty(size, dtype=np.float32)

    for j, (X, idx) in enumerate(zip(X_query[sl], idx_query[sl])):
        log_kernel = get_log_kernel(
            kernel_type,
            bandwidth[idx] if len(bandwidth.shape) == 2 else bandwidth,
        )
        preds[j], log_aleatoric[j], log_epistemic[j] = predict_value(
            X_base[idx], y_base[idx], X, log_kernel
        )

    return i, preds, log_aleatoric, log_epistemic
