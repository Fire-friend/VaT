from collections import defaultdict
from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from nuq.kernels import *
import time


class NuqRegressor():
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

    def fit(
            self,
            X,
            y,
            bandwidth: Optional[np.ndarray] = 11.037089
    ):
        self.X = torch.from_numpy(X).cuda()
        self.y = torch.from_numpy(y).cuda()

        # Get prior y and y^2 estimates
        self.y_mean_ = torch.mean(self.y, dim=0)
        self.y2_mean_ = torch.mean(self.y ** 2, dim=0)

        self.X_norm = F.normalize(self.X, dim=1)
        # self.y_norm = F.normalize(self.y, dim=1)

        # 1. Compute centroid for each class TODO
        if self.use_centroids:
            pass
        self.n_classes_ = int(np.min((torch.max(self.y) + 1).detach().cpu().numpy()))

        # Get prior class weights
        _, counts = torch.unique(self.y, return_counts=True)

        log_prior = torch.log(counts) - np.log(len(self.y))
        self.log_prior = log_prior
        self.class_default_ = torch.argmax(log_prior)
        self.log_prior_default_ = log_prior[self.class_default_] + self.log_pN

        # Move log prior vector to shared memory
        self.log_prior_ref_ = log_prior

        # Tune kernel bandwidth
        self.bandwidth_ref_ = bandwidth

        # kernel function
        self.log_kernel = RBF_grad(bandwidth)

    def predict(self, X_query: torch.tensor, return_uncertainty: Optional[str] = None):

        X_norm = F.normalize(X_query, dim=1)

        # select top K samples
        cos_similar = X_norm @ self.X_norm.T
        v, nn_index = torch.topk(cos_similar, self.n_neighbors, dim=1)

        nn_X_base = self.X[nn_index, :]
        nn_y_base = self.y[nn_index]

        # kernel val
        log_kernel_vals = self.log_kernel(nn_X_base, X_query.unsqueeze(1))

        # pred || y_pr, aleatoric, epistemic
        pred = torch.zeros(size=(len(X_query), 3)).to(X_query.device)

        for l in range(len(log_kernel_vals)):

            # Compute denominator (it is the same for all classes)
            log_denominator = torch.log(torch.sum(torch.exp(log_kernel_vals[l])) + 1)

            # Compute the weights
            weights = torch.exp(log_kernel_vals[l] - log_denominator)
            weights_bias = torch.exp(- log_denominator)

            # Apply the weights to compute the estimates
            y_pr = torch.sum(nn_y_base[l] * weights, dim=0) + self.y_mean_ * weights_bias

            # Compute the aleatoric and epistemic uncertainties
            y2_pr = torch.sum(nn_y_base[l] ** 2 * weights, dim=0) + self.y2_mean_ * weights_bias
            log_variance = torch.log(y2_pr - y_pr ** 2)
            log_epistemic = log_variance - log_denominator

            pred[l, 0] += y_pr
            pred[l, 1] += log_variance
            pred[l, 2] += log_epistemic

        return pred
