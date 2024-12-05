from collections import defaultdict
from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from nuq.kernels import *
import time

class NuqClassifier():
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
            X: np.ndarray,
            y: np.ndarray,
            bandwidth: Optional[np.ndarray] = 1.5404414530745685,
    ):
        self.X = torch.from_numpy(X).cuda()
        self.y = torch.from_numpy(y).cuda()
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

        # pred || cls, prob, uncertainty
        pred = torch.zeros(size=(len(X_query), 3)).to(X_query.device)
        for l in range(len(log_kernel_vals)):
            classes_cur = torch.unique(nn_y_base[l])
            # Get positions for each class
            # indices = defaultdict(list)
            # for i, v in enumerate(encoded):
            #     indices[v].append(i)
            log_ps_cur = torch.zeros(size=(len(classes_cur), 1)).to(X_query.device)
            for k in range(len(classes_cur)):
                log_ps_cur[k] += torch.log(torch.sum(torch.exp(log_kernel_vals[l, nn_y_base[l] == classes_cur[k]])))
            log_ps_total_cur = torch.log(torch.sum(torch.exp(
                torch.cat([
                    log_ps_cur,
                    self.log_pN + self.log_prior[classes_cur.long()].unsqueeze(1),
                    ],
                    dim=1)
            ), dim=1))
            # Compute denominator (it is the same for all classes)
            log_denominator = torch.log(torch.sum(torch.exp(log_ps_cur)) + 1)
            # Select class with top probability
            idx_max = torch.argmax(log_ps_total_cur)
            # If max probability is greater than all prior probabilities,
            # predict the corresponding class
            if log_ps_total_cur[idx_max] > self.log_prior_default_:
                class_pred = classes_cur[idx_max]
                log_numerator_p = log_ps_total_cur[idx_max]
            # If max probability is still less than any prior probability
            # then just predict the top prior class
            else:
                class_pred = self.class_default_
                log_numerator_p = self.log_prior_default_
            # Compute the Nadaraya-Watson estimator
            log_ps_pred = log_numerator_p - log_denominator

            # Uncertainty prediction has the same two cases as probability
            # prediction. By default, the numerator is given by p*(1-p)
            # For convenience, (1-p) is denoted with _1mp
            if log_ps_total_cur[idx_max] > self.log_prior_default_:
                temp = torch.cat([log_ps_cur[:idx_max], log_ps_cur[idx_max + 1:], ])
                log_numerator_1mp = self.log_pN + torch.log1p(-self.log_prior[idx_max])
                if len(temp) > 0:
                    log_numerator_1mp += torch.log(torch.sum(torch.exp(temp)))
            else:
                log_numerator_1mp = self.log_pN + torch.log1p(-self.log_prior[idx_max])
                log_numerator_1mp += torch.log(torch.sum(torch.exp(log_ps_cur)))
            log_uncertainty_total = (
                    log_numerator_p + log_numerator_1mp - 3 * log_denominator
            )
            pred[l, 0] += class_pred
            pred[l, 1] += log_ps_pred
            pred[l, 2] += log_uncertainty_total

        return pred