import numpy as np
import torch


# class RBF_grad():
#     def __init__(self, bandwidth):
#         self.bandwidth = np.min(bandwidth)
#
#     def __call__(self, X, Y):
#         return -torch.sum((((X - Y) / self.bandwidth) ** 2) / 2, dim=-1)

class RBF_grad:
    def __init__(self, bandwidth):
        """
        Initialize the RBF kernel with bandwidth.
        Args:
            bandwidth (float or array-like): The bandwidth parameter for the RBF kernel.
        """
        if isinstance(bandwidth, (list, np.ndarray)):
            self.bandwidth = np.min(bandwidth)  # 取最小值作为全局带宽
        else:
            self.bandwidth = bandwidth

    def __call__(self, X, Y):
        """
        Compute the gradient of the RBF kernel.
        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, features).
            Y (torch.Tensor): Reference tensor of shape (batch_size, features).
        Returns:
            torch.Tensor: The gradient of the RBF kernel.
        """
        # 计算欧氏距离的平方
        diff = (X - Y) / self.bandwidth
        sq_dist = torch.sum(diff ** 2, dim=-1)

        # 返回核值 (未归一化的 RBF 核梯度形式)
        return -sq_dist / 2

def rbf(bandwidth=1.0, keep_grad=False):
    if keep_grad:
        return RBF_grad(bandwidth)
    return lambda X, Y: -np.sum((((X - Y) / bandwidth) ** 2) / 2, axis=-1)


def student(bandwidth=1.0):
    return lambda X, Y: -np.log1p(
        np.sum((((X - Y) / bandwidth) ** 2) / 2, axis=-1)
    )
