import numpy as np


def rbf(bandwidth=1.0):
    return lambda X, Y: -np.sum((((X - Y) / bandwidth) ** 2) / 2, axis=-1)


def student(bandwidth=1.0):
    return lambda X, Y: -np.log1p(
        np.sum((((X - Y) / bandwidth) ** 2) / 2, axis=-1)
    )
