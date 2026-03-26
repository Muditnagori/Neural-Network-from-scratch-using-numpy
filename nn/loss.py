import numpy as np

def binary_cross_entropy(A, Y):
    m = Y.shape[1]
    A = np.clip(A, 1e-8, 1 - 1e-8)
    return -(1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))