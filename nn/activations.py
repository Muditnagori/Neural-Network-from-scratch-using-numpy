import numpy as np

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return Z > 0

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))