from sklearn.datasets import fetch_openml
import sklearn
import numpy as np

def load_mnist():
    X, Y = fetch_openml('mnist_784', version=1, return_X_y=True)

    X = X.to_numpy()
    Y = Y.astype(int).to_numpy()

    X = X / 255.0
    Y = (Y == 0).astype(int)

    X = X.T
    Y = Y.reshape(1, -1)

    return X, Y