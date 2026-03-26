import numpy as np
from .activations import relu, relu_derivative, sigmoid
from .loss import binary_cross_entropy
from .utils import initialize_parameters

class NeuralNetwork:
    def __init__(self, n_x, n_h, n_y, lr=0.1):
        self.W1, self.b1, self.W2, self.b2 = initialize_parameters(n_x, n_h, n_y)
        self.lr = lr

    def forward(self, X):
        self.Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = relu(self.Z1)

        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = sigmoid(self.Z2)

        return self.A2

    def compute_loss(self, Y):
        return binary_cross_entropy(self.A2, Y)

    def compute_accuracy(self, X, Y):
        A2 = self.forward(X)
        preds = (A2 > 0.5).astype(int)
        return (preds == Y).mean()

    def backward(self, X, Y):
        m = X.shape[1]

        dZ2 = self.A2 - Y
        dW2 = (1/m) * np.dot(dZ2, self.A1.T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * relu_derivative(self.Z1)
        dW1 = (1/m) * np.dot(dZ1, X.T)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

        return dW1, db1, dW2, db2

    def update(self, dW1, db1, dW2, db2):
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, Y, epochs=1000):
        losses = []
        accuracies = []

        for i in range(epochs):
            self.forward(X)
            loss = self.compute_loss(Y)

            dW1, db1, dW2, db2 = self.backward(X, Y)
            self.update(dW1, db1, dW2, db2)

            acc = self.compute_accuracy(X, Y)

            losses.append(loss)
            accuracies.append(acc)

            if i % 100 == 0:
                print(f"Epoch {i}, Loss: {loss:.4f}, Acc: {acc:.4f}")

        return losses, accuracies