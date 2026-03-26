import matplotlib.pyplot as plt
from nn.model import NeuralNetwork
from data.mnist_data import load_mnist

X, Y = load_mnist()

# Use subset (important)
X = X[:, :5000]
Y = Y[:, :5000]

model = NeuralNetwork(n_x=784, n_h=32, n_y=1, lr=0.1)

losses, accs = model.train(X, Y, epochs=500)

print("Final Accuracy:", model.compute_accuracy(X, Y))

plt.figure()
plt.plot(losses)
plt.title("Loss Curve (MNIST)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid()
plt.show()

plt.figure()
plt.plot(accs)
plt.title("Accuracy Curve (MNIST)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid()
plt.show()