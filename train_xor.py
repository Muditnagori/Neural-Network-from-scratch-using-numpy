import matplotlib.pyplot as plt
from nn.model import NeuralNetwork
from data.xor_data import load_data

X, Y = load_data()

model = NeuralNetwork(n_x=2, n_h=4, n_y=1, lr=0.1)

losses, accs = model.train(X, Y, epochs=2000)

print("Final Accuracy:", model.compute_accuracy(X, Y))

plt.figure()
plt.plot(losses)
plt.title("Loss Curve (XOR)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid()
plt.show()

plt.figure()
plt.plot(accs)
plt.title("Accuracy Curve (XOR)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid()
plt.show()