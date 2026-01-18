import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def soft_bent(x):
    return x / (1 + np.abs(x))

def soft_bent_derivative(x):
    return 1 / (1 + np.abs(x))**2

X, y = load_iris(return_X_y=True)
y = y.reshape(-1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def train_network(activation, activation_derivative, epochs=200):
    np.random.seed(0)
    W1 = np.random.randn(4, 5)
    b1 = np.zeros((1, 5))
    W2 = np.random.randn(5, 1)
    b2 = np.zeros((1, 1))

    lr = 0.05
    losses = []

    for _ in range(epochs):
        z1 = np.dot(X_train, W1) + b1
        a1 = activation(z1)

        z2 = np.dot(a1, W2) + b2
        y_pred = activation(z2)

        loss = np.mean((y_train - y_pred) ** 2)
        losses.append(loss)

        d2 = (y_pred - y_train) * activation_derivative(y_pred)
        dW2 = np.dot(a1.T, d2)
        db2 = np.sum(d2, axis=0, keepdims=True)

        d1 = np.dot(d2, W2.T) * activation_derivative(a1)
        dW1 = np.dot(X_train.T, d1)
        db1 = np.sum(d1, axis=0, keepdims=True)

        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

    return losses

sigmoid_loss = train_network(sigmoid, sigmoid_derivative)
custom_loss = train_network(soft_bent, soft_bent_derivative)

plt.plot(sigmoid_loss, label="Sigmoid")
plt.plot(custom_loss, label="Soft-Bent (Custom)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Comparison")
plt.legend()
plt.show()

x = np.linspace(-5, 5, 200)
plt.plot(x, soft_bent(x))
plt.title("Custom Activation Function: Soft-Bent")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()
