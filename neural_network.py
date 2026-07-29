import numpy as np

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def init_params(hidden_size=8):
    np.random.seed(1)
    W1 = np.random.randn(hidden_size) * 0.1
    b1 = np.zeros(hidden_size)
    W2 = np.random.randn(hidden_size) * 0.1
    b2 = 0.0
    return W1, b1, W2, b2

def forward(X, W1, b1, W2, b2):
    # X shape: (n,)  -> z1 shape: (n, hidden_size)
    z1 = np.outer(X, W1) + b1        # linear layer 1
    a1 = relu(z1)                    # nonlinearity
    y_hat = a1 @ W2 + b2             # linear layer 2 (output)
    return z1, a1, y_hat

def mse_loss(y_true, y_pred):
    n = len(y_true)
    return np.sum((y_true - y_pred) ** 2) / n

def backward(X, y_true, z1, a1, y_hat, W2):
    n = len(y_true)

    dL_dyhat = (-2 / n) * (y_true - y_hat)        # dL/d(y_hat), shape (n,)

    dW2 = a1.T @ dL_dyhat                         # dL/dW2
    db2 = np.sum(dL_dyhat)                        # dL/db2

    da1 = np.outer(dL_dyhat, W2)                  # dL/da1, shape (n, hidden)
    dz1 = da1 * relu_derivative(z1)               # dL/dz1

    dW1 = X.T @ dz1                               # dL/dW1
    db1 = np.sum(dz1, axis=0)                     # dL/db1

    return dW1, db1, dW2, db2

def train_nn(X, y, hidden_size=8, lr=0.01, epochs=500):
    W1, b1, W2, b2 = init_params(hidden_size)
    losses = []

    for epoch in range(epochs):
        z1, a1, y_hat = forward(X, W1, b1, W2, b2)
        loss = mse_loss(y, y_hat)
        losses.append(loss)

        dW1, db1, dW2, db2 = backward(X, y, z1, a1, y_hat, W2)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    return W1, b1, W2, b2, losses