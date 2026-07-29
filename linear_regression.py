import numpy as np

def predict(X, w, b):
    # y_hat = w*x + b
    return w * X + b

def mse_loss(y_true, y_pred):
    n = len(y_true)
    return np.sum((y_true - y_pred) ** 2) / n

def compute_gradients(X, y_true, y_pred):
    n = len(y_true)
    dw = (-2 / n) * np.sum(X * (y_true - y_pred))
    db = (-2 / n) * np.sum(y_true - y_pred)
    return dw, db

def train_linear_regression(X, y, lr=0.01, epochs=500):
    w, b = 0.0, 0.0
    losses = []

    for epoch in range(epochs):
        y_pred = predict(X, w, b)
        loss = mse_loss(y, y_pred)
        losses.append(loss)

        dw, db = compute_gradients(X, y, y_pred)
        w -= lr * dw
        b -= lr * db

    return w, b, losses