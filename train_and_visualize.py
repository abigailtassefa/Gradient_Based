import numpy as np
import matplotlib.pyplot as plt

from data import X_lin, y_lin, X_nn, y_nn, X_nn_norm
import linear_regression as lr_model
import neural_network as nn_model

# ---------- Train Linear Regression ----------
w, b, lin_losses = lr_model.train_linear_regression(X_lin, y_lin, lr=0.01, epochs=500)
print(f"Linear Regression -> w: {w:.3f}, b: {b:.3f}")

# ---------- Train NN with 3 learning rates ----------
learning_rates = [0.001, 0.01, 0.1]
nn_results = {}

for rate in learning_rates:
    W1, b1, W2, b2, losses = nn_model.train_nn(X_nn_norm, y_nn, hidden_size=8, lr=rate, epochs=500)
    nn_results[rate] = losses

# ---------- Plot 1: Regression line ----------
plt.figure()
plt.scatter(X_lin, y_lin, label="Data")
plt.plot(X_lin, lr_model.predict(X_lin, w, b), color="red", label="Fitted line")
plt.title("Linear Regression Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig("linear_fit.png")

# ---------- Plot 2: LR loss curve ----------
plt.figure()
plt.plot(lin_losses)
plt.title("Linear Regression Loss over Iterations")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.savefig("linear_loss.png")

# ---------- Plot 3: NN learning rate comparison ----------
plt.figure()
for rate, losses in nn_results.items():
    plt.plot(losses, label=f"lr={rate}")
plt.title("Neural Network Loss for Different Learning Rates")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.savefig("nn_lr_comparison.png")

plt.show()