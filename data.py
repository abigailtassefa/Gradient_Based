import numpy as np

np.random.seed(42)

n_samples = 30

# Linear regression data: y = true_m * x + true_b + noise
X_lin = np.linspace(0, 10, n_samples)
true_m, true_b = 2.5, 5
noise_lin = np.random.normal(0, 2, n_samples)
y_lin = true_m * X_lin + true_b + noise_lin

# Neural network data: y = x^2 + noise
X_nn = np.linspace(-5, 5, n_samples)
noise_nn = np.random.normal(0, 3, n_samples)
y_nn = X_nn**2 + noise_nn

# Normalize NN input for stable training
X_nn_norm = (X_nn - X_nn.mean()) / X_nn.std()