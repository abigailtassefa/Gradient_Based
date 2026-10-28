# Gradient-Based Optimization & Neural Networks from Scratch

A Python-based implementation exploring optimization, gradient descent mechanics, and supervised learning algorithms from first principles without relying on high-level machine learning frameworks.

---

## Overview

This repository demonstrates the foundational mathematics and mechanics behind machine learning model training. It provides ground-up implementations of both **Linear Regression** and a **Shallow Neural Network**, evaluating how gradient descent updates parameters, minimizes loss functions, and responds to varying learning rates.

---

## Features

* **Linear Regression from Scratch:** Direct implementation of parameters initialization, linear hypothesis evaluation, mean squared error (MSE) loss computation, and iterative gradient descent parameter updates.
* **Shallow Neural Network Architecture:** Built-in forward pass, activation functions, loss computation, and manual backpropagation to calculate parameter gradients across layers.
* **Learning Rate Dynamics:** Comparative experiments evaluating model convergence and stability across different learning rates.
* **Visual Analytics:** Automated generation and saving of performance plots, including loss landscapes, linear fitting curves, and comparative loss curves over training epochs.

---

## Repository Structure

```text
Gradient_Based/
├── data.py                 # Synthetic/sample dataset generation and preprocessing
├── linear_regression.py    # Linear Regression model class and gradient descent implementation
├── neural_network.py       # Shallow Neural Network architecture with backpropagation
├── train_and_visualize.py  # Training pipeline, learning rate comparisons, and plot rendering
├── linear_fit.png          # Visual plot: Fitted line vs. ground truth data
├── linear_loss.png         # Visual plot: Loss convergence curve for Linear Regression
├── nn_loss.png             # Visual plot: Loss convergence curve for Neural Network
├── nn_lr_comparison.png    # Visual plot: Neural Network loss across different learning rates
├── Report - Abigail T.pdf  # Project documentation detailing derivations and findings
└── README.md               # Project documentation

```

---

## Getting Started

### Prerequisites

Ensure you have Python 3.8 or higher installed along with standard scientific computing libraries:

```bash
pip install numpy matplotlib

```

### Installation & Execution

1. **Clone the repository:**
```bash
git clone https://github.com/abigailtassefa/Gradient_Based.git
cd Gradient_Based

```


2. **Run the training and visualization pipeline:**
```bash
python train_and_visualize.py

```


*Running the script executes model training and updates generated visual plots (`*.png`) in the root directory.*
