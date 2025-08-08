# Stochastic Gradient Descent for Linear Regression

This repository contains an implementation of **Mini-Batch Gradient Descent (MBGD)** from scratch using NumPy for training a simple linear regression model.

## 📂 File Structure

- `stochastic_gradient_descent.ipynb`: Main Jupyter Notebook demonstrating the implementation and training of a linear regression model using MBGD.

## 🚀 Features

- Custom implementation of MBGD (`MBGDRegressor`) without using scikit-learn.
- Batch-based updates using randomly sampled mini-batches.
- Handles any number of features in the input data.
- Simple `fit` and `predict` API similar to `sklearn`.

## 🧠 How It Works

1. **Initialization**:
   - Weights (`coef_`) are initialized using `np.ones`.
   - Bias (`intercept_`) is initialized as `0`.

2. **Mini-Batch Training Loop**:
   - For each epoch:
     - Random mini-batches are selected.
     - Predictions are made using `y_hat = X @ coef_ + intercept_`.
     - Coefficients and intercept are updated using gradient descent.

3. **Prediction**:
   - Uses the learned weights to make predictions on test data.

## 🛠️ Requirements

- Python 3.x
- NumPy

Install dependencies (if not already installed):

```bash
pip install numpy
