---
title: Logistic Regression — Perceptron Trick and Implementation
sidebar_position: 3
---

Logistic Regression is a cornerstone classification algorithm and a natural bridge to Deep Learning. The Perceptron Trick offers a simple way to learn a linear decision boundary and is closely related to Logistic Regression.

This page focuses on the probabilistic view and then walks through a Perceptron‑style training loop, its unified update rule, and why the official Logistic Regression approach is preferred in practice.

## Core requirement: linear separability
- Works best when data is **linearly separable** or **almost linearly separable**.
- Linearly separable: a single straight line (2D), plane (3D), or hyperplane (higher‑D) splits classes.
- Almost separable: allow a few misclassifications but a single boundary explains most points.
- Highly non‑linear class structure → a single linear boundary will underperform without feature engineering.

## The Perceptron Trick (simple learner)
Goal: find coefficients `(A, B, C)` of the line `A*x + B*y + C = 0` that separate the classes.

High‑level loop:
1. Initialise `(A, B, C)` randomly (or zeros) → initial poor line.
2. For many epochs: pick a random training point `(X_i, Y_true)`.
3. If the point is misclassified, update the line to move toward positive points and away from negative points.

### Positive/negative regions for `A*x + B*y + C = 0`
- `(x1, y1)` is positive if `A*x1 + B*y1 + C > 0`.
- Negative if `A*x1 + B*y1 + C < 0`.
- On the line if `A*x1 + B*y1 + C = 0`.

### How coefficients move the line
- Change `C` → parallel shift up/down.
- Change `A` → rotation around Y‑axis.
- Change `B` → rotation around X‑axis.

### Augmented coordinates and case updates
Use augmented inputs `X = [X0, X1, X2]` with `X0 = 1` and weights `W = [W0, W1, W2]` where `W0=C`, `W1=A`, `W2=B`.

- Case 1: Negative point (`Y_true = 0`) predicted positive (`Y_pred = 1`)
	- Update: `W_new = W_old - eta * X_i`
- Case 2: Positive point (`Y_true = 1`) predicted negative (`Y_pred = 0`)
	- Update: `W_new = W_old + eta * X_i`
- `eta` is the learning rate (e.g., `0.01`).

### Unified update rule (concise)
Let `S = W · X_i`. Predict `Y_pred = 1` if `S >= 0`, else `0`.

`W_new = W_old + eta * (Y_true - Y_pred) * X_i`

Behaviour:
- Correct classification → no change (term becomes `0`).
- Positive misclassified → add `eta * X_i`.
- Negative misclassified → subtract `eta * X_i`.

### Convergence and stopping
- Stop when there are no misclassified points (convergence), or after a fixed number of epochs.
- The Perceptron stops as soon as it perfectly classifies the training set; it does not try to "improve the margin" further.

## Perceptron‑style training loop (pseudo‑Python)
```python
import numpy as np

def train_perceptron(X, y, epochs=1000, eta=0.01, seed=42):
		# X: shape (n_samples, 2) for features [x, y]; we augment inside
		rng = np.random.default_rng(seed)
		X_aug = np.hstack([np.ones((X.shape[0], 1)), X])  # add X0 = 1
		W = np.zeros(X_aug.shape[1])  # [C, A, B]

		for _ in range(epochs):
				i = rng.integers(0, X_aug.shape[0])
				Xi, yi = X_aug[i], y[i]
				S = W @ Xi
				y_pred = 1 if S >= 0 else 0
				W = W + eta * (yi - y_pred) * Xi
		return W
```

## Why the Perceptron Trick is suboptimal
- **Stops too early:** halts once all training points are classified, even if the line is very close to one class (small/asymmetric margin).
- **Generalisation risk:** a boundary that hugs one class can overfit; test performance may suffer.
- **No probabilistic output:** pure perceptron uses a hard threshold rather than probabilities.

## Logistic Regression (what we actually train in practice)
Logistic Regression learns a weight vector `W` and bias `b` to model a probability:

`p(y=1 | x) = sigmoid(W · x + b)`  where `sigmoid(z) = 1 / (1 + exp(-z))`

It optimises a smooth **logistic (cross‑entropy) loss** rather than a 0/1 misclassification rule, typically via gradient descent/solvers. This encourages a boundary that better balances the classes and improves generalisation. Regularisation (L2/L1/Elastic Net) is commonly added.

### Minimal scikit‑learn example
```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Scale features first; saga/liblinear solvers are common choices
logreg = make_pipeline(
		StandardScaler(),
		LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
)

logreg.fit(X_train, y_train)
print("Validation accuracy:", logreg.score(X_val, y_val))
```

## Key takeaways
1. Perceptron Trick is a great teaching tool: simple updates, clear geometry.
2. Unified update: `W_new = W_old + eta * (Y_true - Y_pred) * X_i` with augmented inputs.
3. It can perfectly separate training data but may yield a poor margin and generalisation.
4. Logistic Regression optimises a probabilistic loss and typically generalises better; add regularisation when needed.

