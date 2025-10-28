---
title: Logistic Regression
sidebar_position: 2
---

Logistic Regression is one of the most important ML algorithms and a natural bridge to Deep Learning. Its building block, the Perceptron, is closely related to Logistic Regression, so understanding it pays dividends later.

## What is it?
- A linear classifier that models the probability of a class using the logistic (sigmoid) link.
- This note takes the probabilistic point of view rather than the geometric one.

## When it works best
- Works well when data is **linearly separable** or **almost linearly separable**.
- Linearly separable: classes can be split by a straight line (2D), plane (3D), or hyperplane (higher‑D).
- Almost separable: allow a few misclassifications, but a single boundary still explains most points.
- If classes are highly non‑linear, a single linear boundary will struggle → consider feature engineering or non‑linear models.

## The Perceptron Trick (intuition and simple learner)
Purpose: a simple approach to learn a separating line; it’s foundational for Deep Learning.

Notes:
- Simple and easy to implement, though not always the optimal logistic solution.
- Goal: learn coefficients `(A, B, C)` for the line `A*x + B*y + C = 0` that best separate the classes.

### Core loop
1. Initialise `(A, B, C)` randomly (or zeros) → an initial, likely poor line.
2. Repeat for several epochs or until convergence:
   - Pick a random training point `(X_i, Y_true)`.
   - Ask: is it correctly classified by the current line?
   - If misclassified, update the line to move toward the point if it’s positive, or away if it’s negative.

### Positive/negative regions for `A*x + B*y + C = 0`
- Point `(x1, y1)` is positive if `A*x1 + B*y1 + C > 0`.
- Negative if `A*x1 + B*y1 + C < 0`.
- On the line if `A*x1 + B*y1 + C = 0`.

### How changing coefficients moves the line
- Change `C` → parallel shift up/down.
- Change `A` → rotate around Y‑axis.
- Change `B` → rotate around X‑axis.

### Update rules with augmented coordinates
Use augmented inputs with `X0 = 1` and weights `W = [W0, W1, W2]` where `W0=C`, `W1=A`, `W2=B`.

- Case 1: Negative point (`Y_true = 0`) predicted positive (`Y_pred = 1`) → subtract: `W_new = W_old - eta * X_i`.
- Case 2: Positive point (`Y_true = 1`) predicted negative (`Y_pred = 0`) → add: `W_new = W_old + eta * X_i`.
  - `eta` is the learning rate (e.g., `0.01`).

### Unified update rule
Let `S = W·X_i` (dot product with augmented `X_i`). Predict `Y_pred = 1` if `S >= 0`, else `0`.

`W_new = W_old + eta * (Y_true - Y_pred) * X_i`

Behaviours:
- If correct: `Y_true == Y_pred` → no change.
- If positive misclassified (`1` vs `0`): add `eta * X_i`.
- If negative misclassified (`0` vs `1`): subtract `eta * X_i`.

### Convergence and stopping
- Stop when there are no misclassified points (convergence) or after a fixed number of epochs.

## Perceptron‑style training loop (pseudo‑Python)
```python
import numpy as np

def train_perceptron(X, y, epochs=1000, eta=0.01, seed=42):
    # X: shape (n_samples, 2) for features [x, y]; we'll augment inside
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

## Key takeaways
1. Logistic Regression is central to ML and underpins Perceptron/Deep Learning ideas.
2. It works best when data is (almost) linearly separable; a single linear boundary is assumed.
3. The Perceptron Trick offers a simple learning procedure with intuitive add/subtract updates.
4. Use augmented coordinates and the unified rule `W_new = W_old + eta * (Y_true - Y_pred) * X_i`.
