---
title: Sigmoid Function and Fixing the Perceptron Trick
sidebar_position: 5
---

This note revisits the Perceptron Trick, highlights its core flaw, and introduces the **Sigmoid** function as the key ingredient that lets every point influence learning—bringing us closer to true Logistic Regression.

## The Perceptron Trick (recap) and its flaw
- Loop many times, pick a random point, update only if it is misclassified.
- If a point is correctly classified, coefficients don’t change.
- Stopping rule: once all training points are correctly classified, the algorithm stops.

Why this is a problem:
- It can stop at a boundary that is very close to one class (asymmetric margin).
- The found line separates training points but isn’t necessarily the **best** line for generalisation.
- Compared to proper Logistic Regression, the perceptron boundary can overfit and perform worse on test data.

## Learning from all points (push/pull idea)
We want both misclassified and correctly classified points to influence the boundary.

- Misclassified points should **pull** the line toward them.
- Correctly classified points should **push** the line **away** from them.
- The magnitude should depend on distance: near the boundary → stronger effect; far away → weaker effect.

## The update rule and the `Y_predicted` issue
Perceptron-style unified update:

`W_new = W_old + eta * (Y_true - Y_predicted) * X_i`

- In the perceptron, `Y_predicted` is produced by a step function → it is exactly `0` or `1`.
- For correctly classified points, `(Y_true - Y_predicted) = 0`, so no update happens.
- To let every point contribute, we need `Y_predicted` to be **continuous**, not just `0/1`.

## Enter the Sigmoid
Define a continuous mapping from any real `Z` to `(0, 1)`:

`sigma(Z) = 1 / (1 + exp(-Z))`

- If `Z >> 0` → `sigma(Z) → 1`
- If `Z << 0` → `sigma(Z) → 0`
- If `Z = 0` → `sigma(Z) = 0.5`

Use the model score `Z = W · X` (with augmented bias) and set:

`Y_predicted = sigma(Z)`

This yields a **probability** view:
- `Y_predicted` is the probability of the positive class.
- Decision boundary is where `sigma(Z) = 0.5` (i.e., `Z = 0`).
- The whole space gets a **probability gradient**, not just hard 0/1 labels.

## How Sigmoid fixes the dynamics
With `Y_predicted ∈ (0, 1)`, the term `(Y_true - Y_predicted)` rarely becomes exactly zero.

Examples:
- Correct positive, confident: `Y_true = 1`, `Y_predicted = 0.8` → term `= 0.2` → push boundary away from that point.
- Misclassified negative: `Y_true = 0`, `Y_predicted = 0.65` → term `= -0.65` → pull boundary toward that point.

Distance sensitivity emerges naturally:
- Points near the boundary have `sigma(Z)` close to `0.5`, so `(Y_true - Y_predicted)` is larger in magnitude → stronger push/pull.
- Far-away points have `sigma(Z)` very close to `0` or `1`, so the term is small → weaker effect.

## Updated learning loop (pseudo‑Python)
```python
import numpy as np

def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))

def train_sigmoid_perceptron(X, y, epochs=1000, eta=0.01, seed=42):
	# X: (n_samples, n_features). We augment with a bias column of ones.
	rng = np.random.default_rng(seed)
	X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
	W = np.zeros(X_aug.shape[1])

	for _ in range(epochs):
		i = rng.integers(0, X_aug.shape[0])
		Xi, yi = X_aug[i], y[i]
		z = W @ Xi
		y_pred = sigmoid(z)
		# continuous update: every point contributes
		W = W + eta * (yi - y_pred) * Xi
	return W
```

Notes:
- You can still form a hard class decision via a threshold: predict `1` if `sigma(Z) >= 0.5`, else `0`.
- But the update uses the **continuous** probability to keep refining the boundary.

## Where this gets us (and what remains)
- This approach is a big improvement over the step‑based perceptron: all points contribute, with distance‑weighted magnitude.
- However, it’s still not the full **Logistic Regression** objective used by libraries like scikit‑learn, which minimise a **logistic (cross‑entropy) loss** with sound optimisation and optional regularisation.

## Key takeaways
1. Perceptron stops as soon as training points are perfectly separated → can yield a poor margin.
2. Make every point matter: misclassified points pull; correctly classified points push.
3. Replace the step with **sigmoid** so `Y_predicted ∈ (0, 1)` and updates don’t vanish.
4. Magnitude of updates naturally depends on distance via the sigmoid output.
5. For production, train true Logistic Regression (cross‑entropy + regularisation); this sigmoid‑based update is a stepping stone.

