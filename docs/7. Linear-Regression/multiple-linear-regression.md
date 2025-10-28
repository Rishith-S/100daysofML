---
sidebar_position: 4
---

# Multiple Linear Regression

When there are multiple input features (columns), instead of fitting a line (in 2D) we fit a hyperplane that best explains the target. In 3D this looks like a plane splitting the space; in higher dimensions it’s a hyperplane.

## Model Definition

- Scalar form (with k features):
```
ŷ = β₀ + β₁x₁ + β₂x₂ + … + β_k x_k
```

- Matrix form (for n samples and k features):
```
y = Xβ + ε
```
Where:
- `X ∈ ℝ^{n×(k+1)}` is the design matrix (first column of ones for intercept)
- `β ∈ ℝ^{(k+1)×1}` are the parameters `[β₀, β₁, …, β_k]^T`
- `y ∈ ℝ^{n×1}` is the target vector
- `ε ∈ ℝ^{n×1}` is the noise (residuals)

Predictions and residuals:
```
ŷ = Xβ
e = y − ŷ
```

## Objective (Least Squares)

Ordinary Least Squares chooses β to minimize the sum of squared residuals:
```
SSE(β) = e^T e = (y − Xβ)^T (y − Xβ)
MSE(β) = (1/n) · e^T e
```

## Closed-Form Solution (Normal Equation)

If `X^T X` is invertible:
```
β* = (X^T X)^{-1} X^T y
```
Notes:
- Use a pseudoinverse (or `np.linalg.lstsq`) if `X^T X` is singular or ill-conditioned.
- Add an intercept by prepending a column of ones to `X`.

## Gradient Descent (Iterative)

When k or n is large, we often optimize iteratively:
```
Initialize β
Repeat until convergence:
	β ← β − η · (2/n) · X^T (Xβ − y)
```
Where `η` is the learning rate. Variants include stochastic and mini-batch gradient descent.

## Assumptions (Classical Ordinary Least Squares)

1. Linearity: relationship between features and target is linear in parameters.
2. Independence: residuals are independent.
3. Homoscedasticity: constant variance of residuals.
4. Normality: residuals are normally distributed (primarily for inference).
5. No multicollinearity: features aren’t perfectly collinear (helps stability).

Violations can degrade estimates or inflate variance. Feature engineering and diagnostics help detect and mitigate issues.

## Practical Tips

- Scale features when using gradient-based methods to improve convergence.
- Check multicollinearity (e.g., with VIF) and remove/regularize correlated features.
- Consider regularization (Ridge/Lasso/Elastic Net) when overfitting or collinearity is present.
- Evaluate with metrics such as MAE, MSE/RMSE, R², and Adjusted R² (see the metrics page).

## Small Examples

### Numpy: closed-form via least squares (adds intercept)

```python
import numpy as np

# Example data: n samples, k features
X = np.array([
		[1.0, 2100, 3],
		[1.0, 1600, 2],
		[1.0, 2400, 3],
		[1.0, 1416, 2],
		[1.0, 3000, 4],
])
y = np.array([399900, 329900, 369000, 232000, 539900])

# β by least squares (more stable than explicit inverse)
beta, *_ = np.linalg.lstsq(X, y, rcond=None)

print("β (intercept first):", beta)
pred = X @ beta
residuals = y - pred
SSE = residuals.T @ residuals
print("SSE:", SSE)
```

### Scikit-learn: Ordinary Least Squares

```python
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([
		[2100, 3],
		[1600, 2],
		[2400, 3],
		[1416, 2],
		[3000, 4],
])
y = np.array([399900, 329900, 369000, 232000, 539900])

model = LinearRegression(fit_intercept=True)
model.fit(X, y)

print("intercept:", model.intercept_)
print("coefficients:", model.coef_)
print("R^2:", model.score(X, y))
```

## Quick Geometry Intuition

- Simple linear regression fits a line in 2D.
- Multiple linear regression fits a hyperplane in higher-dimensional feature space.
- OLS finds the β that minimizes the squared vertical distances from points to this hyperplane.

## Key Equations Recap

```
ŷ = Xβ
e = y − ŷ
SSE = e^T e
β* = (X^T X)^{-1} X^T y   (use pseudoinverse/least squares in practice)
```

