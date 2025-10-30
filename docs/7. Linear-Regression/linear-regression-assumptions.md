---
sidebar_position: 5
title: Linear Regression — Assumptions and Diagnostics
---

Linear Regression relies on several classical assumptions. Understanding and checking them helps you build reliable models and interpret coefficients correctly.

## The 5 key assumptions

### 1) Linearity (feature to target)

- Each input feature should relate to the target approximately linearly (or be linearisable via transformations).
- Positive linear: as X increases, Y increases roughly in a straight line. Negative linear: as X increases, Y decreases linearly.
- Strongly non‑linear relationships (for example, quadratic) violate this unless you engineer features (polynomial or transforms).

How to check

- Scatter plots: plot each feature versus target and look for straight‑line trends.
- Residuals vs fitted plot: after fitting, plot predicted values on X‑axis and residuals on Y‑axis; residuals should be centered around zero without a curve.

### 2) No multicollinearity (features not strongly correlated to each other)

- Linear regression estimates the marginal effect of each feature holding others constant. If features move together, unique effects become unstable and standard errors inflate.

How to check

- Variance Inflation Factor (VIF): values near 1 indicate low collinearity; above about 5 suggests a problem (some use 10 as a looser threshold).
- Correlation heatmap: low off‑diagonal correlations are reassuring; high values hint at collinearity.

### 3) Normality of residuals (for inference)

- Residuals (errors) $e_i = y_i - \hat{y}_i$ should be approximately normal if you want valid t‑tests and confidence intervals for coefficients.
- For pure prediction, modest deviations from normality are less critical.

How to check

- Histogram or kernel density plot of residuals: bell‑shaped around zero.
- Q‑Q plot: points should lie near the diagonal.

### 4) Homoscedasticity (constant variance of residuals)

- The spread of residuals should be roughly constant across the range of fitted values. If variance grows or shrinks systematically, that is heteroscedasticity.

How to check

- Residuals vs fitted plot: look for equal scatter. Funnel or cone shapes indicate heteroscedasticity.
- Statistical tests: Breusch–Pagan or White tests.

### 5) No autocorrelation in residuals (independence)

- Residuals should not follow patterns across observations. In time series, adjacent errors often correlate; plain OLS then underestimates uncertainty.

How to check

- Plot residuals in observation order (or time): should look like white noise.
- Durbin–Watson statistic: values near 2 suggest no first‑order autocorrelation (less than 2 implies positive, greater than 2 implies negative autocorrelation).

---

## Quick Python diagnostics

Below are minimal examples to fit a model and run common checks. Replace `X` (2D array or DataFrame) and `y` (1D array) with your data.

```python
# Fit with scikit-learn
from sklearn.linear_model import LinearRegression
import numpy as np

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
residuals = y - y_pred
```

### 1) Linearity and homoscedasticity plots

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(5,4))
sns.scatterplot(x=y_pred, y=residuals, s=20)
plt.axhline(0.0, color="red", linestyle="--", linewidth=1)
plt.xlabel("Fitted values (y_hat)")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted")
plt.tight_layout()
```

### 2) Normality of residuals

```python
import scipy.stats as stats
import matplotlib.pyplot as plt

# Histogram and KDE
plt.figure(figsize=(5,4))
plt.hist(residuals, bins=30, density=True, alpha=0.6)
stats.gaussian_kde(residuals)(np.linspace(min(residuals), max(residuals), 200))
plt.title("Residuals distribution")
plt.tight_layout()

# Q-Q plot
import statsmodels.api as sm
sm.qqplot(residuals, line="45")
plt.title("Q-Q plot of residuals")
plt.tight_layout()
```

### 3) Multicollinearity (VIF)

```python
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# If X is a numpy array, wrap it in a DataFrame to name columns
X_df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])

vif = pd.Series(
	[variance_inflation_factor(X_df.values, i) for i in range(X_df.shape[1])],
	index=X_df.columns,
)
print("VIF by feature:\n", vif)
```

### 4) Homoscedasticity test (Breusch–Pagan)

```python
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

X_const = sm.add_constant(X)
ols = sm.OLS(y, X_const).fit()
bp_test = het_breuschpagan(ols.resid, ols.model.exog)
labels = ["LM statistic", "LM p-value", "F statistic", "F p-value"]
print(dict(zip(labels, bp_test)))
```

### 5) Autocorrelation (Durbin–Watson)

```python
from statsmodels.stats.stattools import durbin_watson

dw = durbin_watson(residuals)
print("Durbin–Watson:", dw)
```

---

## What to do when assumptions fail

- Linearity issues: engineer features (polynomial terms, logs), or try more flexible models.
- Multicollinearity: drop or combine correlated features, apply regularisation (Ridge or Lasso), or use PCA.
- Non‑normal residuals: check for outliers and skew; transform target or features if you need valid inference.
- Heteroscedasticity: transform variables (log), model variance (for example, Weighted Least Squares), or use robust standard errors.
- Autocorrelation: use time‑series aware models (ARIMA, SARIMAX) or add lagged predictors; consider GLS.

---

## Handy equations

- Residuals: $e_i = y_i - \hat{y}_i$.
- OLS objective (for n samples): $\text{MSE} = \tfrac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$.

These diagnostics help you validate the fit, trust your inferences, and guide corrective actions when assumptions do not hold.

