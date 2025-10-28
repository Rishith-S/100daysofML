---
sidebar_position: 3
---

# Evaluation Metrics for Linear Regression

Evaluation metrics are crucial for assessing the performance of linear regression models. Different metrics provide different perspectives on model accuracy and help in choosing the right model for specific scenarios.

## 1. Mean Absolute Error (MAE)

### Definition
MAE measures the average absolute difference between actual and predicted values.

### Formula
```
MAE = (1/n) × Σ|yi - ŷi|
```

Where:
- n = number of observations
- yi = actual values
- ŷi = predicted values

### Characteristics

#### Advantages:
- **Intuitive interpretation**: Easy to understand as average error
- **Robust to outliers**: Less sensitive to extreme values
- **Same units as target variable**: Directly interpretable

#### Disadvantages:
- **Not differentiable**: Cannot be used as a loss function for optimization
- **Less sensitive to large errors**: Treats all errors equally

### When to Use:
- **Data has many outliers**: MAE is more robust to extreme values
- **When you want equal penalty for all errors**: Regardless of magnitude
- **Interpretability is important**: Easy to explain to stakeholders

### Interpretation:
- **Lower MAE = Better model performance**
- MAE of 0 means perfect predictions

## 2. Mean Squared Error (MSE)

### Definition
MSE measures the average of squared differences between actual and predicted values.

### Formula
```
MSE = (1/n) × Σ(yi - ŷi)²
```

### Characteristics

#### Advantages:
- **Differentiable**: Can be used as a loss function for optimization
- **Penalizes large errors more**: Emphasizes the cost of big mistakes
- **Mathematically convenient**: Works well with calculus-based optimization

#### Disadvantages:
- **Different units than output**: Units are squared (e.g., if target is in dollars, MSE is in dollars²)
- **Harsh penalty on outliers**: Large errors are squared, making them much more significant
- **Less interpretable**: Harder to understand the actual error magnitude

### When to Use:
- **Data has few outliers**: MSE works well with normally distributed errors
- **Large errors are costly**: When you want to heavily penalize big mistakes
- **Using gradient-based optimization**: MSE is differentiable

### Interpretation:
- **Lower MSE = Better model performance**
- MSE of 0 means perfect predictions

## 3. Root Mean Squared Error (RMSE)

### Definition
RMSE is the square root of MSE, providing error measurement in the same units as the target variable.

### Formula
```
RMSE = √MSE = √[(1/n) × Σ(yi - ŷi)²]
```

### Characteristics

#### Advantages:
- **Same units as output**: Directly interpretable (e.g., if target is in dollars, RMSE is in dollars)
- **Differentiable**: Can be used for optimization
- **Penalizes large errors**: Similar to MSE but in correct units

#### Disadvantages:
- **Not robust to outliers**: A statistical method is heavily influenced by extreme values (outliers) in the data. A small change in the data caused by an outlier can drastically alter the results or predictions produced by the method.

### When to Use:
- **When you need interpretable units**: Same scale as your target variable
- **Balancing MSE benefits with interpretability**: Want MSE's properties but in correct units
- **Data has few outliers**: RMSE is sensitive to extreme values

### Interpretation:
- **Lower RMSE = Better model performance**
- RMSE of 0 means perfect predictions

## 4. R² Score (Coefficient of Determination)

### Definition
R² measures the proportion of variance in the dependent variable that is predictable from the independent variable(s).

### Formula
```
R² = 1 - (SSres / SStot)
```

Where:
- SSres = Σ(yi - ŷi)² (Sum of Squares of Residuals)
- SStot = Σ(yi - ȳ)² (Total Sum of Squares)
- ȳ = mean of actual values

### Interpretation:
- **R² = 0.8**: The model explains 80% of the variance in the target variable
- **R² = 1.0**: Perfect fit (all variance explained)
- **R² = 0.0**: Model performs as well as simply predicting the mean
- **R² < 0**: Model performs worse than predicting the mean

### Range:
- **0 ≤ R² ≤ 1** (for most cases)
- **R² can be negative** if the model is extremely poor

### When to Use:
- **Comparing models**: Easy to compare different models
- **Understanding model quality**: Intuitive percentage of variance explained
- **Feature selection**: Higher R² indicates better feature combinations

## 5. Adjusted R² Score

### Definition
Adjusted R² is a modified version of R² that accounts for the number of predictors in the model, preventing artificial inflation of R² when adding irrelevant features.

### Formula
```
Adjusted R² = 1 - [(1 - R²) × (n - 1) / (n - k - 1)]
```

Where:
- n = number of observations
- k = number of independent variables (features)

### Why Adjusted R² is Important:

#### Problem with R²:
- **R² always increases or stays the same** when adding new features
- **Even irrelevant features** can increase R² slightly
- **Overfitting risk**: Adding noise features can inflate R²

#### Solution with Adjusted R²:
- **Penalizes unnecessary features**: Decreases when adding irrelevant features
- **Prevents overfitting**: More honest assessment of model quality
- **Better model selection**: Choose model with highest adjusted R²

### When to Use:
- **Multiple regression models**: When comparing models with different numbers of features
- **Feature selection**: To avoid overfitting
- **Model comparison**: More reliable than R² for model selection

### Interpretation:
- **Higher Adjusted R² = Better model** (considering complexity)
- **Adjusted R² ≤ R²**: Always less than or equal to regular R²
- **Can be negative**: If model is very poor relative to its complexity

## Summary Table

| Metric | Units | Robust to Outliers | Differentiable | Best Use Case |
|--------|-------|-------------------|----------------|---------------|
| MAE | Same as target | Yes | No | Data with many outliers |
| MSE | Squared units | No | Yes | Few outliers, optimization |
| RMSE | Same as target | No | Yes | Interpretable units needed |
| R² | Unitless (0-1) | No | Yes | Model comparison |
| Adjusted R² | Unitless | No | Yes | Multiple regression comparison |

## Python Implementation Example

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def calculate_metrics(y_true, y_pred):
    """
    Calculate all regression metrics
    """
    # Basic metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Adjusted R²
    n = len(y_true)
    k = 1  # number of features (adjust as needed)
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2,
        'Adjusted R²': adjusted_r2
    }

# Example usage
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
metrics = calculate_metrics(y_true, y_pred)
print(metrics)
```

Choose the appropriate metric based on your specific use case, data characteristics, and business requirements.
