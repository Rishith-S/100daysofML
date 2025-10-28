---
sidebar_position: 2
---

# Simple Linear Regression

Simple linear regression establishes a linear relationship between one independent variable (x) and one dependent variable (y) using the equation:

**y = mx + b**

Where:
- **m** = weightage (slope/coefficient)
- **b** = offset (y-intercept)
- **x** = independent variable (input feature)
- **y** = dependent variable (predicted output)

## Objective

Our goal is to find the **best fit line** that minimizes the sum of squared errors between actual and predicted values.

## Methods to Find the Linear Regression Line

### 1. Closed Form Method (Direct Solution)

The closed form method provides an exact mathematical solution without requiring iterative approximation techniques.

#### Key Characteristics:
- **No differentiation and integration** required
- **Direct formula** approach
- **Scikit-learn uses this method** called **Ordinary Least Squares**

#### Mathematical Formulas:

**Slope (m):**
```
m = Σ(xi - x̄)(yi - ȳ) / Σ(xi - x̄)²
```

**Y-intercept (b):**
```
b = ȳ - m * x̄
```

Where:
- **x̄** = mean of x values
- **ȳ** = mean of y values
- **xi** = individual x values
- **yi** = individual y values

#### Step-by-Step Process:

1. **Calculate means:**
   - x̄ = (x₁ + x₂ + ... + xₙ) / n
   - ȳ = (y₁ + y₂ + ... + yₙ) / n

2. **Calculate slope (m):**
   - For each data point, calculate (xi - x̄) and (yi - ȳ)
   - Multiply these differences: (xi - x̄)(yi - ȳ)
   - Sum all these products
   - Calculate (xi - x̄)² for each point and sum them
   - Divide the first sum by the second sum

3. **Calculate y-intercept (b):**
   - Use the formula: b = ȳ - m * x̄

### 2. Non-Closed Form Method (Approximation Techniques)

This method uses iterative approximation techniques to find the solution when closed form solutions are not feasible or practical.

#### Common Techniques:
- **Gradient Descent**
- **Stochastic Gradient Descent**
- **Newton's Method**
- **Iterative algorithms**

#### When to Use:
- Large datasets where closed form is computationally expensive
- Complex models where closed form doesn't exist
- Online learning scenarios
- When memory constraints exist

## Advantages of Closed Form Method

1. **Exact solution** - no approximation errors
2. **Computationally efficient** for small to medium datasets
3. **No hyperparameter tuning** required
4. **Guaranteed convergence** to optimal solution
5. **Easy to implement** and understand

## Example Implementation

```python
import numpy as np

def simple_linear_regression_closed_form(x, y):
    """
    Calculate slope and intercept using closed form method
    """
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Calculate slope (m)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    m = numerator / denominator
    
    # Calculate y-intercept (b)
    b = y_mean - m * x_mean
    
    return m, b

# Example usage
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])
m, b = simple_linear_regression_closed_form(x, y)
print(f"Slope (m): {m}")
print(f"Y-intercept (b): {b}")
print(f"Equation: y = {m:.2f}x + {b:.2f}")
```

This closed form approach is the foundation of linear regression and is widely used in practice due to its simplicity and reliability.
