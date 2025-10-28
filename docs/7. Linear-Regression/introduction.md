---
sidebar_position: 1
---

# Linear Regression

Linear regression is a fundamental supervised learning algorithm used for predicting continuous numerical values. It establishes a linear relationship between input features and the target variable.

## Types of Linear Regression

### 1. Simple Linear Regression (Single Linear Regression)
- **Single input**: Uses only one independent variable (feature) to predict the dependent variable
- **Formula**: `y = mx + b` where:
  - `y` = dependent variable (target)
  - `x` = independent variable (feature)
  - `m` = slope
  - `b` = y-intercept
- **Example**: Predicting house price based only on square footage

### 2. Multiple Linear Regression
- **Multiple inputs**: Uses two or more independent variables to predict the dependent variable
- **Formula**: `y = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ`
- **Example**: Predicting house price based on square footage, number of bedrooms, age, and location

### 3. Polynomial Linear Regression
- **Non-linear data**: Used when the relationship between variables is not linear but can be modeled with polynomial terms
- **Formula**: `y = b₀ + b₁x + b₂x² + b₃x³ + ... + bₙxⁿ`
- **Example**: Modeling growth patterns, temperature changes over time, or any curved relationships

### 4. Regularized Linear Regression
- **Prevents overfitting**: Adds penalty terms to the cost function to control model complexity
- **Two main types**:
  - **Ridge Regression (L2)**: Adds sum of squared coefficients penalty
  - **Lasso Regression (L1)**: Adds sum of absolute coefficients penalty
- **Benefits**: Reduces overfitting, handles multicollinearity, performs feature selection (especially Lasso)

## Understanding Stochastic Errors

Real-world data contains **stochastic errors** - random noise that cannot be mathematically predicted. These include:
- **Measurement errors** and environmental factors
- **Unpredictable influences** that affect the outcome
- **Random variations** around the true linear trend

This is why data appears "sort of linear" rather than perfectly linear. Linear regression works well because it finds the best linear approximation to data that has underlying linear relationships but includes this random noise.