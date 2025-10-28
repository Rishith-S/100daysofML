---
sidebar_position: 3
---

# Ridge Regularization - Part 2

Ridge Regression is a regularisation technique applied to linear models. It works by adding a penalty term to the original loss function, which helps reduce overfitting and decrease the values of the model's coefficients. This penalty term is the sum of squared coefficients, multiplied by a hyperparameter called **Lambda (λ)**, also referred to as **Alpha** in some contexts.

## 1. How Coefficients are Affected by Ridge Regression

### Shrinkage Towards Zero
- When Ridge Regression is applied, all **coefficients shrink or decrease towards zero**
- However, they **never become exactly zero**, even with very large Lambda values (e.g., λ = 1000)
- Coefficients become extremely close to zero but never reach it precisely

### Impact of Lambda (λ)
- **λ = 0**: Ridge Regression becomes simple Linear Regression (regularisation term vanishes)
- **λ increases**: The **shrinkage of coefficients becomes more pronounced**, pushing them closer to zero
- **Example**: Increasing λ from 0 → 10 → 100 → 1000 significantly reduces the range of coefficient values
  - Initial range: -150 to 150
  - Final range (λ = 1000): -0.4 to 0.8

## 2. Differential Impact on Coefficient Magnitudes

### Large vs. Small Coefficients
- **Larger coefficients shrink much faster and more significantly** than smaller coefficients as λ increases
- **Example**: A coefficient starting at 561 might drop to 0.207, while a smaller coefficient decreases less dramatically
- The regularisation **penalises larger coefficients more heavily**, bringing them closer to the values of smaller coefficients
- **Key Rule**: No coefficient becomes exactly zero, regardless of λ magnitude (even λ = 10,000)

## 3. Effect on Bias and Variance

### The Bias-Variance Trade-off
Ridge Regression addresses the fundamental bias-variance trade-off in models:

#### Small Lambda (λ) - Approaching Zero
- Model tends to **overfit** the training data
- **Low bias** (good performance on training data)
- **High variance** (poor generalisation to unseen data)
- Similar to a highly complex model trying to touch every data point

#### Large Lambda (λ) - Approaching Infinity
- Model tends to **underfit** the training data
- **High bias** (poor performance on both training and test data)
- **Significantly decreased variance**

### Optimal Lambda Selection
- As λ increases:
  - **Bias curve** generally increases
  - **Variance curve** generally decreases
  - **Generalisation error** = Bias + Variance (approximately)
- **Goal**: Find the optimal λ value at the intersection point (or slightly before) where:
  - Variance has significantly reduced
  - Bias is still relatively low
  - This region provides the best balance between underfitting and overfitting
  - Results in better generalisation performance on test data

## 4. Impact on the Loss Function

### Loss Function Composition
The loss function in Ridge Regression (plain text to avoid math plugin requirements):
`Loss = MSE + λ Σ β_i^2`

### Effect of Increasing Lambda
- **The minimum of the loss function graph shifts towards the origin** (where coefficients = 0)
- The loss function graph tends to **"shrink" and "rise up"**
- This movement of the minimum provides **intuition for why coefficients shrink towards zero**
- Behaviour is consistent whether dealing with one or multiple coefficients

## 5. Why it's Called "Ridge Regression"

### Geometric Interpretation
The name comes from the geometric interpretation of how the solution for coefficients is found under regularisation.

### Constraint Visualization
- **Mean Squared Error (MSE) component**: Forms a **contour plot** (often elliptical or spherical in higher dimensions)
  - Minimum at the "unconstrained" solution (standard linear regression coefficients)
- **Penalty term component**: Defines a **circle (or hypersphere)** centred at the origin
  - Size of circle is related to the value of λ
  - Represents the constraint: `∑ β_i^2 ≤ k` (for some constant k)

### Solution Location
- Ridge Regression solution is found at the point where the **contour plot first touches the perimeter of the circle**
- The solution is **always found on the boundary (perimeter)** of this circle/sphere
- It's the point on the circle's perimeter **closest to the unconstrained linear regression solution**
- This boundary solution is why the technique is called **"Ridge Regression"**

:::note
Illustration placeholder: To add the geometric diagram, place an image at `static/img/ridge-regression-geometric.png` and reference it here as:

`![Ridge Regression Geometric Interpretation](/img/ridge-regression-geometric.png)`

The previous relative path was removed to avoid build errors when the image is missing.
:::

### Constraint Classification
- Ridge Regression is considered a **"soft constraint"** regression
- Unlike Lasso (L1 regularisation), which creates a diamond-shaped constraint, Ridge uses circular/spherical constraints

## Practical Considerations

### When to Use Ridge Regression
- **Most useful with 2+ input columns**: Multiple coefficients need regularisation
- **Very effective** when there are many input columns
- Ridge Regression becomes increasingly powerful as the number of features grows

### When NOT to Use Ridge Regression
- **Limited utility with 1 input column**: Single coefficient requires less regularisation
- Consider other methods when feature count is very small

## Key Takeaways

1. Ridge Regression shrinks coefficients towards (but not to) zero
2. Larger coefficients shrink faster than smaller ones
3. Lambda (λ) controls the bias-variance trade-off
4. Optimal λ is found where variance reduction meets acceptable bias levels
5. Loss function minimum shifts towards the origin as λ increases
6. "Ridge" terminology derives from the geometric constraint interpretation
7. Most effective with multiple input features
