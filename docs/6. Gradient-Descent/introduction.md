---
sidebar_position: 1
---

# Gradient Descent

- **Gradient Descent (GD)** is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function.
- **Gradient:** The slope or direction of steepest ascent. In GD, we move in the opposite direction of the gradient.
- **Learning Rate:** A hyperparameter that controls the step size at each iteration.
- **Convergence:** The state where the algorithm has reached or is very close to the minimum.

**When to stop:**
- When the change in parameter values between iterations is below a certain threshold.
- After a fixed number of epochs (iterations).

### Gradient Descent: Detailed Notes

Gradient Descent is a fundamental **first-order iterative optimisation algorithm** used for **finding the local minimum of a differentiable function**. It is a core concept in machine learning and deep learning, aimed at achieving the **best possible performance** or "tuning" a model.

### 1. What is Gradient Descent?

- **Definition**: Gradient Descent is an optimisation algorithm that finds the local minimum of a differentiable function by repeatedly taking steps in the **opposite direction of the gradient** of the function at the current point.
- **Purpose**: Its main purpose is to find the **minimum value of a given function**. In machine learning, this "function" is typically a **loss function** (or cost function), and the minimum of this function corresponds to the optimal parameters for a model.
- **Opposite Direction**: Moving in the direction of the gradient would lead to a local maximum (this procedure is called Gradient Ascent), while moving in the opposite direction leads to a local minimum.
- **Iterative Process**: The algorithm works by taking repeated "steps".

### 2. Why is Gradient Descent Important?

- **Limitations of Direct Solutions**: For problems like Multiple Linear Regression, a direct "normal equation" solution can be used to find optimal parameters (like 'm' and 'b'). However, this involves calculating the inverse of a matrix, which becomes computationally **very costly and time-consuming** in multi-dimensional or high-dimensional problems.
- **Necessity in Complex Models**: Gradient Descent provides an **alternative approach** for calculating coefficients ('m' and 'b') when direct solutions are infeasible.
- **Versatility**: It is a **general algorithm** not limited to Linear Regression. It is widely used in:
    - Logistic Regression.
    - Algorithms like SVMs (Support Vector Machines).
    - **Deep Learning**: It forms the "backbone" of deep learning, making it an indispensable topic for anyone pursuing deep learning.

### 3. The Core Intuition: Optimising for a Single Parameter ('b' - Intercept)

The video simplifies the problem by initially focusing on optimising only the intercept ('b') while keeping the slope ('m') constant in a Linear Regression context.

- **Loss Function**: In Linear Regression, the **loss function (L)** is typically the Sum of Squared Errors (SSE) or Mean Squared Error (MSE).
    - `L = Σ (y_actual - y_predicted)^2`.
    - `y_predicted = m * x + b`.
    - Therefore, `L = Σ (y_actual - (m * x + b))^2`.
- **Dependency**: The loss function `L` depends on the values of `m` and `b`. If `m` is fixed, `L` becomes a function solely of `b`.
- **Visualising the Loss Function**: When `L` is plotted against `b` (with `m` fixed), it typically forms a **parabolic shape**.
    - The **goal** is to find the value of `b` for which `L` is at its **minimum** point on this parabola.
- **The Iterative Process**:
    1. **Start with a Random 'b'**: Begin with any random initial value for 'b'.
    2. **Determine the Direction**: From the current point on the loss curve, how do we know whether to increase or decrease 'b' to reach the minimum? The **slope** of the curve at that point provides this information.
        - If the slope is **negative**, increasing 'b' will move towards the minimum.
        - If the slope is **positive**, decreasing 'b' will move towards the minimum.
- **The Update Rule (Initial Idea)**:
    - `b_new = b_old - slope`.
    - This equation automatically handles the direction: if the slope is negative, `b_new` increases; if positive, `b_new` decreases. This aligns with taking steps in the **opposite direction of the gradient**.

### 4. The Learning Rate (`alpha` / `η`)

- **Problem with Simple Update**: Simply subtracting the slope can lead to **drastic changes** in 'b', causing the algorithm to "overshoot" the minimum and oscillate around it, or even diverge.
- **Solution**: Introduce a **learning rate (`alpha`)**, a small positive constant, to control the step size.
    - **Update Rule with Learning Rate**: `b_new = b_old - learning_rate * slope`.
    - The term `learning_rate * slope` is called the **step size**.
- **Impact of Learning Rate**:
    - **Too Small (`alpha` ≈ 0.001)**: The algorithm takes very small steps, making convergence **very slow** and requiring many more iterations (epochs) to reach the minimum. This means slower training and more computation.
    - **Optimal (`alpha` ≈ 0.1 or 0.2)**: The algorithm converges efficiently and stably to the minimum. It takes larger steps when far from the minimum (where the slope is steeper) and smaller steps when close (where the slope flattens), naturally slowing down near the optimum.
    - **Too High (`alpha` ≈ 0.5 or higher)**: The algorithm overshoots the minimum repeatedly, leading to **oscillations, zigzagging**, and potentially **divergence** (moving further away from the minimum). It might never reach the minimum or converge to incorrect values.
- **Tuning**: Learning rate is a **hyperparameter** that needs to be carefully chosen and often requires experimentation.

### 5. Stopping Criteria (When to Stop Iterations)

Since Gradient Descent is an iterative algorithm, it needs criteria to determine when to stop updating the parameters.

- **Small Change in Parameter**: Stop when the absolute difference between `b_new` and `b_old` (or `m_new` and `m_old`) becomes **very small** (e.g., less than 0.0001). This indicates that the parameter value is no longer significantly changing, implying convergence.
- **Fixed Number of Epochs**: Set a predefined maximum number of iterations (epochs). This is common, and you can determine a suitable number by observing the loss curve (see below).
- **Loss Stabilisation**: Monitor the **loss function's value over epochs**. When the loss graph becomes **flat** (i.e., the loss value is no longer significantly decreasing), it indicates that the algorithm has converged, and further iterations won't provide much improvement.

### 6. Step-by-Step Algorithm (1D Case - 'b' Optimisation)

1. **Initialise 'b'**: Set `b_old` to a random value (e.g., 0 or -120).
2. **Set Hyperparameters**: Choose a `learning_rate` (e.g., 0.01) and a number of `epochs` (e.g., 100).
3. **Iterate (Loop)**: For each epoch (from 0 to `epochs`1):
a. **Calculate Predicted `y`**: For each data point `(x_i, y_i)`, calculate `y_predicted_i = m_fixed * x_i + b_old`.
b. **Calculate the Slope (Derivative of Loss w.r.t 'b')**:
* The loss function is `L = Σ (y_actual_i - (m * x_i + b))^2`.
* The derivative `dL/db` (slope) is `dL/db = -2 * Σ (y_actual_i - m * x_i - b)`.
c. **Update 'b'**: `b_new = b_old - learning_rate * slope`.
d. **Assign for Next Iteration**: Set `b_old = b_new`.
4. **Output**: After all epochs, the final `b` value is the optimised intercept.

### 7. Visualisations (1D Case)

The video uses several visualisations to enhance understanding.

1. **Line Convergence**: Shows how the best-fit line (representing `y = m*x + b`) gradually moves from an initial random position towards the optimal line as 'b' is updated over epochs.
2. **Cost vs. 'b' (Parabolic Descent)**: Illustrates the loss function (L) plotted against 'b'. It shows the iterative "steps" taken down the parabolic curve from the starting random 'b' towards the minimum, demonstrating that steps get smaller as the slope flattens near the minimum.
3. **Cost vs. Epochs**: Plots the loss function value against the number of epochs. This graph typically shows a rapid decrease in loss initially, followed by a flattening curve, indicating convergence. This is crucial for determining when to stop training.
4. **'b' vs. Epochs**: Plots the 'b' parameter value against the number of epochs. It shows how 'b' rapidly changes initially and then stabilises around the optimal value as epochs increase.

### 8. Universality of Gradient Descent

- **Beyond Linear Regression**: Gradient Descent is a **general optimisation technique**. Its core update equation (`parameter_new = parameter_old - learning_rate * slope`) remains consistent across different machine learning algorithms.
- **Loss Function is Key**: The only part that changes between algorithms is the **loss function** itself and, consequently, its **derivative (slope/gradient)**.
    - **Differentiable Loss Function**: As long as the loss function is **differentiable**, Gradient Descent can be applied.
    - **Examples**: Logistic Regression has a different loss function (e.g., Cross-Entropy Loss), and Deep Learning models also use various loss functions (e.g., Categorical Cross-Entropy, MSE). In each case, the derivative of that specific loss function with respect to the parameters is used in the Gradient Descent update rule.

### 9. Gradient Descent for Multiple Parameters ('m' and 'b')

When optimising both 'm' (slope) and 'b' (intercept), the problem becomes multi-dimensional.

- **Initialisation**: Randomly initialise both `m` and `b`.
- **Loss Surface (3D)**: The loss function `L` (which depends on both `m` and `b`) now represents a **3D surface**, typically a **paraboloid** or "bowl" shape. The goal is to find the point (combination of `m` and `b`) at the bottom of this bowl where `L` is minimum.
- **Gradient**: When a function depends on multiple variables, its derivative is called a **gradient**. The gradient is a vector of **partial derivatives**, where each component is the derivative of the loss function with respect to one of the parameters.
    - **Partial Derivative for 'b'**: `∂L/∂b = -2 * Σ (y_actual_i - m * x_i - b)`.
    - **Partial Derivative for 'm'**: `∂L/∂m = -2 * Σ (y_actual_i - m * x_i - b) * x_i`.
- **Simultaneous Update**: In each epoch, both `m` and `b` are updated simultaneously using their respective partial derivatives (slopes) and the learning rate.
    - `b_new = b_old - learning_rate * (∂L/∂b)`.
    - `m_new = m_old - learning_rate * (∂L/∂m)`.
- **Direction in 3D**: The gradient provides the "steepest descent" direction on the 3D loss surface, guiding the algorithm towards the minimum.
- **Generalisation**: This concept extends to any number of parameters (e.g., 10-15-20 or more features in multi-variate regression or deep learning networks).

### 10. Visualisations (2D Case - 'm' and 'b' optimisation)

1. **Line Convergence (3D effect)**: Similar to the 1D case, it shows how the line gradually transforms (both slope and intercept changing) from a poor initial fit to the best-fit line over epochs.
2. **Contour Plots**: A **2D projection of the 3D loss surface**, using colours or contour lines to represent the "height" (loss value). The path of 'm' and 'b' updates over epochs is shown as a line gradually moving towards the darkest central region (the minimum). This helps visualise the descent path in a 2D plane.
3. **Cost vs. Epochs**: Similar to the 1D case, showing the decrease and stabilisation of the loss function over iterations.
4. **'b' vs. Epochs**: Shows the convergence of the intercept parameter 'b'.
5. **'m' vs. Epochs**: Shows the convergence of the slope parameter 'm'.

### 11. Factors Affecting Gradient Descent's Performance

1. **Learning Rate**: As discussed earlier, the learning rate is **critical**.
    - **Too Low**: Slow convergence, requires many epochs.
    - **Too High**: Leads to overshooting, zigzagging, or divergence, preventing convergence to the minimum.
    - **Optimal**: Achieves fast and stable convergence.
2. **Loss Function**: The shape of the loss function significantly impacts Gradient Descent.
    - **Convex Functions**: Have only **one global minimum**. Linear Regression's Mean Squared Error is a convex function, making it easy for Gradient Descent to find the optimal solution.
    - **Non-Convex Functions**: Can have multiple minima:
        - **Local Minima**: Points where the slope is zero, but they are not the absolute lowest point on the surface. Gradient Descent can get **stuck** in a local minimum if it starts too close to it, providing a sub-optimal solution.
        - **Saddle Points**: Flat regions where the slope is close to zero, but the function increases in some directions and decreases in others (like a saddle). Gradient Descent can **slow down significantly** or get stuck on these plateaus, taking a very long time to escape or not reaching the global minimum.
    - Non-convex functions are common in Deep Learning, making these issues a significant challenge.
3. **Data Scaling (Feature Scaling)**:
    - If features (columns in your dataset, e.g., 'x' values) are on **different scales** (e.g., one feature ranges from 0-1 and another from 0-1000), the contour plot of the loss function will be **elongated and elliptical**.
    - This causes Gradient Descent to take a **longer and more zigzagging path** to converge, slowing down the training process.
    - **Solution**: It is highly recommended to **scale your features** (e.g., using standardisation or normalisation) before applying Gradient Descent. This makes the loss surface more **spherical**, allowing Gradient Descent to converge much faster and more directly.

### 12. Conclusion

Gradient Descent is a powerful and universally applicable optimisation algorithm in machine learning. While sensitive to parameters like the learning rate and properties of the loss function and data scaling, when properly tuned and applied, it efficiently finds optimal model parameters, even from very inaccurate starting points.
