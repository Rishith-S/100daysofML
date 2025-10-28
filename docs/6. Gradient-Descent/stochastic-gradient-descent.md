---
sidebar_position: 2
---


# Stochastic Gradient Descent

- **Stochastic Gradient Descent (SGD):** An optimization algorithm that updates model parameters one data point (row) at a time, instead of using the entire dataset.
- **Why use SGD?**
    - **Fast & Efficient for Big Data:** Much faster than Batch Gradient Descent on large datasets because it makes frequent updates.
    - **Low Memory Usage:** Only needs one data row in memory at a time.
    - **Escapes Local Minima:** The "noisy" or random updates help it jump out of local minima in complex, non-convex loss functions (common in deep learning).
- **Problems with SGD:**
    - **Fluctuating Solution:** The updates are erratic, so the solution can be unstable and fluctuate around the minimum.
    - **Learning Schedule:** To fix this, the learning rate is often decreased over time (a "learning schedule") to help it stabilize.
- **Batch vs. Stochastic vs. Mini-Batch:**
    - **Batch GD (BGD):** Uses the *entire dataset* for one update. Slow and memory-intensive, but smooth convergence.
    - **Stochastic GD (SGD):** Uses *one random row* for one update. Fast and memory-efficient, but noisy convergence.
    - **Mini-Batch GD (MBGD):** A compromise. Uses a small *batch* of rows (e.g., 32, 64) for one update. The most common approach in practice.

---

### Detailed Notes

Gradient Descent is a fundamental **optimisation algorithm** used in machine learning to find optimal solutions. The video primarily discusses **Stochastic Gradient Descent (SGD)**, contrasting it with **Batch Gradient Descent (BGD)** and briefly mentioning **Mini-Batch Gradient Descent (MBGD)**.

### Problems with Batch Gradient Descent (BGD)

The discussion begins by highlighting the limitations of BGD, which necessitate alternative gradient descent methods like SGD.

- **High Computational Cost and Slowness**:
    - BGD calculates the slope (derivative) for parameter updates using the **entire dataset**. For each parameter, it requires calculating the derivative across **all 'm' rows** of the data.
    - This leads to an **enormous number of derivative calculations**, especially for large datasets. For example, a dataset with 100,000 rows, 100 columns, and 1,000 epochs would require 10^10 derivative calculations, making the algorithm **very slow**.
    - This makes BGD generally **unsuitable for "big data"**.
- **High Memory Requirements**:
    - BGD relies on **vectorisation** (e.g., NumPy's `np.dot` product) to calculate predictions (`y_hat`) for all rows simultaneously.
    - To perform these operations, the **entire `X_train` dataset must be loaded into RAM at once**.
    - For **very large datasets**, this can exceed available memory, leading to **hardware limitations** and errors, making the operation impossible to perform.
- **Inability to Escape Local Minima (Implicit)**: While not directly stated as a problem for BGD, SGD's ability to escape local minima is a major advantage. BGD's smooth, consistent convergence means it can get **stuck in local minima** in non-convex loss functions, failing to find the global optimum.

---

### Stochastic Gradient Descent (SGD)

stochastic meaning : **randomly determined; having a random probability distribution or pattern that may be analyzed statistically but may not be predicted precisely.**

SGD addresses the problems of BGD and is widely used, particularly in deep learning.

- **Core Mechanism: Update per Row**:
    - Unlike BGD, SGD **updates parameters based on a single row (data point) at a time**.
    - For each update, a **single row is randomly selected** from the dataset.
    - If a dataset has 'N' rows, SGD will perform **N parameter updates per epoch** (one update for each row in an epoch).
- **Key Differences in Implementation**:
    - The `fit` method in SGD loops through epochs, and **within each epoch, it loops through each data row (or a randomly selected row)** to perform an update.
    - Derivative calculation in SGD is for a **single row** (effectively `m=1`), removing the need for the summation term present in BGD's derivative formulae.

### Why Use Stochastic Gradient Descent? (Advantages)

SGD is preferred over BGD for several compelling reasons:

1. **Computational Efficiency and Speed for Big Data**:
    - By updating parameters one row at a time, SGD performs **more frequent updates**.
    - While BGD might be faster per epoch (as it processes everything in one go), **SGD generally converges faster to a good solution in terms of overall training time** because it requires **fewer total epochs**.
    - This efficiency makes it highly suitable for **large datasets**, which are common in deep learning applications involving images (CNNs) and text (RNNs).
2. **Lower Memory Requirements**:
    - SGD only needs to load **one row at a time** into memory for parameter updates.
    - This significantly **reduces hardware requirements** and allows it to handle datasets that are too large to fit entirely in RAM, overcoming the memory limitations faced by BGD.
3. **Faster Convergence (Fewer Epochs)**:
    - Due to its frequent updates (N updates per epoch for N rows), SGD reaches a solution **faster and requires fewer epochs** compared to BGD, which only updates once per epoch. The video shows SGD converging to a reasonable result in 15 epochs, whereas BGD might require 100 epochs for a similar performance on the small example dataset.
4. **Ability to Escape Local Minima in Non-Convex Functions**:
    - This is a crucial advantage for complex models like neural networks, which often have **non-convex loss functions**.
    - BGD's smooth convergence can lead it to get **stuck in local minima**.
    - SGD's **stochastic (random) nature** means its updates are "jumpy" or "noisy". This randomness can help the algorithm **"jump out" of local minima** and continue searching for the global minimum.
    - The **visualisation** shows SGD taking a more erratic path, sometimes performing worse than the previous step, but ultimately reaching the correct region. This erratic behaviour (momentary poor performance) is what enables it to escape traps.

### Characteristics and Considerations (Disadvantages)

- **Fluctuating Solution**: The random nature of SGD means that the solution obtained is **not "steady" or perfectly precise**. Running SGD multiple times on the same data will yield slightly different results because of these random updates.
- **Fluctuation near Optimum**: Even when the algorithm is close to the optimal solution, SGD's random updates can cause the solution to **fluctuate** around the minimum, rather than converging smoothly to a single point.

---

### Addressing Fluctuation: Learning Schedule

To mitigate the fluctuations observed near the optimal solution with SGD, the concept of a **learning schedule** is used.

- **Concept**: A learning schedule involves **varying the learning rate** over epochs, rather than keeping it constant.
- **Purpose**: The primary goal is to **gradually decrease the learning rate as training progresses**. This helps the model to **stabilise its solution** and fine-tune its approach to the optimum, reducing the erratic "jumps" when it's already close to the target.
- **Implementation Example**: A function for calculating the learning rate can be defined based on the current epoch and total number of rows. As the epoch number (`i`) increases, the learning rate decreases (e.g., `learning_rate = initial_lr / (1 + i / some_constant)`).
- **Relevance**: While demonstrated for linear regression, learning schedules are **more commonly and critically used in deep learning**.
- **Scikit-learn's `SGDRegressor`**: This class offers different learning rate strategies: `constant`, `optimal`, and `adaptive`. `constant` means the learning rate remains fixed, while the others allow for variations.

---

### When to Use Stochastic Gradient Descent (Summary)

SGD is highly recommended in two key scenarios:

1. **Big Data**: When dealing with datasets that have **numerous rows and/or columns**. SGD's efficiency and lower memory footprint make it a practical choice.
2. **Non-Convex Loss Functions**: When the problem involves a **non-convex cost function** (common in deep learning, e.g., neural networks). SGD's "jumpy" updates help it avoid getting trapped in local minima and enable it to find the global minimum.

---

### Visualisations of Gradient Descent

The video uses animations to illustrate the behaviour of BGD and SGD:

- **Batch Gradient Descent (BGD) Visualisation**:
    - Shows a **smooth, consistent, and gradual improvement** towards the optimal solution.
    - The path is direct and always moves towards a better position with each step.
    - On a contour plot of the cost function, BGD follows a **straight, consistent path** from the starting point to the minimum, like steadily walking downhill.
- **Stochastic Gradient Descent (SGD) Visualisation**:
    - Demonstrates a **less consistent, "jumpy" or "random" path**.
    - At certain points, the algorithm might **temporarily perform worse** than the previous step (e.g., step N+1 might be worse than step N).
    - Despite this, **over time, it converges to the correct solution**.
    - On a contour plot, SGD's path appears **erratic and winding**, seemingly jumping around. However, these random movements help it navigate complex landscapes and eventually reach the minimum.
    - The results of SGD can **vary with each run** due to the random selection of data points for updates.

---

### Scikit-learn's `SGDRegressor` Class

The `SGDRegressor` class in Scikit-learn provides an implementation of Stochastic Gradient Descent.

- **Flexibility**: It is a general-purpose class that can apply gradient descent to **various machine learning algorithms** by allowing different **loss functions** (e.g., 'squared_loss' for linear regression, 'huber', 'epsilon_insensitive' for SVMs).
- **Parameters**:
    - `max_iter`: Specifies the **number of epochs** to run.
    - `tol`: A **tolerance value** that determines when to stop training early if parameters are no longer improving significantly.
    - `shuffle`: Whether to **shuffle the training data** after each epoch to ensure randomness.
    - `random_state`: Used to fix the random seed, ensuring **consistent results** across multiple runs for a given set of parameters. Without it, SGD's stochastic nature will lead to different results each time.
    - `learning_rate`: Defines the **learning rate strategy** (`constant`, `optimal`, `adaptive`).
    - `eta0`: The **initial learning rate** value if using a constant or other varying strategies.
    - `penalty`: Allows for applying **regularisation** (L1, L2, Elastic Net).
- **Usage**: While building a custom SGD class helps understand the internal workings, in practice, engineers often use pre-built implementations like `SGDRegressor` for convenience and robustness.

---

### Mini-Batch Gradient Descent (MBGD)

The video briefly introduces MBGD as a middle ground:

- **Balance**: MBGD strikes a balance between BGD (processing the entire dataset) and SGD (processing one row at a time).
- **Batch Size**: It updates parameters based on a **user-defined 'batch size'** (e.g., 30 rows).
- **Updates**: If a dataset has 300 rows and a batch size of 30, it will perform 10 updates per pass through the dataset (300 rows / 30 batch size = 10 batches).
- **General Usage**: MBGD is often implicitly implemented within the broader "SGD" framework in many deep learning libraries.
