---
sidebar_position: 3
---

# Mini-Batch Gradient Descent

**Mini-Batch Gradient Descent (MBGD)** is a compromise between Batch Gradient Descent (BGD) and Stochastic Gradient Descent (SGD). It processes data in small groups (batches) and is the most widely used approach in modern deep learning.

---

## Recapping BGD and SGD

To understand MBGD, it's helpful to briefly review the other two types:

### Batch Gradient Descent (BGD)

- **Update Frequency**: Parameters are updated only **once per epoch**. This update occurs after processing the **entire dataset** (all 'm' rows).
- **Computational Cost**: It's **slow** because it requires seeing all rows to perform a single update.
- **Suitability**: Preferable for **small, convex datasets**, though it's **rarely used in practice**.

### Stochastic Gradient Descent (SGD)

- **Update Frequency**: Parameters are updated for **each individual row**. If there are 'N' rows, SGD performs **N updates per epoch**.
- **Computational Cost & Speed**: It **converges very fast** and requires **fewer epochs** to reach a solution compared to BGD. This makes it **preferable for large datasets**.
- **Memory Requirements**: Processes one row at a time, significantly reducing memory footprint.
- **Local Minima Escape**: Due to its random nature, SGD can **jump out of local minima** in non-convex functions (common in deep learning) and reach the global minimum.
- **Disadvantages**: The solution obtained can be **random** and **not always perfectly optimal**. Repeating the training might yield **different answers** due to the stochasticity.

---

## What is Mini-Batch Gradient Descent (MBGD)?

MBGD is introduced as a solution that aims to combine the benefits of both BGD and SGD.

### Core Concept

Instead of processing the entire dataset (BGD) or one row at a time (SGD), MBGD processes data in **small "batches" or "groups of rows"**.

### Update Frequency

Parameters are updated **after processing each batch**.

- If you have 'm' rows and a chosen `batch_size`, the number of updates per epoch will be `m / batch_size`.
- **Example**: If a dataset has 3000 rows and the `batch_size` is 100, there will be 30 updates per epoch (3000 / 100 = 30).

### Relationship to BGD and SGD

MBGD is a **generalized form** that can mimic both extremes:

- If `batch_size` is set to the **total number of rows (N)**, MBGD behaves like **Batch Gradient Descent** (1 update per epoch).
- If `batch_size` is set to **1**, MBGD behaves like **Stochastic Gradient Descent** (N updates per epoch).

---

## Advantages of Mini-Batch Gradient Descent

MBGD offers several compelling benefits:

### 1. **Balance between Speed and Stability**

It offers a good balance. It's faster than BGD (more frequent updates) and less random/jumpy than SGD, providing a smoother convergence path.

### 2. **Reduced Randomness**

It **reduces the randomness** inherent in SGD, which can be beneficial when you want a more stable solution while still benefiting from more frequent updates.

### 3. **Suitable for Deep Learning**

**Mini-Batch Gradient Descent is very widely used in deep learning** due to its efficiency and ability to handle large datasets effectively.

### 4. **Memory Efficiency**

By processing data in batches, it avoids loading the entire dataset into memory, making it suitable for large datasets that might not fit in RAM.

---

## Addressing Fluctuation: Learning Schedules

Even with MBGD, there can be some fluctuation in the solution, especially when it's close to the optimal point.

- **Learning Schedule**: To counter this, a **learning schedule** is employed.
- **Mechanism**: A learning schedule involves **gradually decreasing the learning rate** as training progresses and the algorithm approaches the minimum.
- **Benefit**: This helps to **reduce the random "jumps"** and stabilise the algorithm's behaviour when it is fine-tuning around the optimal solution, leading to a more precise and stable convergence. Learning schedules are **also very important in deep learning**.

---

## Visualisation of Gradient Descent Behaviours

The three gradient descent variants display distinct behaviours on a cost function contour plot:

### Batch Gradient Descent (BGD)

Shows a **very strict, straight, and consistent path** directly towards the solution. It is slow but very deterministic.

### Stochastic Gradient Descent (SGD)

Displays a **random, erratic, and "jumpy" path**. It may take steps that temporarily move away from the minimum, but overall, it converges to the correct region.

### Mini-Batch Gradient Descent (MBGD)

Its behaviour is **in between BGD and SGD**. It's neither as straight as BGD nor as random as SGD. It strikes a balance, moving towards the minimum with some controlled "wobble".

---

## Implementing Mini-Batch Gradient Descent from Scratch

When building an MBGD class from scratch, the following modifications are made compared to BGD and SGD:

### 1. Constructor (`__init__` Method)

The class's constructor is modified to accept `batch_size` as a parameter.

```python
def __init__(self, learning_rate=0.01, epochs=100, batch_size=32):
    self.learning_rate = learning_rate
    self.epochs = epochs
    self.batch_size = batch_size
```

### 2. Fit Method Modifications

The `fit` method undergoes significant changes:

- **Outer Loop**: The outer loop iterates through **epochs**.
- **Inner Loop**: An inner loop is added to iterate through **batches** within each epoch. The number of batches is calculated as `int(X_train.shape[0] / self.batch_size)`.
- **Random Batch Selection**: Inside this inner loop, **random indices** corresponding to the `batch_size` are generated using `random.sample` from the total number of rows.
- **Subset Creation**: `X_train_subset` and `y_train_subset` are created using these random indices.
- **Calculations on Subset**: All subsequent operations – including calculating `y_hat`, the derivatives of the intercept and coefficients, and updating the parameters – are performed **only on this selected subset** of data for that particular batch.

```python
def fit(self, X_train, y_train):
    for epoch in range(self.epochs):
        # Calculate number of batches
        num_batches = int(X_train.shape[0] / self.batch_size)
        
        for batch in range(num_batches):
            # Random batch selection
            batch_indices = random.sample(range(X_train.shape[0]), self.batch_size)
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            # Forward pass
            y_hat = X_batch @ self.m + self.b
            
            # Calculate derivatives on subset
            dm = (-2 / self.batch_size) * (X_batch.T @ (y_batch - y_hat)).sum()
            db = (-2 / self.batch_size) * (y_batch - y_hat).sum()
            
            # Update parameters
            self.m = self.m - self.learning_rate * dm
            self.b = self.b - self.learning_rate * db
```

### Key Insight

By tuning `batch_size`, learning rate, and epochs, you can significantly improve performance metrics like `r2_score`.

---

## Mini-Batch Gradient Descent with Scikit-learn

While Scikit-learn's `SGDRegressor` class is a general-purpose implementation of Stochastic Gradient Descent, it presents a challenge for explicit mini-batch control:

### The Challenge: No Direct `batch_size` Parameter

The `SGDRegressor` class **does not have a direct hyperparameter for batch_size**.

### The Workaround: Using `partial_fit`

To simulate mini-batch behaviour with `SGDRegressor`, one needs to use the `partial_fit` method:

- `partial_fit` performs a partial fit over a mini-batch of samples.
- Internally, `partial_fit` uses a `batch_size` of 1 when processing individual samples, meaning it acts like SGD.

### Manual Mini-Batch Implementation ("Jugad")

To achieve mini-batch training with `SGDRegressor`, you manually:

1. Loop for your desired number of `epochs`.
2. Inside the epoch loop, manually select **random subsets (batches)** of `X_train` and `y_train`.
3. Call `model.partial_fit(X_batch, y_batch)` with each of these manually created batches.

```python
from sklearn.linear_model import SGDRegressor

model = SGDRegressor(max_iter=1, random_state=42, warm_start=True)

epochs = 100
batch_size = 32

for epoch in range(epochs):
    # Shuffle data
    indices = np.random.permutation(len(X_train))
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]
    
    # Process in mini-batches
    for i in range(0, len(X_train), batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        model.partial_fit(X_batch, y_batch)
```

### Better Alternatives

Other deep learning frameworks like **Keras** or **PyTorch** often provide a direct `batch_size` parameter, simplifying mini-batch implementation:

```python
# PyTorch example
from torch.utils.data import DataLoader

train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(epochs):
    for X_batch, y_batch in train_loader:
        # Train on batch
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
```

---

## Summary: Gradient Descent Variants

| Aspect | BGD | MBGD | SGD |
|--------|-----|------|-----|
| **Updates per Epoch** | 1 | m / batch_size | m |
| **Speed** | Slow | Fast | Very Fast |
| **Memory Usage** | High | Moderate | Low |
| **Stability** | Very Stable | Balanced | Unstable |
| **Local Minima Escape** | Poor | Good | Excellent |
| **Common Usage** | Rarely | Very Common | Common (with tuning) |

---

## Key Takeaway

Mini-Batch Gradient Descent strikes the **optimal balance** between computational efficiency, memory usage, and convergence stability. It is the **de facto standard** in modern deep learning frameworks and is essential for training large neural networks on real-world datasets. Understanding MBGD's implementation and how to tune its hyperparameters is crucial for mastering machine learning and deep learning.
