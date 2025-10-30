---
sidebar_position: 1
title: SVM (Support Vector Machine)
---

### Support Vector Machines (SVMs) Overview

- SVMs are a **powerful classification and regression algorithm**.
- They aim to find an **optimal hyperplane** that best separates data points of different classes.
- A key goal of SVMs is to **maximise the margin** between the classes, meaning to create the largest possible gap between the decision boundary and the closest data points.
- The data points closest to the decision boundary are called **Support Vectors**. These are crucial because they directly influence the position and orientation of the hyperplane.
- SVMs are designed to create a model that **generalises well to unseen data**.

### Hard Margin SVM

- **Concept:** Hard Margin SVM is used when data is **linearly separable** without any overlapping points.
- **Goal:** To find a decision boundary (hyperplane) that perfectly separates the classes with the maximum possible margin, ensuring no data points fall within the margin or on the wrong side of the boundary.
- **Mathematical Formulation:**
    - The decision boundary is represented by the equation `w.x + b = 0`, where `w` is the weight vector and `b` is the bias.
    - For a given data point `x_i` with its true label `y_i` (either +1 or -1), the constraint for correct classification is `y_i * (w.x_i + b) >= 1`.
    - The goal is to **maximise the margin**, which is mathematically equivalent to **minimising `||w||`** or `(1/2) * ||w||^2`. The margin width is `2 / ||w||`.
- **Limitations:** Hard Margin SVM is not suitable for **non-linearly separable data** or data with noise/outliers, as it requires perfect separation. If the data cannot be perfectly separated by a linear boundary, a hard margin SVM will not find a solution.

### Soft Margin SVM

- **Concept:** Soft Margin SVM is an **extension of Hard Margin SVM** designed to handle data that is **not perfectly linearly separable** or contains noise.
- It allows for **some misclassifications** or points to fall within the margin, making it more robust for real-world scenarios.

### The Kernel Trick

- **Purpose:** The Kernel Trick is a crucial feature of SVMs that allows them to classify **non-linear data**.
- **Mechanism:** It works by implicitly mapping the original data from a lower-dimensional space (e.g., 2D) into a higher-dimensional feature space (e.g., 3D) where the data becomes **linearly separable**. Once transformed, a linear decision boundary (a hyperplane) can then be used to separate the classes in this new higher-dimensional space.
- **Computational Efficiency:** A key benefit of the kernel trick is that it **avoids explicit transformation** of data points into the higher-dimensional space. Instead, it uses a "kernel function" that calculates the dot product of the transformed features in the original low-dimensional space, saving significant computational cost. You don't actually create new features; you just fit them into the formula.

### Example: Radial Basis Function (RBF) Kernel

- The RBF kernel is a common non-linear kernel function.
- **Example Transformation:** For 2D data (x, y) that is circularly separated (e.g., red points in the centre, blue points around them), applying an RBF kernel transforms it into 3D space.
- **Effect of RBF:** The RBF function (e.g., `e^(-x^2)`) causes the **central data points to be lifted upwards** in the new dimension, while the outer points remain relatively lower. This transformation makes the data linearly separable by a plane in the 3D space.
- **Demonstration:** In an example, a linear SVM applied to non-linear data achieved only 55% accuracy, but when the **RBF kernel was applied, the accuracy jumped to 100%** without any explicit feature engineering. This demonstrates the power of the kernel trick.

### Other Kernels

- Besides RBF, other kernel functions can be used, such as the **Polynomial Kernel**.
- The effectiveness of polynomial kernels can vary depending on their "degree". For instance, a polynomial kernel with degree 2 might perform well, while a degree 3 polynomial might yield poorer results.

In essence, the kernel trick allows SVMs to classify complex, non-linear patterns by finding a way to make them linearly separable in a higher-dimensional space without the high computational cost of actually projecting the data.