---
sidebar_position: 3
title: DBSCAN
---

## DBSCAN Clustering Algorithm Notes

### I. Introduction and Necessity

**DBSCAN** is a type of **Density-Based Clustering** algorithm. Unlike Centroid-Based Clustering (like K-Means), DBSCAN groups data points based on the **density** between them.

It was developed to address several key flaws in the K-Means algorithm:

1.  **Requirement of $K$ (Number of Clusters):** K-Means requires the user to specify the number of clusters beforehand, which is difficult or impossible for high-dimensional data. The elbow method often yields ambiguous results. DBSCAN, conversely, figures out the number of clusters on its own.
2.  **Sensitivity to Outliers:** K-Means is highly sensitive to outliers because distance calculations cause centroids to shift away from their true positions. DBSCAN has a built-in capability to identify outliers.
3.  **Non-Spherical Data:** K-Means (a centroid-based algorithm) performs poorly when given non-spherical or weirdly shaped data and clusters it incorrectly. DBSCAN can find **arbitrarily shaped clusters**.

### II. Core Concepts and Hyperparameters

DBSCAN uses density to separate dense regions from sparse regions. This density is defined by two crucial hyperparameters that must be tuned:

| Hyperparameter | Description | Role |
| :--- | :--- | :--- |
| **Epsilon ($\epsilon$ or `eps`)** | Defines the radius or distance of a local neighborhood (e.g., 1 unit). | Determines the size of the circle drawn around a point to check density. |
| **Min Points (`min_samples`)** | Defines the minimum number of data points required within the $\epsilon$-neighborhood to consider that neighborhood "dense". | If the count is $\ge$ Min Points, the area is dense; otherwise, it is sparse. |

### III. Types of Data Points

DBSCAN classifies every data point into one of three categories based on $\epsilon$ and Min Points:

1.  **Core Point:**
    *   A data point is a Core Point if its **$\epsilon$-neighborhood** contains a number of data points **greater than or equal to Min Points**.
    *   *Example:* If Min Points = 5, and the $\epsilon$-neighborhood contains 5 or more points, it is a Core Point.

2.  **Border Point:**
    *   A data point is a Border Point if its $\epsilon$-neighborhood contains **fewer than Min Points** data points.
    *   **BUT**, its $\epsilon$-neighborhood must contain **at least one Core Point**.

3.  **Noise Point (Outlier):**
    *   A data point is a Noise Point if it is **neither a Core Point nor a Border Point**.
    *   Noise points are typically left unclustered.

### IV. Density Connectedness (Clustering Logic)

If two points (A and B) are **Density Connected**, they can be placed into the same cluster.

*   **Definition:** Points A and B are Density Connected if they are **indirectly connected via a chain of Core Points**.
*   **Crucial Condition:** The distance between **all adjacent pairs of Core Points** in the chain must be **less than or equal to $\epsilon$**.
*   If the path includes a Border Point or a Noise Point, or if the distance between any adjacent Core Points exceeds $\epsilon$, the path breaks, and the points are not Density Connected.

### V. DBSCAN Algorithm Execution (4 Steps)

The algorithm executes in four steps (following setup):

1.  **Step 0 (Setup):** Decide the values for Min Points and $\epsilon$.
2.  **Step 1 (Point Identification):** Calculate and label all data points as Core, Border, or Noise Points.
3.  **Step 2 (Cluster Core Points):**
    *   Take an unclustered **Core Point** and use it to create a new cluster.
    *   Add all other **unclustered points** that are **density connected** to the initial point into this cluster.
    *   Repeat this process for all remaining unclustered Core Points to form new clusters.
4.  **Step 3 (Assign Border Points):**
    *   For every unclustered **Border Point**, assign it to the cluster belonging to its **nearest Core Point**.
5.  **Step 4 (Handle Noise Points):** Noise Points are left unclustered.

*Note: In implementations like scikit-learn, Noise Points are assigned the label **-1**.*

### VI. Advantages and Limitations

#### Advantages

*   **Handles Arbitrary Shapes:** Can cluster data effectively regardless of shape (unlike K-Means).
*   **Robust Outlier Detection:** Has a built-in mechanism (Noise Points) to handle outliers, making it useful for anomaly detection.
*   **No $K$ Required:** Automatically determines the number of clusters.
*   **Few Hyperparameters:** Requires tuning only Min Points and $\epsilon$.

#### Limitations

*   **Hyperparameter Sensitivity:** The algorithm is highly sensitive; small changes in Min Points or $\epsilon$ can drastically alter the clustering results.
*   **Varying Densities:** Performs poorly when dealing with clusters that have significantly different densities (e.g., one cluster is sparse and the other is tightly packed).
*   **No Prediction Capability:** The algorithm can only run on training data and cannot make predictions for new, unseen data points without rerunning the entire clustering calculation (training process).