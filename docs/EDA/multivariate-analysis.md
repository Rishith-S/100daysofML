---
sidebar_position: 3
---

# EDA: Bivariate & Multivariate Analysis

This comprehensive guide covers key concepts and practical tools for uncovering relationships in data using bivariate and multivariate analysis, based on Day 21 of the "100 Days of ML" series.

## Table of Contents
1. [Introduction](#1-introduction)
2. [Numerical-Numerical Analysis](#2-numerical-numerical-analysis)
3. [Numerical-Categorical Analysis](#3-numerical-categorical-analysis)
4. [Categorical-Categorical Analysis](#4-categorical-categorical-analysis)
5. [Pair Plots](#5-pair-plots--comprehensive-relationship-mapping)
6. [Line Plots](#6-line-plots--time-series-and-trend-analysis)
7. [Best Practices & Tips](#7-best-practices--tips)
8. [Conclusion](#8-conclusion)

---

## 1. Introduction

### What is Bivariate vs Multivariate Analysis?

- **Bivariate Analysis**: Analyzing relationships between exactly two variables to understand how they influence each other.
- **Multivariate Analysis**: Analyzing more than two variables simultaneously to uncover complex patterns and interactions.

### Key Data Type Combinations

Understanding these combinations is crucial for selecting the right analysis technique:

- **Numerical-Numerical**: Both variables are continuous (e.g., height vs weight, price vs sales)
- **Numerical-Categorical**: One continuous, one discrete (e.g., age vs gender, income vs education level)
- **Categorical-Categorical**: Both variables are discrete (e.g., gender vs survival, product category vs region)

### Datasets Used in This Guide

- **Tips**: Restaurant tipping data (numerical relationships)
- **Titanic**: Passenger survival data (mixed data types)
- **Flights**: Time-series passenger data
- **Iris**: Flower measurements by species (classification example)

---

## 2. Numerical-Numerical Analysis

### Primary Tool: Scatter Plots

**Function**: `seaborn.scatterplot()`

Scatter plots are the go-to visualization for understanding relationships between two numerical variables.

#### Basic Example: Total Bill vs. Tip
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Load the tips dataset
tips = sns.load_dataset('tips')

# Create basic scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=tips, x='total_bill', y='tip')
plt.title('Total Bill vs Tip Amount')
plt.xlabel('Total Bill ($)')
plt.ylabel('Tip ($)')
plt.show()
```

**Key Insights to Look For**:
- **Linear relationships**: Positive/negative correlation
- **Non-linear patterns**: Curves, clusters, or complex shapes
- **Outliers**: Points that don't follow the general pattern
- **Strength of relationship**: How tightly points cluster around a trend line

#### Multivariate Enhancements

Scatter plots become incredibly powerful when you add additional dimensions:

```python
# Enhanced scatter plot with multiple variables
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=tips, 
    x='total_bill', 
    y='tip',
    hue='sex',           # Color by gender
    style='smoker',      # Shape by smoker status
    size='size',         # Size by party size
    alpha=0.7
)
plt.title('Multivariate Analysis: Bill vs Tip by Demographics')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
```

**What Each Enhancement Shows**:
- `hue`: Different colors for different categories
- `style`: Different markers for different groups
- `size`: Point size represents magnitude of another variable
- `alpha`: Transparency to handle overlapping points

**Maximum Variables**: Scatter plots can effectively display up to 5 variables simultaneously (x, y, hue, style, size).
    

---

## 3. Numerical-Categorical Analysis

When analyzing the relationship between a numerical variable and a categorical variable, we have several powerful tools at our disposal.

### a. Bar Plots – Statistical Summaries

**Function**: `seaborn.barplot()`

Bar plots show the mean value of a numerical variable for each category, with confidence intervals indicating uncertainty.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset
titanic = sns.load_dataset('titanic')

# Create bar plot: Average Age by Passenger Class
plt.figure(figsize=(10, 6))
sns.barplot(data=titanic, x='class', y='age', ci=95)
plt.title('Average Age by Passenger Class')
plt.ylabel('Average Age (years)')
plt.xlabel('Passenger Class')
plt.show()
```

**Key Insights**:
- **Class 1**: ~38 years (highest average age)
- **Class 2**: ~29 years (middle)
- **Class 3**: ~25 years (youngest passengers)
- **Confidence intervals**: Show statistical uncertainty in the means

### b. Box Plots – Distribution Analysis

**Function**: `seaborn.boxplot()`

Box plots reveal the full distribution of numerical data across categories, including quartiles, medians, and outliers.

```python
# Box plot: Age distribution by Gender and Survival
plt.figure(figsize=(12, 6))
sns.boxplot(data=titanic, x='sex', y='age', hue='survived')
plt.title('Age Distribution by Gender and Survival Status')
plt.ylabel('Age (years)')
plt.xlabel('Gender')
plt.show()
```

**What Box Plots Show**:
- **Median**: Central line in each box
- **Quartiles**: Box boundaries (25th and 75th percentiles)
- **Whiskers**: Extend to 1.5 × IQR from box boundaries
- **Outliers**: Individual points beyond whiskers

### c. Distribution Plots – Density Comparison

**Function**: `seaborn.displot()`

Distribution plots compare the shape and spread of numerical data across different categories.

```python
# Distribution plot: Age of Survivors vs Non-Survivors
plt.figure(figsize=(12, 6))
sns.displot(
    data=titanic, 
    x='age', 
    hue='survived', 
    kind='hist', 
    bins=30, 
    alpha=0.7
)
plt.title('Age Distribution: Survivors vs Non-Survivors')
plt.show()
```

**Key Findings**:
- **Children under 5**: Higher survival rates (clearly visible peak)
- **Young adults (20-30)**: Large population, mixed survival
- **Elderly passengers**: Lower survival rates
- **Distribution shapes**: Can reveal bimodal or skewed patterns
    

---

## 4. Categorical-Categorical Analysis

Understanding relationships between two categorical variables requires different approaches than numerical analysis.

### a. Cross-Tabulation – Frequency Tables

**Function**: `pandas.crosstab()`

Cross-tabulation creates frequency tables showing how often category combinations occur.

```python
import pandas as pd
import seaborn as sns

# Load Titanic dataset
titanic = sns.load_dataset('titanic')

# Create cross-tabulation: Class vs Survival
crosstab = pd.crosstab(titanic['class'], titanic['survived'], margins=True)
print("Passenger Class vs Survival Status")
print(crosstab)

# Example output:
# survived    0    1   All
# class                  
# First      80  136   216
# Second     97   87   184
# Third     372  119   491
# All       549  342   891
```

### b. Heatmaps – Visual Cross-Tabulation

**Function**: `seaborn.heatmap()`

Heatmaps provide an intuitive visual representation of cross-tabulation data.

```python
# Create heatmap of the crosstab
plt.figure(figsize=(10, 6))
sns.heatmap(
    crosstab.iloc[:-1, :-1],  # Exclude 'All' row/column
    annot=True, 
    fmt='d', 
    cmap='Blues',
    cbar_kws={'label': 'Count'}
)
plt.title('Passenger Class vs Survival Status (Counts)')
plt.ylabel('Passenger Class')
plt.xlabel('Survived')
plt.show()
```

**Interpretation**:
- **Lighter shades**: Higher frequencies
- **Darker shades**: Lower frequencies
- **Annotations**: Exact counts in each cell

### c. Percentage Analysis – Survival Rates

**Function**: `groupby().mean() * 100`

Converting counts to percentages reveals patterns more clearly.

```python
# Calculate survival rates by class
survival_rates = titanic.groupby('class')['survived'].mean() * 100
print("Survival Rates by Class:")
print(survival_rates)

# Visualize survival rates
plt.figure(figsize=(10, 6))
survival_rates.plot(kind='bar', color=['skyblue', 'lightcoral', 'lightgreen'])
plt.title('Survival Rates by Passenger Class')
plt.ylabel('Survival Rate (%)')
plt.xlabel('Passenger Class')
plt.xticks(rotation=0)
plt.show()
```

**Key Findings**:
- **First Class**: ~63% survival rate
- **Second Class**: ~47% survival rate  
- **Third Class**: ~24% survival rate

### d. Cluster Maps – Hierarchical Grouping

**Function**: `seaborn.clustermap()`

Cluster maps group similar categories together based on their behavior patterns.

```python
# Create a more complex crosstab for clustering
complex_crosstab = pd.crosstab([titanic['class'], titanic['sex']], 
                              titanic['survived'])

# Create clustermap
plt.figure(figsize=(8, 10))
sns.clustermap(
    complex_crosstab, 
    annot=True, 
    fmt='d',
    cmap='viridis',
    figsize=(8, 10)
)
plt.title('Hierarchical Clustering: Class & Gender vs Survival')
plt.show()
```

**What Clustering Reveals**:
- **Similar patterns**: Categories with similar survival rates group together
- **Hierarchical structure**: Shows how categories relate to each other
- **Hidden relationships**: May reveal unexpected groupings

---

## 5. Pair Plots – Comprehensive Relationship Mapping

**Function**: `seaborn.pairplot()`

Pair plots create a matrix of scatter plots showing all pairwise relationships between numerical variables in your dataset.

### Basic Pair Plot

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Load Iris dataset
iris = sns.load_dataset('iris')

# Create basic pair plot
plt.figure(figsize=(12, 10))
sns.pairplot(iris)
plt.suptitle('Iris Dataset: All Pairwise Relationships', y=1.02)
plt.show()
```

### Enhanced Pair Plot with Categories

```python
# Pair plot with species differentiation
plt.figure(figsize=(12, 10))
sns.pairplot(
    iris, 
    hue='species',           # Color by species
    diag_kind='hist',        # Histograms on diagonal
    plot_kws={'alpha': 0.7}  # Transparency for overlapping points
)
plt.suptitle('Iris Dataset: Relationships by Species', y=1.02)
plt.show()
```

### What Pair Plots Reveal

**Diagonal Elements**: 
- Distribution of each individual variable
- Can use 'hist' (histogram) or 'kde' (density) plots

**Off-Diagonal Elements**: 
- Scatter plots between each pair of variables
- **Petal length vs. width**: Strong positive correlation
- **Species separation**: Clear clustering by species
- **Linear relationships**: Easy to spot across all variable pairs

### Advanced Customization

```python
# Customized pair plot with regression lines
g = sns.pairplot(
    iris, 
    hue='species',
    diag_kind='kde',          # Kernel density on diagonal
    kind='reg',               # Add regression lines
    palette='husl'            # Custom color palette
)
g.fig.suptitle('Iris Dataset: Relationships with Trend Lines', y=1.02)
plt.show()
```

### When to Use Pair Plots

- **Exploratory phase**: Get quick overview of all relationships
- **Feature selection**: Identify highly correlated variables
- **Pattern detection**: Spot non-linear relationships or clusters
- **Small to medium datasets**: Works best with 2-8 numerical variables

---

## 6. Line Plots – Time-Series and Trend Analysis

**Function**: `seaborn.lineplot()`

Line plots excel at showing trends over time and are essential for time-series analysis.

### Basic Time-Series Analysis

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Load flights dataset
flights = sns.load_dataset('flights')

# Create basic line plot: Passengers over time
plt.figure(figsize=(12, 6))
sns.lineplot(data=flights, x='year', y='passengers')
plt.title('Air Passenger Traffic Over Time (1949-1960)')
plt.xlabel('Year')
plt.ylabel('Number of Passengers (thousands)')
plt.show()
```

**Key Trend**: Clear upward trend from ~100k to ~600k passengers from 1949–1960.

### Multivariate Time-Series

```python
# Line plot with monthly breakdown
plt.figure(figsize=(14, 8))
sns.lineplot(data=flights, x='year', y='passengers', hue='month', 
             palette='tab10', marker='o')
plt.title('Air Passenger Traffic by Month and Year')
plt.xlabel('Year')
plt.ylabel('Number of Passengers (thousands)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
```

**Seasonal Patterns**:
- **Summer months (July-August)**: Consistently highest traffic
- **Winter months**: Lower passenger counts
- **Growth**: All months show upward trends over time

### Advanced Analysis: Pivot Tables + Heatmaps

For complex time-series patterns, combine pivot tables with heatmaps:

```python
# Create pivot table for heatmap
flights_pivot = flights.pivot(index='month', columns='year', values='passengers')

# Create heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(
    flights_pivot, 
    annot=True, 
    fmt='d', 
    cmap='YlOrRd',
    cbar_kws={'label': 'Passengers (thousands)'}
)
plt.title('Air Passenger Traffic: Month vs Year Heatmap')
plt.xlabel('Year')
plt.ylabel('Month')
plt.show()
```

### Cluster Analysis of Time Patterns

```python
# Clustermap to identify similar patterns
plt.figure(figsize=(12, 10))
sns.clustermap(
    flights_pivot, 
    cmap='viridis', 
    annot=True, 
    fmt='d',
    figsize=(12, 10)
)
plt.title('Hierarchical Clustering: Seasonal Traffic Patterns')
plt.show()
```

**What Clustering Reveals**:
- **Summer cluster**: July-August group together (peak season)
- **Winter cluster**: December-February show similar patterns
- **Transition months**: Spring/Fall months cluster separately

### When to Use Line Plots

- **Time-series data**: Any data with temporal ordering
- **Trend identification**: Spotting long-term patterns
- **Seasonal analysis**: Understanding cyclical patterns
- **Comparative trends**: Multiple categories over time
- **Forecasting preparation**: Understanding historical patterns

---

## 7. Best Practices & Tips

### Choosing the Right Visualization

| Data Types | Primary Tools | When to Use |
|------------|---------------|-------------|
| Numerical-Numerical | Scatter plots, pair plots | Finding correlations, relationships |
| Numerical-Categorical | Box plots, bar plots, violin plots | Comparing distributions across groups |
| Categorical-Categorical | Crosstabs, heatmaps, mosaic plots | Understanding frequency patterns |
| Time-Series | Line plots, seasonal plots | Trend and seasonal analysis |

### Common Pitfalls to Avoid

1. **Overplotting**: Too many points obscuring patterns
   - Solution: Use `alpha` transparency, sampling, or hexbin plots

2. **Scale Issues**: Variables with very different ranges
   - Solution: Normalize or standardize data before plotting

3. **Missing Data**: Gaps can skew interpretations
   - Solution: Handle missing data explicitly, document assumptions

4. **Correlation vs Causation**: Strong correlations don't imply causation
   - Solution: Always consider confounding variables and domain knowledge

### Performance Tips

```python
# For large datasets, consider sampling
large_data_sample = large_df.sample(n=10000, random_state=42)

# Use appropriate figure sizes
plt.figure(figsize=(12, 8))  # Adjust based on content

# Save high-quality plots
plt.savefig('analysis_plot.png', dpi=300, bbox_inches='tight')
```

### Color and Accessibility

```python
# Use colorblind-friendly palettes
sns.set_palette("colorblind")

# Or specify explicitly
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
```

---

## 8. Conclusion

### Key Takeaways

Bivariate and multivariate EDA is essential for:

1. **Pattern Discovery**: Uncovering hidden relationships in your data
2. **Feature Understanding**: Knowing how variables interact and influence each other
3. **Model Preparation**: Identifying which variables to include/exclude in models
4. **Assumption Validation**: Checking if your data meets model requirements
5. **Storytelling**: Creating compelling narratives from data insights

### Next Steps in Your Analysis Journey

1. **Statistical Testing**: Follow up visual insights with formal statistical tests
2. **Feature Engineering**: Create new variables based on discovered relationships
3. **Model Selection**: Choose appropriate algorithms based on EDA findings
4. **Validation**: Always validate patterns on unseen data

### Essential Tools Summary

- **Scatter plots**: The foundation of numerical analysis
- **Box plots**: Distribution comparison across categories
- **Heatmaps**: Powerful for categorical relationships and correlation matrices
- **Pair plots**: Comprehensive overview of all variable relationships
- **Line plots**: Essential for time-based patterns

### Further Resources

- **Documentation**: [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html)
- **Advanced Techniques**: Look into facet grids, joint plots, and interactive visualizations
- **Statistical Background**: Study correlation coefficients, significance testing, and effect sizes

Remember: EDA is an iterative process. The more you explore, the deeper your understanding becomes. Each visualization should lead to new questions and deeper analysis.