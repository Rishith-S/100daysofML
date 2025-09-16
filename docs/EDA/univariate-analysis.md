---
sidebar_position: 2
---

# EDA using Univariate Analysis

Univariate analysis is the simplest form of data analysis where we examine one variable at a time. It helps us understand the distribution, central tendency, and spread of individual variables in our dataset. This is often the first step in EDA as it provides insights into each feature independently.

## What is Univariate Analysis?

Univariate analysis focuses on analyzing a single variable to understand its:
- **Distribution**: How values are spread across the range
- **Central Tendency**: Mean, median, mode
- **Spread**: Variance, standard deviation, range
- **Shape**: Skewness, kurtosis (the sharpness of the peak of a frequency-distribution curve.)
- **Outliers**: Unusual values that deviate from the norm

## For Categorical Data

Categorical data represents groups or categories. Common visualization techniques include:

### 1. **Count Plot (Bar Graph)**
- **Purpose**: Shows how many observations fall into each category
- **Best Use**: When you want to compare frequencies across categories
- **Interpretation**: Taller bars indicate more frequent categories
- **Example**: Number of customers by product category, count of different car brands

A bar chart showing the frequency of each category:

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Example: Count of categories in a column
sns.countplot(data=df, x='category_column')
plt.title('Count Plot of Categories')
plt.show()
```

```python
import pandas as pd

# Example: Count plot using pandas
df['category_column'].value_counts().plot(kind='bar')
plt.title('Count Plot of Categories')
plt.show()
```

### 2. **Pie Charts**
- **Purpose**: Shows the proportion of each category relative to the whole
- **Best Use**: When you want to show relative proportions or percentages
- **Interpretation**: Larger slices represent higher proportions
- **Limitations**: Hard to compare when you have many categories or similar proportions
- **Example**: Market share of different companies, distribution of survey responses

A circular chart divided into slices representing proportions:

```python
import matplotlib.pyplot as plt

# Example: Pie chart from value counts
counts = df['category_column'].value_counts()
plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Category Proportions')
plt.axis('equal')
plt.show()
```

```python
import pandas as pd

# Example: Pie chart using pandas
df['category_column'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Category Proportions')
plt.axis('equal')
plt.show()
```

## For Numerical Data

Numerical data represents measurable quantities. Key visualization techniques include:

### 1. **Histogram**
- **Purpose**: Visualizes the frequency distribution of continuous data
- **How it works**: Data is divided into bins (intervals), and bars show frequency in each bin
- **Best Use**: Understanding data distribution, identifying patterns, detecting outliers
- **Key Insights**: 
  - Shape of distribution (normal, skewed, bimodal)
  - Central tendency and spread
  - Presence of outliers or unusual patterns
- **Example**: Distribution of ages, income levels, test scores

A bar chart showing the distribution of numerical data:

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Example: Histogram of a numerical column
sns.histplot(data=df, x='numeric_column', bins=30)
plt.title('Histogram of Numeric Column')
plt.show()
```

```python
import pandas as pd

# Example: Histogram using pandas
df['numeric_column'].hist(bins=30)
plt.title('Histogram of Numeric Column')
plt.show()
```

### 2. **Distribution Plot (Distplot)**
- **Purpose**: Shows both the frequency distribution and probability density
- **Components**: 
  - Histogram bars showing frequency
  - Smooth curve (KDE - Kernel Density Estimation) showing probability density
- **Best Use**: Understanding the shape of data distribution and probability of values
- **Key Insights**:
  - Probability of randomly selecting a particular value
  - Shape of the underlying distribution
  - Comparison with theoretical distributions (normal, uniform, etc.)
- **Example**: Distribution of heights, stock prices, exam scores

Combines histogram with a smooth curve showing probability density:

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Example: Histogram with KDE curve
sns.histplot(data=df, x='numeric_column', bins=30, kde=True)
plt.title('Distribution with KDE')
plt.show()
```

```python
import pandas as pd

# Example: Distribution plot using pandas (histogram + KDE)
df['numeric_column'].plot(kind='density')
plt.title('Distribution with KDE')
plt.show()
```

### 3. **Box Plot (Box-and-Whisker Plot)**
- **Components**:
  - **Box**: Shows interquartile range (IQR) - middle 50% of data
  - **Median Line**: Line inside the box showing the median (50th percentile)
  - **Whiskers**: Lines extending to show data range (typically 1.5 × IQR)
  - **Outliers**: Points beyond the whiskers
- **Purpose**: Summarize distribution, identify outliers, compare groups
- **Key Insights**:
  - Median and quartiles
  - Data spread and symmetry
  - Presence of outliers
  - Comparison between groups
- **Example**: Comparing test scores across different classes, analyzing salary distributions

Shows the distribution using quartiles and identifies outliers:

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Example: Box plot of a numerical column
sns.boxplot(data=df, x='numeric_column')
plt.title('Box Plot of Numeric Column')
plt.show()
```

```python
import pandas as pd

# Example: Box plot using pandas
df['numeric_column'].plot(kind='box')
plt.title('Box Plot of Numeric Column')
plt.show()
```

## Key Statistics for Univariate Analysis

### **For Categorical Data**
- **Frequency**: Count of each category
- **Relative Frequency**: Proportion of each category
- **Mode**: Most frequent category

### **For Numerical Data**
- **Central Tendency**: Mean, median, mode
- **Spread**: Range, variance, standard deviation, IQR
- **Shape**: Skewness (asymmetry), kurtosis (tail heaviness)
- **Outliers**: Values beyond 1.5 × IQR from quartiles

## Best Practices

### **1. Choose Appropriate Visualizations**
- Use count plots for categorical data with few categories
- Use pie charts sparingly and only when showing proportions
- Use histograms for understanding distribution shape
- Use box plots for identifying outliers and comparing groups

### **2. Consider Data Characteristics**
- **Sample Size**: Larger samples provide more reliable distributions
- **Data Type**: Ensure you're using the right visualization for your data type
- **Outliers**: Always check for and understand outliers
- **Missing Values**: Account for missing data in your analysis

### **3. Interpretation Guidelines**
- Look for patterns, trends, and unusual observations
- Compare with expected distributions (normal, uniform, etc.)
- Consider the business context and domain knowledge
- Document your findings and any data quality issues

---

**Remember**: Univariate analysis is your foundation - it helps you understand each variable before moving to more complex multivariate analysis. Always start here to build your data intuition!