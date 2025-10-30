---
sidebar_position: 6
title: Naive Bayes Classifier
---

## Why Naive Bayes

Naive Bayes (NB) is a fast, surprisingly strong baseline for classification, especially in high‑dimensional sparse settings like text. It is a generative model that estimates $P(y)$ and $P(x \mid y)$, then predicts the most probable class with Bayes’ rule.

## Bayes rule and the “naive” assumption

Given a feature vector $x = (x_1, \dots, x_d)$ and class $y \in \{1, \dots, K\}$:

$$
P(y \mid x) \propto P(y) \cdot P(x \mid y)
$$

NB assumes conditional independence of features given the class:

$$
P(x \mid y) = \prod_{j=1}^{d} P(x_j \mid y)
$$

Taking logs to avoid underflow and to get a linear form in sufficient statistics:

$$
\hat{y}(x) = \arg\max_y \Big( \log P(y) + \sum_{j=1}^{d} \log P(x_j \mid y) \Big)
$$

The choice of $P(x_j \mid y)$ defines NB variants.

## Common variants and likelihoods

1) Gaussian NB (continuous real‑valued features)

- Assumes $x_j \mid y \sim \mathcal{N}(\mu_{jy}, \sigma_{jy}^2)$.
- Log‑likelihood for a feature value $v$:

$$
\log P(v \mid y) = -\tfrac{1}{2}\log(2\pi\sigma_{jy}^2) - \frac{(v-\mu_{jy})^2}{2\sigma_{jy}^2}
$$

2) Multinomial NB (counts, e.g., bag‑of‑words)

- Suitable for non‑negative integer counts per feature.
- Parameters: per‑class token probabilities $\theta_{jy}$ with $\sum_j \theta_{jy} = 1$.
- For count vector $x$ the log‑likelihood (dropping constants) is $\sum_j x_j \log \theta_{jy}$.
- Smoothing (Lidstone/Dirichlet) with parameter $\alpha$:

$$
\hat{\theta}_{jy} = \frac{N_{jy} + \alpha}{N_{\cdot y} + \alpha d}
$$

where $N_{jy}$ is the total count of feature $j$ in class $y$ and $N_{\cdot y}$ is the sum over features.

3) Bernoulli NB (binary indicators)

- For features indicating presence or absence, with per‑class probabilities $\theta_{jy} = P(x_j=1 \mid y)$.
- Log‑likelihood term per feature: $x_j \log \theta_{jy} + (1-x_j)\log(1-\theta_{jy})$.

4) Complement NB (text, strong imbalance)

- Estimates $\theta$ from all documents not in class $y$ to counteract imbalance and variance; often stronger than Multinomial NB on skewed text.

5) Categorical NB (discrete categories with small alphabet)

- Uses categorical likelihood with per‑class categorical probabilities for each feature.

## Training, prediction, and complexity

- Training estimates $P(y)$ and per‑class feature likelihood parameters by simple counts or moments. One pass over the data suffices.
- Complexity is $\mathcal{O}(n d)$ for $n$ samples and $d$ features (times $K$ classes), highly parallelizable.
- Prediction computes class scores $\log P(y) + \sum_j \log P(x_j \mid y)$; complexity $\mathcal{O}(dK)$ per sample (sparse features make this near linear in the number of non‑zeros).

## Smoothing and zero counts

Without smoothing, unseen events yield zero likelihood and kill the product. Lidstone (add‑$\alpha$) smoothing with $\alpha \in [10^{-3}, 1]$ is common. Larger $\alpha$ increases bias and stabilizes rare features; tune on validation data.

## Class priors

- Empirical priors: $\hat{P}(y)=n_y/n$.
- Uniform priors when class frequencies are not representative of deployment.
- Custom priors reflect business costs; in scikit‑learn set `class_prior` or use sample weights.

## Numerical stability

- Always work in log space.
- For Gaussian NB, apply `var_smoothing` by adding a small value to variances to avoid division by very small numbers.

## Relationship to linear models

For Bernoulli/Multinomial NB in the binary case, the decision function is linear in features. The weight for feature $j$ is approximately $\log \frac{\theta_{j1}}{\theta_{j0}}$ and the bias includes $\log \frac{P(y=1)}{P(y=0)}$. This underpins the NB‑SVM trick used in text classification.

## Pros and cons

Pros

- Extremely fast to train and predict; scales to millions of features with sparse matrices.
- Needs little data; robust with strong regularization via smoothing.
- Competitive for text, bag‑of‑words, and certain sensor features.

Cons

- Independence assumption is rarely true; correlated features can degrade accuracy.
- Probabilities are often poorly calibrated; apply Platt or isotonic calibration if probabilities drive decisions.
- Gaussian NB can be weak if continuous features are far from normal or if variances differ in a class‑conditional, non‑diagonal way (violates independence).

## scikit‑learn: quick recipes

### GaussianNB for continuous features

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB

X, y = make_classification(n_samples=4000, n_features=10, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X_tr, y_tr)
y_pr = clf.predict(X_te)
print(classification_report(y_te, y_pr))
```

### MultinomialNB for text (bag‑of‑words or tf‑idf)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.pipeline import make_pipeline

texts = [
	"great product works perfectly",
	"awful quality broke immediately",
	"love this excellent purchase",
	"terrible waste of money",
]
labels = [1, 0, 1, 0]

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.5, random_state=42, stratify=labels)

pipe = make_pipeline(TfidfVectorizer(ngram_range=(1,2), min_df=1), MultinomialNB(alpha=0.5))
pipe.fit(X_train, y_train)
print(classification_report(y_test, pipe.predict(X_test)))

# Complement NB (often better under imbalance)
cnb = make_pipeline(TfidfVectorizer(), ComplementNB(alpha=0.5))
cnb.fit(X_train, y_train)
print(classification_report(y_test, cnb.predict(X_test)))
```

### Partial‑fit (online) learning

`MultinomialNB`, `BernoulliNB`, and `CategoricalNB` support `partial_fit`, enabling streaming updates and class‑by‑class ingestion.

```python
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB(alpha=1.0)
clf.partial_fit(X_batch1, y_batch1, classes=[0,1])  # first call needs classes
clf.partial_fit(X_batch2, y_batch2)
```

## Tuning guide

- Choose the variant to match feature type: Gaussian for continuous, Multinomial for counts, Bernoulli for binary indicators, Complement NB for skewed text, Categorical for small discrete categories.
- Tune smoothing `alpha` on a log scale (for Gaussian use `var_smoothing`).
- For text, try both raw counts and tf‑idf; MultinomialNB expects non‑negative values.
- Handle imbalance with class weights or dataset resampling; ComplementNB is a strong baseline.
- Calibrate probabilities if needed (`CalibratedClassifierCV`).

## Edge cases and tips

- Remove features that are always zero in a class to reduce noise (or let smoothing handle them with a small `alpha`).
- Standardization is usually unnecessary for Multinomial/Bernoulli; for Gaussian NB, standardizing can help when variances differ by orders of magnitude.
- Missing values: GaussianNB can handle NaNs poorly; impute before training.

## References

- Mitchell, “Machine Learning”, Chapter on Bayesian Learning
- McCallum and Nigam, “A Comparison of Event Models for Naive Bayes Text Classification”
- Rennie et al., “Tackling the Poor Assumptions of Naive Bayes Text Classifiers” (Complement NB)
