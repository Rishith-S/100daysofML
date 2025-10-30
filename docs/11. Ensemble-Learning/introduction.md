---
sidebar_position: 1
title: Ensemble Learning — Introduction
---

## What is Ensemble Learning?

Ensemble learning combines multiple machine learning models to produce a stronger, more reliable predictor. It leverages the “Wisdom of the Crowd”: many diverse opinions, when aggregated properly, often outperform a single expert model.

### Wisdom of the Crowd

- Collective intelligence can surpass individual judgments.
- Everyday examples:
	- KBC Audience Poll: crowds often vote for the correct answer.
	- Product Reviews: a 4.5★ average from 15k reviews is more trustworthy than a single review.
	- IMDb Ratings: aggregated user scores guide watch decisions.
	- Democracy: majority voting reflects collective choice.
	- Weight‑guessing experiment: the mean of many guesses is near the true weight.

## Core Idea at Prediction Time

An ensemble is a collection of base models (a.k.a. learners). For a new input, each base model predicts; the ensemble then aggregates:

- Classification: majority vote (hard voting) or averaged probabilities (soft voting).
- Regression: average (mean) of the base predictions.

The key is to turn many individual opinions into a single decision that is more accurate and robust.

## The Importance of Diversity

To fully benefit, base models should be different (diverse). If everyone in the “crowd” has the same background, they make similar errors. Diversity reduces the chance that all models make the same mistake.

Ways to induce diversity:

1. Different algorithms (e.g., Linear Regression, SVM, Decision Tree).
2. Same algorithm, different data subsets (e.g., bootstrap samples in bagging).
3. Combine both (different learners trained on different subsets).

## Four Common Ensemble Techniques

1) Voting (Voting Ensemble)

- Concept: Train multiple different algorithms on the same dataset.
- Prediction: aggregate their outputs.
	- Classification: majority vote (or probability average).
	- Regression: mean of predictions.
- Diversity source: different underlying algorithms.

2) Stacking (Stacked Generalization)

- Concept: Like Voting, but add a meta‑model (blender) that learns how to weigh base predictions.
- Setup: Base models train on the same data; their predictions become features for the meta‑model.
- Benefit: The meta‑model can assign higher weight to stronger learners and lower weight to weaker ones.

3) Bagging (Bootstrap Aggregation)

- Concept: Use the same base algorithm but train each model on a different bootstrap sample (sample with replacement) of the training data.
- Prediction: Aggregate base model outputs (vote/mean).
- Classic example: Random Forest (bagging with decision trees; also sub-samples features at each split for extra diversity).

4) Boosting

- Concept: Train models sequentially; each new model focuses more on the previous model’s mistakes.
- Effect: Gradually “boosts” performance by correcting errors over iterations.
- Popular algorithms: AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost.

## Why Ensembles Work

- Better decision boundaries: voting averages out idiosyncratic boundaries from individual classifiers.
- Smoother regression: averaging predictions yields a more central, balanced estimate and reduces noise.
- Bias–variance balancing: bagging reduces variance of high‑variance learners (e.g., trees); boosting can reduce bias of high‑bias learners.

## Benefits

1. Improved accuracy and generalisation compared to single models.
2. Lower variance (bagging) and/or lower bias (boosting) — moving toward the ideal of low bias and low variance.
3. Robustness: performance is less sensitive to dataset quirks.

## Disadvantages

- Higher computational cost: training and maintaining multiple models.
- Potential loss of interpretability compared to a single, simple model.

## When to Use Ensembles

- In most practical projects as a final step after data cleaning, feature engineering, and baseline modeling.
- In competitive ML (e.g., Kaggle) where ensembles like XGBoost and stacked models are ubiquitous.
- Particularly strong on small-to-medium tabular datasets; often outperform deep learning in this regime.

---

## Quick orientation

- Voting and Stacking combine different algorithms trained on the same data.
- Bagging creates diversity via resampled datasets (often with the same algorithm).
- Boosting builds models sequentially to fix prior mistakes and is among the strongest performers.

