---
title: Binary Cross‑Entropy (Log Loss) and Sigmoid Derivative
sidebar_position: 6
---

This note motivates why Logistic Regression uses a **loss function**, derives **Binary Cross‑Entropy (BCE)** from **Maximum Likelihood**, and records the **sigmoid derivative** used in optimisation.

## Why we need a loss function
- Perceptron and the naive sigmoid update move boundaries point‑by‑point but offer no principled way to pick the **best** model.
- Machine learning formalises “best” via a **loss function** that we minimise.

## From Maximum Likelihood to Cross‑Entropy
1. Predict probability with sigmoid: `ŷ_i = sigma(z_i)`, where `z_i = W · x_i + b` and `sigma(z) = 1 / (1 + exp(-z))`.
2. For each example `i` with true label `y_i ∈ {0,1}` we select the probability of the actual class:
	- if `y_i = 1` → use `ŷ_i`
	- if `y_i = 0` → use `1 - ŷ_i`
3. Model likelihood over all N i.i.d. examples is the product of these probabilities. To avoid tiny products, take logs and sum:
	- Log‑likelihood `LL = Σ [ y_i * log(ŷ_i) + (1 - y_i) * log(1 - ŷ_i) ]`
4. We minimise the **negative** log‑likelihood (i.e., maximise `LL`). This gives **Binary Cross‑Entropy (Log Loss)** per example:

`BCE_i = - [ y_i * log(ŷ_i) + (1 - y_i) * log(1 - ŷ_i) ]`

Dataset average (what optimisers typically minimise):

`BCE = (1/N) * Σ_i BCE_i`

Behaviour:
- If `y_i = 1`, the loss reduces to `-log(ŷ_i)` → encourage `ŷ_i → 1`.
- If `y_i = 0`, the loss reduces to `-log(1 - ŷ_i)` → encourage `ŷ_i → 0`.

## Optimisation: Gradient Descent (high level)
- There is no closed‑form solution for `argmin_W BCE(W)`; we use **Gradient Descent** or variants (LBFGS, SAG, SAGA, Adam…).
- Core step: `W := W - lr * ∇_W BCE`. The gradients rely on the **sigmoid derivative**.

## Sigmoid and its derivative
Definition:

`sigma(z) = 1 / (1 + exp(-z))`

Equivalent form (useful for differentiation): `sigma(z) = exp(z) / (1 + exp(z))`.

Derivative (result to remember):

`sigma'(z) = sigma(z) * [ 1 - sigma(z) ]`

One‑line proof sketch (algebraic):
1. Let `s = sigma(z)`. Then `s = 1 / (1 + e^{-z})` ⇒ `1 - s = e^{-z} / (1 + e^{-z})`.
2. Differentiate `s` w.r.t. `z` using quotient/chain rules, or note `s = (1 + e^{-z})^{-1}` so `ds/dz = (1) * (1 + e^{-z})^{-2} * e^{-z}`.
3. Substitute back to express in terms of `s`: `ds/dz = s * (1 - s)`.

## Putting it together: gradient intuition
For a single example, with `z = W · x + b` and `ŷ = sigma(z)`, the gradient of BCE w.r.t. `W` simplifies to:

`∂BCE/∂W = (ŷ - y) * x`

and w.r.t. bias:

`∂BCE/∂b = (ŷ - y)`

This neat form arises from the chain rule with `sigma'(z) = sigma(z) * (1 - sigma(z))` and makes implementation straightforward.

## Key takeaways
1. BCE is the negative log‑likelihood of the Bernoulli model with sigmoid link.
2. We minimise BCE (not multiply probabilities) for numerical stability and convenient gradients.
3. The crucial identity `sigma'(z) = sigma(z) * (1 - sigma(z))` yields simple gradients `(ŷ - y)`.
4. Use Gradient Descent/solvers to find weights that minimise BCE; add regularisation (L2/L1/Elastic Net) as needed.

