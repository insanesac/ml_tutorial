# Volume 2: Logistic Regression

## Why Linear Regression Fails for Classification

Linear regression outputs any real number:
- `-15`
- `42`
- `500`

These values cannot be interpreted as probabilities.

For binary classification, we need output in `[0, 1]`.

## Core Idea

1. Compute a linear score
2. Convert that score to probability

Linear score:

`z = wx + b`

Probability:

`p = sigmoid(z)`

## What Is Sigmoid?

Not just a formula — think of it as a compression machine.

It maps:
- input: `(-∞, +∞)`
- output: `(0, 1)`

So any linear score becomes a valid probability.

## What Is `z`?

`z` is the raw model score before sigmoid.

Interpretation: **log-odds** (also called logit).

This section matters because confusion about `z` creates confusion in both loss and gradients.

## Odds and Log-Odds Intuition

Start with probability.

Example: 75% rain chance.
- `p = 0.75`
- `1 - p = 0.25`

Odds are:

`odds = p / (1 - p) = 0.75 / 0.25 = 3`

Meaning: `3:1` in favor.

Why log-odds?
- Odds range: `(0, ∞)`
- Log-odds range: `(-∞, +∞)`

Linear models naturally produce values on `(-∞, +∞)`, so predicting log-odds is a perfect fit.

Then sigmoid converts log-odds back into probability.

## Why Not MSE?

For classification, MSE is usually a poor fit:
- optimization can be less stable
- probability calibration behavior is weaker

Use **cross-entropy loss** instead.

## Cross-Entropy Intuition

Think in four cases:

- Correct + certain -> tiny loss
- Correct + unsure -> moderate loss
- Wrong + unsure -> moderate/high loss
- Wrong + certain -> huge loss

This is exactly the behavior we want from a classification loss.

## The "Magic" Gradient: `p - y`

A favorite interview result:

For sigmoid + cross-entropy, gradient w.r.t. `z` simplifies to:

`p - y`

Why this is powerful:
- elegant derivation
- efficient implementation
- easy debugging signal

This simplification comes from combining:
- sigmoid derivative
- cross-entropy derivative

## Training Loop

1. Compute `z = wx + b`
2. Compute `p = sigmoid(z)`
3. Compute cross-entropy loss
4. Compute gradients (`p - y` core term)
5. Update `w`, `b`
6. Repeat

## Complexity

- Training: `O(END)`
- Inference: `O(D)` per sample

## Common Traps

- Forgetting sigmoid in forward pass
- Taking `log(0)` (fix with clipping)
- Using predicted class instead of probability inside loss
- Confusing logit (`z`) with probability (`p`)

## 30-Second Explanation

Logistic regression performs binary classification by modeling log-odds as a linear function of features and converting that score into probability through sigmoid. It learns parameters by minimizing cross-entropy loss, with the key gradient term simplifying to `p - y`.
