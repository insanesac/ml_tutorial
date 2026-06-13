# Logistic Regression

## What problem does it solve?

Binary classification.

Examples:
- Spam / Not Spam
- Pass / Fail
- Fraud / Not Fraud

## Core Intuition

Linear regression predicts any number.

Classification needs 0 to 1 probabilities.

## Sigmoid

Maps (-∞, +∞) to (0, 1)

$$
z = wx + b
$$

Called:
- Logit
- Log-Odds

Large positive z: Probability → 1

Large negative z: Probability → 0

## Why Not MSE?

Classification probabilities behave poorly with MSE.

Use **Cross Entropy** instead.

## Cross Entropy Intuition

Correct and confident: small loss

Wrong and confident: huge loss

## Magic Result

Gradient simplifies to:

$$
p - y
$$

Huge interview favorite. Know it.

## Complexity

**Training:** O(END)

**Inference:** O(D)

## Common Traps
- Forget sigmoid.
- Forget clipping before log.
- Using predicted class instead of probability in loss.
- Confusing logit and probability.

## 30 Second Explanation

Logistic regression performs binary classification by converting a linear score into a probability using the sigmoid function and optimizing cross-entropy loss.
