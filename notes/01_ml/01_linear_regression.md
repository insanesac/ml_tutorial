# Volume 1: Linear Regression & Gradient Descent

## The Learning Problem

Machine learning starts with a simple question:

Given examples of inputs and outputs, how do we learn a rule that generalizes to unseen data?

### Why not memorize points?

Memorization gives zero error on training samples but usually fails on new inputs.

A model should capture pattern, not store a lookup table.

### Why fit a line?

A line is the simplest useful relationship between input and output.

Simple models are easier to train, interpret, and debug.

### What is a parameter?

A parameter is a controllable value the model learns from data.

In linear regression:
- `w` controls how strongly `x` changes prediction
- `b` shifts prediction up/down

### What is a model?

A model is a parameterized function that maps input to output.

For 1D linear regression:

`y_hat = wx + b`

## Prediction Intuition

You already discovered:

`y_hat = wx + b`

Interpretation:
- `w` = slope knob
- `b` = vertical shift

## Loss: How Wrong Are We?

We need a scalar score of “badness” to optimize.

For linear regression, we use Mean Squared Error (MSE):

`MSE = (1/N) * Σ(y_hat - y)^2`

Why square?
- Large mistakes are punished heavily.
- Positive and negative errors cannot cancel.

## What Is a Gradient?

Not “just a derivative formula.”

Gradient answers this practical question:

If I nudge a parameter, what happens to loss?

- Positive gradient: increasing parameter increases loss.
- Negative gradient: increasing parameter decreases loss.

So the gradient is a direction signal for update.

## Why Move Opposite the Gradient?

One of the most important intuitions.

- Gradient points uphill (increase in loss).
- We want downhill (decrease in loss).

So we update:

`w_new = w - lr * dL/dw`

Same for `b`.

## Why Does `x` Appear in `dL/dw`?

Major insight:

`w` affects prediction only through `wx`.

So if `x` is large, a tiny change in `w` causes a larger change in `y_hat`, which causes a larger effect on loss.

That is why `x` naturally appears in the gradient.

## Chain Rule (Intuition First)

You do not need to treat chain rule as a trick.

Think of dependencies:

`w -> prediction -> error -> loss`

Changing `w` cannot affect loss directly.
It must flow through each intermediate node.

Chain rule = multiply local sensitivities along that path.

This dependency-tracking view is crucial later for:
- Logistic Regression
- Neural Networks
- Backpropagation

## Gradient Descent Loop

1. Predict `y_hat`
2. Compute error
3. Compute gradients
4. Update parameters
5. Repeat

## Learning Rate Intuition

Learning rate (`lr`) controls step size.

- Too small: very slow learning
- Medium: stable progress
- Too large: overshoot, oscillation, exploding loss

Important clarification:
- Overshooting from large `lr` is not vanishing gradient.
- Vanishing gradient is about tiny gradients, usually in deep networks.

## Bias vs Variance (Introduced Here)

This is a general ML concept, not just KNN.

- High bias: model too simple, underfits patterns.
- High variance: model too flexible, overfits noise.

Linear regression can show both, depending on features, regularization, and data complexity.

## Complexity

- Training: `O(END)`
  - `E` = epochs
  - `N` = samples
  - `D` = features
- Inference: `O(D)` per sample

## Common Traps

- Forgetting learning rate
- Wrong error sign in gradient
- Mixing MSE and RMSE
- Using vector bias when scalar bias is intended

## 30-Second Explanation

Linear regression predicts continuous values by fitting a weighted linear relationship between input features and target values. It learns `w` and `b` by minimizing mean squared error using gradient descent, where gradients tell us how to change parameters to reduce loss.
