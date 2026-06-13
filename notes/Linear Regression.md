# Linear Regression

## What problem does it solve?

Predict a continuous value.

Examples:
- House price
- Temperature
- Sales

Not probabilities.

Not classes.

## Core Intuition

Fit the best line through data.

Prediction:

$$
\hat{y} = wx + b
$$

Goal: Make predictions as close as possible to actual values.

## Loss Function

Mean Squared Error.

Why square?
- Large mistakes punished heavily.
- Positive and negative errors don't cancel.

## Gradient Intuition

Gradient answers:

> If I increase w slightly, what happens to loss?

**Positive gradient:**
- Increasing w increases loss.
- Therefore: Decrease w.

**Negative gradient:**
- Increasing w decreases loss.
- Therefore: Increase w.

## Why Does x Appear In dL/dw?

Because w only affects prediction through wx.

Large x: small change in w → large change in prediction.

Therefore larger influence on gradient.

## Gradient Descent

1. Predict
2. Compute Error
3. Compute Gradient
4. Update Parameters
5. Repeat

## Complexity

**Training:** O(END)
- E = epochs
- N = samples
- D = features

**Inference:** O(D)

## Common Traps
- Forget learning rate.
- Wrong error sign.
- Mixing MSE and RMSE.
- Using vector of biases instead of scalar bias.

## 30 Second Explanation

Linear regression predicts continuous values by fitting a weighted linear relationship between features and target. Parameters are learned by minimizing mean squared error using gradient descent.
