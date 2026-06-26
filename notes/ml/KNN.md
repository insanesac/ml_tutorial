# KNN

## What problem does it solve?

Classification using nearby labeled examples.

## Core Intuition

New point? Look at nearby labeled points. Let neighbors vote.

## Training Phase

Nothing. Store data.

This is why KNN is called: **Lazy Learner**

## Prediction Phase

1. Compute distances
2. Find K nearest
3. Get labels
4. Majority vote
5. Return class

## K = 1

Very flexible. Can memorize noise.

- Low Bias
- High Variance

## Large K

Very smooth. Uses global information.

- High Bias
- Low Variance

## Bias

Strong assumptions. Misses patterns. Underfitting.

## Variance

Sensitive to training data changes. Overfitting.

## Why Inference Is Expensive

Every prediction compares against: **All training points**

## Complexity

Distance computation: O(ND)

Full implementation: O(ND + NlogN)

Using argpartition: O(ND)

## Common Traps
- Voting on indices instead of labels.
- Using threshold instead of nearest neighbors.
- Forgetting K nearest voting.
- Hardcoding class labels.

## 30 Second Explanation

KNN is a supervised classification algorithm that predicts a new point's label by finding the K nearest labeled examples and using majority voting.
