# Volume 4: KNN

## What Problem Does It Solve?

KNN performs supervised classification using nearby labeled examples.

Given a new point, it predicts label from the labels of nearest training points.

## Why It Is Called "Lazy"

KNN has almost no explicit training phase.

It mostly stores the dataset.

Computation is deferred to prediction time.

That is why KNN is called a **lazy learner**.

## Prediction Flow

1. Compute distance to all training points
2. Select `K` nearest points
3. Gather their labels
4. Majority vote
5. Return predicted class

## Why Inference Is Expensive

For each query, KNN compares against all stored points.

That means prediction cost scales with training set size.

## `K = 1`

- Very flexible decision boundary
- Can memorize noise
- Low bias, high variance

## Large `K`

- Smoother decision boundary
- Uses broader/global neighborhood
- High bias, low variance

## Bias-Variance Tradeoff via KNN

KNN is one of the clearest ways to explain bias/variance:

- Small `K`: reacts strongly to local fluctuations -> variance risk
- Large `K`: oversmooths and misses local structure -> bias risk

## Complexity

Per prediction:

- Distance computation: `O(ND)`
- Full sort approach: `O(ND + NlogN)`
- Using partial selection (`argpartition` style): near `O(ND)`

Where:
- `N` = training samples
- `D` = features

## Practical Notes

- Feature scaling is important (distance-based method)
- Class imbalance can skew voting
- Weighted voting can improve robustness

## Common Traps

- Voting on neighbor indices instead of neighbor labels
- Using fixed threshold instead of nearest-neighbor voting
- Forgetting to use exactly `K` nearest points
- Hardcoding label assumptions

## 30-Second Explanation

KNN predicts a sample’s class by finding the `K` nearest labeled training points and using majority vote. It is easy to understand and often strong as a baseline, but inference is expensive because each prediction compares against the full dataset.
