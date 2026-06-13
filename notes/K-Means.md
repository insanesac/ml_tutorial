# K-Means

## What problem does it solve?

Clustering.

No labels.

Discover groups automatically.

## Core Intuition

Points should belong to the nearest centroid.

Centroids should represent the average position of their cluster.

## Algorithm

1. Initialize centroids.
2. Repeat:
   - Assign points to nearest centroid
   - Move centroid to cluster mean
   - Repeat

## Assignment Step

Uses: **Distance**

Usually Euclidean.

## Update Step

Uses: **Mean**

Not mean distance. Mean position.

This was a major insight.

## Loss Function

K-Means minimizes: **Within Cluster Sum of Squares (WCSS)**

But: No gradients.

## Elbow Method

Run K-Means for multiple K values.

Plot: K vs WCSS

Choose elbow.

## Weaknesses
- Sensitive to outliers: One bad point can drag centroid.
- Must choose K: Algorithm doesn't know.
- Sensitive to initialization: Different starts → different results.
- Assumes roughly spherical clusters.

## Complexity

O(INKD)

- I = iterations
- N = samples
- K = clusters
- D = features

## Common Traps
- Averaging distances instead of points.
- Using all points when updating centroid.
- Forgetting reassignment step.
- Hardcoding cluster count.

## 30 Second Explanation

K-Means is an unsupervised clustering algorithm that repeatedly assigns points to the nearest centroid and updates centroids to the mean of assigned points until convergence.
