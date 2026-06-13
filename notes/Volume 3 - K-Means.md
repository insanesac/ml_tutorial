# Volume 3: K-Means

## Why Clustering Exists

Sometimes we have data with no labels.

We still want structure:
- natural groups
- segments
- prototypes

K-Means is a simple way to discover such groups automatically.

## Core Intuition

Two principles repeat until convergence:

1. Each point belongs to nearest centroid.
2. Each centroid should be the average of its assigned points.

## Assignment vs Update (Critical Distinction)

You struggled with this before, so separate them clearly.

### Assignment Step

Each point chooses nearest centroid by distance (usually Euclidean).

Keyword: **distance**.

### Update Step

Each centroid moves to mean of points assigned to that cluster.

Keyword: **mean position**, not mean distance.

## Why Mean?

K-Means minimizes squared distances.

For squared-error objectives, the arithmetic mean is the minimizer.

Why not median?
- Median minimizes absolute distance (`L1`), not squared distance (`L2`).

Why not mode?
- Mode is not a geometric center for continuous vectors.

## K-Means Objective

K-Means minimizes:

Within-Cluster Sum of Squares (WCSS)

`WCSS = Σ ||x_i - mu_{c_i}||^2`

Important:
- It has a clear loss/objective.
- But classical K-Means is not gradient descent.

It uses alternating optimization:
- assignment step
- centroid update step

## Why Outliers Break K-Means

Your `100,100` style example is perfect.

One extreme point can pull a centroid far away because mean is sensitive to outliers.

Result:
- distorted cluster centers
- worse assignments for normal points

## Elbow Method

To choose `K`:
1. Run K-Means for multiple `K` values.
2. Plot `K` vs WCSS.
3. Look for an “elbow” where gain starts diminishing.

Engineering weakness you spotted:
- This requires many full runs, which can be expensive.

## Complexity

`O(INKD)`

- `I` = iterations
- `N` = samples
- `K` = clusters
- `D` = features

## Strengths

- Simple and fast baseline
- Easy to interpret centroids
- Works well for compact, spherical clusters

## Weaknesses

- Must choose `K`
- Sensitive to initialization
- Sensitive to outliers
- Assumes roughly spherical/equal-variance clusters

## Common Traps

- Averaging distances instead of points
- Updating centroid with all points, not assigned points
- Forgetting reassignment after centroid move
- Hardcoding cluster count without validation

## 30-Second Explanation

K-Means is an unsupervised clustering algorithm that alternates between assigning points to their nearest centroid and updating centroids to the mean of assigned points. It minimizes within-cluster squared distance (WCSS) through iterative assignment-update steps until convergence.
