# Classical ML: Decision Trees, Random Forest, XGBoost & PCA

## 1. Decision Trees

### Core Idea

Recursively partition the feature space by selecting splits that maximize information gain or minimize impurity.

```
Example:
Is Weather = Rain?
├── Yes → Don't Play
└── No
    │
    ▼
Is Humidity High?
├── Yes → Don't Play
└── No → Play
```

- Each **internal node** = a question (feature split)
- Each **branch** = an answer
- Each **leaf** = final prediction

### Impurity

| Node Type | Composition | Impurity |
|---|---|---|
| Pure | All same class | 0 |
| Mixed | Multiple classes | High |

Objective: recursively create pure leaf nodes.

### Gini Impurity

Most commonly used split criterion:

```
G = 1 - Σ p_i²

Example: Yes=0.5, No=0.5
G = 1 - (0.25 + 0.25) = 0.5
```

Lower Gini = better split.

### Entropy

Alternative impurity measure:

```
Entropy = -Σ p_i log(p_i)

Information Gain = Entropy_before - Entropy_after
```

The tree selects the split with the highest information gain.

### Continuous Features

Decision Trees automatically search for thresholds:

```
Age < 25?  Age < 30?  Age < 35?
```

The threshold producing the purest children is selected.

### Preventing Overfitting

| Control | Description |
|---|---|
| `max_depth` | Limit tree depth |
| `min_samples_split` | Minimum samples to split a node |
| `min_samples_leaf` | Minimum samples in a leaf |
| `max_leaf_nodes` | Maximum number of leaves |

### Complexity

- Training: O(n·d·log n)
- Prediction: O(depth)

### Advantages

- Easy to interpret (visual tree)
- Handles nonlinear relationships
- Works with numerical and categorical features
- Minimal preprocessing (no scaling needed)

### Disadvantages

- High variance — small data changes produce different trees
- Overfits easily without pruning

### Interview Q&A

**Why don't trees require feature scaling?**
Thresholds compare feature values. Scaling changes `Age < 25` to `Age < 2500` — ordering remains identical, so scaling has no effect.

**Gini vs Entropy?**
Gini is computationally faster and the common default. Entropy has information-theoretic interpretation. Often produces similar trees.

---

## 2. Random Forest

### Motivation

A single Decision Tree has high variance. Random Forest reduces variance by **averaging many trees**.

### Core Idea

Train many independent trees and combine predictions:

- **Classification:** Majority Vote
- **Regression:** Average Prediction

### Two Sources of Randomness

**1. Bootstrap Sampling**

Each tree is trained on a bootstrap sample (sampling with replacement):

```
Original:  1 2 3 4 5
Bootstrap: 2 5 5 1 3   (some repeat, some omitted)
```

**2. Random Feature Selection**

At each split, only a random subset of features is considered. Different trees learn different decision boundaries.

### Out-of-Bag (OOB) Error

~1/3 of training samples are not selected for each tree. Those samples become an internal validation set — provides performance estimate without a separate validation split.

### Advantages

- Lower variance
- Less overfitting
- High accuracy
- Robust to noisy features

### Disadvantages

- Larger model (many trees)
- Slower inference
- Harder to interpret

### Random Forest vs Decision Tree

| | Decision Tree | Random Forest |
|---|---|---|
| Trees | One | Many |
| Variance | High | Lower |
| Interpretability | Easy | Less interpretable |
| Overfitting | Easy | Better generalization |

---

## 3. XGBoost (Extreme Gradient Boosting)

### Motivation

Random Forest builds trees independently. XGBoost builds trees **sequentially** — each new tree corrects the mistakes of the previous trees.

### Workflow

```
Tree 1 → Compute Errors → Tree 2 learns residuals → Update Predictions
→ Tree 3 corrects remaining errors → Continue until convergence
```

### Final Prediction

```
Prediction = Tree_1 + Tree_2 + Tree_3 + …
```

### Residual Learning

```
Residual = Actual - Prediction
```

The next tree predicts the residual, not the original target. Classification uses gradients of the loss function instead of simple residuals — hence **Gradient Boosting**.

### Learning Rate

Each tree contributes only a fraction:

```
Prediction = Old + η × NewTree
```

Small learning rates improve generalization.

### Regularization

XGBoost explicitly penalizes model complexity:

```
Objective = Loss + Regularization
```

### Why "Extreme"?

Engineering improvements: regularization, efficient tree construction, missing value handling, parallel split finding, memory optimization.

### Advantages

- Excellent performance on tabular data
- Built-in regularization
- Handles nonlinear relationships
- Provides feature importance

### Disadvantages

- Sequential training (slower)
- More hyperparameters
- Can overfit if not tuned

### Random Forest vs XGBoost

| | Random Forest | XGBoost |
|---|---|---|
| Trees | Independent | Sequential |
| Training | Parallel | Sequential |
| Tuning | Less | More |
| Noise robustness | Better | Can fit noise if overtrained |
| Accuracy | Strong baseline | Often highest |

---

## 4. Principal Component Analysis (PCA)

### Motivation

Many features are highly correlated (e.g., Height & Weight). PCA compresses correlated features into a smaller set of informative components.

### Core Idea

Find new **orthogonal directions** that maximize variance — called Principal Components.

- **PC1:** Captures maximum variance
- **PC2:** Captures maximum remaining variance, orthogonal to PC1

### Algorithm

```
Step 1: Center the data (subtract mean from every feature)
Step 2: Compute covariance matrix  Cov = (1/n) X^T X
Step 3: Compute eigenvectors and eigenvalues
        Eigenvectors → principal directions
        Eigenvalues  → variance explained by each direction
Step 4: Sort components by decreasing eigenvalues
Step 5: Keep top k principal components
Step 6: Project data onto these components
```

### Variance Explained

```
Component    Variance
PC1          80%
PC2          15%
PC3           5%
```

Keeping PC1 + PC2 preserves 95% of total variance.

### Advantages

- Dimensionality reduction
- Faster training, lower memory
- Removes multicollinearity
- Useful for visualization

### Disadvantages

- Components lose interpretability
- Linear technique only
- May discard predictive low-variance features

### Interview Q&A

**Why center the data?**
Centering removes the mean so PCA captures variance rather than absolute offsets.

**Why doesn't PCA use labels?**
PCA is unsupervised — operates only on the feature matrix X. Target labels are completely ignored.

**Why can PCA hurt accuracy?**
PCA removes low-variance components, but low-variance features may still be highly predictive. Always validate on downstream model performance.

---

## Key Interview Takeaways

| Algorithm | Key Concept | Strength | Weakness |
|---|---|---|---|
| Decision Tree | Recursive splitting, Gini/Entropy | Interpretable, nonlinear | High variance, overfits |
| Random Forest | Bootstrap + random features + voting | Low variance, robust | Less interpretable, slower |
| XGBoost | Sequential residual/gradient boosting | Highest accuracy, regularized | Sequential, more tuning |
| PCA | Eigenvectors of covariance matrix | Dimensionality reduction | Unsupervised, loses interpretability |
