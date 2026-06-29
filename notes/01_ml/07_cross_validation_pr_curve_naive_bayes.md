# Final Classical ML Topics: Cross Validation, PR Curve & Naive Bayes

## 1. Cross Validation

### Motivation

A single train-test split may produce misleading evaluation results:

```
80% Train | 20% Test → Accuracy = 91%
Was this split representative?
```

Cross Validation evaluates the model over **multiple train-validation splits** for a more reliable estimate of generalization.

### K-Fold Cross Validation

```
K = 5
Split: Fold1 | Fold2 | Fold3 | Fold4 | Fold5

Round 1: Train [2,3,4,5]  Validate [1]
Round 2: Train [1,3,4,5]  Validate [2]
...
Round 5: Train [1,2,3,4]  Validate [5]
```

Every fold serves as validation exactly once.

### Final Performance

```
Validation accuracies: 90%, 92%, 91%, 89%, 93%
Final score: Mean = 91%
```

Much more stable than a single split.

### Advantages

- Better estimate of generalization
- Efficient use of limited data
- Useful for hyperparameter tuning
- Reduces dependence on one random split

### Disadvantages

- Computationally expensive (K training rounds)
- Rarely used for very large deep learning models

### Why Not For LLM Training?

Training a modern LLM already requires enormous compute. Repeating training K times is generally infeasible.

---

## 2. Precision-Recall Curve

### Motivation

Accuracy becomes misleading for imbalanced datasets:

```
1000 Patients: 990 Healthy, 10 Cancer
Model predicts "Everyone Healthy" → Accuracy = 99%
Yet the model completely fails to detect cancer.
```

### Precision

**Of all predicted positives, how many were actually positive?**

```
Precision = TP / (TP + FP)
```

High Precision → Few False Positives

**Think:** Can I trust the model when it predicts Positive?

### Recall

**Of all actual positives, how many did the model detect?**

```
Recall = TP / (TP + FN)
```

High Recall → Few False Negatives

**Think:** Did the model find all the positive examples?

### Memory Trick

| Metric | Denominator | Thinks About |
|---|---|---|
| **Precision** | Predicted Positives (TP + FP) | "Of what I predicted as positive..." |
| **Recall** | Real Positives (TP + FN) | "Of what is actually positive..." |

### Threshold Trade-off

| Threshold | Precision | Recall | Effect |
|---|---|---|---|
| High | ↑ | ↓ | Only highly confident predictions accepted |
| Low | ↓ | ↑ | More positives detected, but more false positives |

### PR Curve

Each classification threshold produces a (Precision, Recall) pair. Plotting these produces the PR Curve. **Larger area under the curve = better classifier.**

### PR vs ROC

| | ROC Curve | PR Curve |
|---|---|---|
| Best for | Balanced datasets | Imbalanced datasets |

---

## 3. Naive Bayes

### Motivation

```
Email contains: "Free", "Money", "Lottery"
Spam or Not Spam?
```

Naive Bayes computes P(Spam | Email) using Bayes' Theorem.

### Bayes Theorem

```
P(A|B) = P(B|A) × P(A) / P(B)

For classification:
P(Class | Features) → predict class with highest posterior
```

### Why "Naive"?

Assumes all input features are **conditionally independent**:

```
P(Spam | Free, Money, Lottery) ∝ P(Spam) × P(Free|Spam) × P(Money|Spam) × P(Lottery|Spam)
```

This assumes "Free" is independent of "Money" — rarely true, but the classifier often performs surprisingly well.

### Training

Estimate:
- **Class Prior:** P(Class)
- **Feature Likelihood:** P(Feature | Class)

### Prediction

```
P(Class) × P(Feature_1|Class) × P(Feature_2|Class) × ...
→ Choose class with highest posterior probability
```

### Types

| Type | Features | Use Case |
|---|---|---|
| Gaussian | Continuous numerical | Real-valued features |
| Multinomial | Word counts | Text classification (most common) |
| Bernoulli | Binary (present/not) | Binary feature presence |

### Advantages

- Extremely fast
- Requires little training data
- Strong baseline for text classification
- Easy to implement

### Disadvantages

- Strong independence assumption
- Usually less accurate than modern discriminative models

### Why Does It Still Work?

Although calculated probabilities may be inaccurate, the **correct class often still receives the highest score**. Classification depends on ranking, not perfectly calibrated probabilities.

---

## Precision vs Recall Cheat Sheet

| Metric | Formula | Measures |
|---|---|---|
| Precision | TP / (TP + FP) | Of predicted positives, how many were correct? |
| Recall | TP / (TP + FN) | Of actual positives, how many were found? |

## Classical ML Summary

| Topic | Core Idea |
|---|---|
| Decision Tree | Recursive feature splitting using impurity reduction |
| Random Forest | Ensemble of independent trees using bagging |
| XGBoost | Sequential boosting using residual correction |
| PCA | Dimensionality reduction through maximum variance |
| Cross Validation | Reliable evaluation using multiple train-validation splits |
| PR Curve | Evaluation metric for imbalanced classification |
| Naive Bayes | Probabilistic classifier using Bayes' Theorem |
