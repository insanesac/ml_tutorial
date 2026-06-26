"""Classification metrics: accuracy, precision, recall, F1, ROC-AUC.

Confusion matrix:
                Predicted 0    Predicted 1
    Actual 0       TN              FP
    Actual 1       FN              TP

Metrics:
- Accuracy:  (TP + TN) / total — overall correctness.
- Precision: TP / (TP + FP) — of all predicted positive, how many are correct?
- Recall:    TP / (TP + FN) — of all actual positive, how many did we find?
- F1 Score:  2 * P * R / (P + R) — harmonic mean of precision and recall.
- ROC-AUC:   area under ROC curve — ranking quality of predicted scores.

When to use which?
- Balanced data → accuracy is fine.
- Imbalanced data → precision/recall/F1 are more informative.
- Need ranking quality → ROC-AUC.
"""

import numpy as np

# --- Dataset ---
y_true = np.array([0, 0, 0, 1, 1, 1])
y_pred = np.array([0, 0, 1, 1, 1, 0])  # 1 false positive, 1 false negative

# --- Confusion Matrix ---
tp = np.sum((y_true == 1) & (y_pred == 1))  # true positives
fp = np.sum((y_true == 0) & (y_pred == 1))  # false positives
tn = np.sum((y_true == 0) & (y_pred == 0))  # true negatives
fn = np.sum((y_true == 1) & (y_pred == 0))  # false negatives

print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")

# --- Metrics ---
# Accuracy: overall fraction of correct predictions.
accuracy = (tp + tn) / len(y_true)

# Precision: TP / (TP + FP) — guard against division by zero.
precision = tp / (tp + fp) if (tp + fp) > 0 else 0

# Recall: TP / (TP + FN) — guard against division by zero.
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

# F1: harmonic mean of precision and recall.
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# --- ROC-AUC ---
# Requires continuous scores (not binary predictions).
# Measures how well the model ranks positive examples above negative ones.
y_scores = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
from sklearn.metrics import roc_auc_score
print(f"ROC-AUC:   {roc_auc_score(y_true, y_scores):.4f}")
