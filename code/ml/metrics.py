import numpy as np

y_true = np.array([0, 0, 0, 1, 1, 1])
y_pred = np.array([0, 0, 1, 1, 1, 0])

# Confusion Matrix
tp = np.sum((y_true == 1) & (y_pred == 1))
fp = np.sum((y_true == 0) & (y_pred == 1))
tn = np.sum((y_true == 0) & (y_pred == 0))
fn = np.sum((y_true == 1) & (y_pred == 0))

print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")

accuracy = (tp + tn) / len(y_true)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# ROC-AUC (simplified with scores)
y_scores = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
from sklearn.metrics import roc_auc_score
print(f"ROC-AUC:   {roc_auc_score(y_true, y_scores):.4f}")
