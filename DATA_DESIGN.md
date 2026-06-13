# Revision Dataset Philosophy

Every dataset should satisfy:

**Rule 1:** You can predict the answer before running code.

**Rule 2:** You can calculate one iteration by hand if needed.

**Rule 3:** Dataset fits on one screen.

**Rule 4:** Expected output is obvious.

---

## 1. Linear Regression (Single)

**Goal:** Learn prediction, MSE, gradient descent.

```python
X = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 7, 9, 11])
```

**True relationship:** `y = 2x + 1`

**Expected:** `w ≈ 2`, `b ≈ 1`

---

## 2. Multi-Feature Linear Regression

**Goal:** Learn `X @ w`, `X.T @ error`.

```python
X = np.array([
    [1, 2],
    [2, 1],
    [3, 3],
    [4, 2],
    [5, 4]
])
y = np.array([9, 8, 16, 15, 23])
```

**True relationship:** `y = 2*x1 + 3*x2 + 1`

**Expected:** `w ≈ [2, 3]`, `b ≈ 1`

---

## 3. Logistic Regression (Single)

**Goal:** Learn sigmoid, cross entropy, classification.

```python
X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([0, 0, 0, 1, 1, 1])
```

**Expected:** small X → class 0, large X → class 1. Weight positive.

---

## 4. Multi-Feature Logistic Regression

```python
X = np.array([
    [1, 1],
    [1, 2],
    [2, 1],
    [3, 3],
    [4, 3],
    [4, 4]
])
y = np.array([0, 0, 0, 1, 1, 1])
```

**Expected:** Both features increase with class label. `w1 > 0`, `w2 > 0`, `b < 0`.

---

## 5. K-Means

```python
X = np.array([
    [1, 1],
    [1, 2],
    [2, 1],
    [10, 10],
    [10, 11],
    [11, 10],
    [20, 20],
    [20, 21],
    [21, 20]
])
K = 3
```

**Expected centroids:** `[~1.33, ~1.33]`, `[~10.33, ~10.33]`, `[~20.33, ~20.33]`

---

## 6. KNN

```python
X_train = np.array([
    [1, 1],
    [2, 1],
    [10, 10],
    [11, 10]
])
y_train = np.array([0, 0, 1, 1])

x_new_1 = np.array([1.5, 1.5])   # Expected: Class 0
x_new_2 = np.array([10.5, 10.5]) # Expected: Class 1
```

---

## 7. Softmax

3 classes.

```python
X = np.array([
    [1, 1],
    [2, 1],
    [10, 10],
    [11, 10],
    [20, 20],
    [21, 20]
])
y = np.array([0, 0, 1, 1, 2, 2])
```

**Expected:** small → class 0, medium → class 1, large → class 2.

---

## 8. Neural Network

Reuse Softmax dataset. One hidden layer.

---

## 9. Metrics

```python
y_true = np.array([0, 0, 0, 1, 1, 1])
y_pred = np.array([0, 0, 1, 1, 1, 0])
```

Use for: Accuracy, Precision, Recall, F1, Confusion Matrix.

---

## 10. Bias-Variance

```python
X = np.array([1, 2, 3, 4, 5])
y = np.array([1, 4, 9, 16, 25])
```

**True relationship:** `y = x^2`

Use for: underfitting, overfitting, bias, variance discussion.

---

## Summary

Every dataset is:
- tiny
- deterministic
- visually obvious
- easy to debug
- easy to explain in an interview
