# Google ML Interview Preparation Tracker

## Daily Ritual (Every Morning)

Reimplement from memory. No notes.

### Core Algorithms

* [x] Linear Regression (Multi Feature)
* [x] Logistic Regression (Multi Feature)

Target:

* Linear Regression: < 5 min
* Logistic Regression: < 10 min

### Oral Drill

For every algorithm:

* What problem does it solve?
* Prediction equation
* Loss function
* Why this loss?
* Gradient intuition
* Training loop
* Inference process
* Common limitations

---

# Day 1

## KNN

### Theory

* [ ] Intuition
* [ ] Distance Metrics
* [ ] Choosing K
* [ ] Curse of Dimensionality

### Implementation

* [ ] Euclidean Distance
* [ ] Neighbor Search
* [ ] Majority Voting
* [ ] Full KNN from Scratch

### Oral Drill

* [ ] Why is training fast?
* [ ] Why is inference slow?
* [ ] What happens when K=1?
* [ ] What happens when K is very large?

---

## K-Means

### Theory

* [ ] Clustering Objective
* [ ] Centroids
* [ ] Convergence
* [ ] Choosing K

### Implementation

* [ ] Distance to Centroids
* [ ] Cluster Assignment
* [ ] Centroid Update
* [ ] Full K-Means from Scratch

### Oral Drill

* [ ] Why does K-Means converge?
* [ ] Sensitivity to initialization?
* [ ] What if clusters are non-spherical?

---

## Softmax

### Theory

* [ ] Why sigmoid is insufficient for multiclass classification
* [ ] Probability interpretation
* [ ] Numerical stability

### Implementation

* [ ] Softmax Function
* [ ] Class Prediction

### Oral Drill

* [ ] Softmax vs Sigmoid
* [ ] Why exponentials?

---

## Multiclass Cross Entropy

### Theory

* [ ] Intuition
* [ ] Relation to Binary Cross Entropy

### Implementation

* [ ] Loss Computation

### Oral Drill

* [ ] Why not MSE?
* [ ] What happens if model is confidently wrong?

---

# Day 2

## Neural Networks

### Theory

* [ ] Neuron
* [ ] Layers
* [ ] Hidden Layers
* [ ] Forward Pass

### Implementation

* [ ] Single Neuron
* [ ] Dense Layer
* [ ] Multi-Layer Forward Pass

### Oral Drill

* [ ] Why are neural networks more expressive than logistic regression?
* [ ] What is a hidden layer learning?

---

## ReLU

### Theory

* [ ] Why ReLU?
* [ ] Vanishing Gradient Intuition

### Implementation

* [ ] ReLU Activation

### Oral Drill

* [ ] ReLU vs Sigmoid
* [ ] Dead ReLU Problem

---

## Backpropagation

### Theory

* [ ] Chain Rule
* [ ] Error Propagation

### Oral Drill

* [ ] Why does backprop work?
* [ ] Why chain rule?

---

# Day 3

## Metrics

* [ ] Accuracy
* [ ] Precision
* [ ] Recall
* [ ] F1 Score
* [ ] ROC-AUC
* [ ] Confusion Matrix

### Oral Drill

* [ ] Precision vs Recall
* [ ] When is Accuracy misleading?

---

## Bias / Variance

* [ ] Underfitting
* [ ] Overfitting
* [ ] Bias-Variance Tradeoff

### Oral Drill

* [ ] High Bias Symptoms
* [ ] High Variance Symptoms

---

## Regularization

* [ ] L1
* [ ] L2
* [ ] Weight Decay

### Oral Drill

* [ ] Why large weights are dangerous?
* [ ] L1 vs L2

---

## Data Splits

* [ ] Train Set
* [ ] Validation Set
* [ ] Test Set

### Oral Drill

* [ ] Why not tune on test data?

---

# Mock Interview Readiness

## Implement From Memory

* [ ] Linear Regression
* [ ] Logistic Regression
* [ ] KNN
* [ ] K-Means
* [ ] Softmax
* [ ] Multiclass Cross Entropy
* [ ] Single Neuron Forward Pass

## Explain From First Principles

* [ ] Gradient Descent
* [ ] MSE
* [ ] Cross Entropy
* [ ] Log Odds
* [ ] Overfitting
* [ ] Precision / Recall
* [ ] Backpropagation
