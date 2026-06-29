# Production ML: Data Pipelines

## Overview

A Data Pipeline is the end-to-end system responsible for collecting, validating, transforming, storing, and serving data for machine learning.

Unlike traditional software, ML systems continuously depend on data quality. **Poor data almost always leads to poor models.**

## End-to-End Pipeline

```
Raw Data
  ↓
Data Ingestion
  ↓
Validation
  ↓
Cleaning
  ↓
Feature Engineering
  ↓
Feature Store
  ↓
Training Dataset
  ↓
Model Training
  ↓
Evaluation
  ↓
Model Registry
  ↓
Deployment
  ↓
Monitoring
  ↓
Data Drift Detection
  ↓
Retraining
```

---

## 1. Data Collection

### Sources

Databases, Logs, Sensors, User Events, APIs, Documents, Images, Videos

### Goals

- Reliable
- Complete
- Representative

---

## 2. Data Validation

### Checks

| Check | Example |
|---|---|
| Missing values | Null fields |
| Invalid schema | Wrong column types |
| Duplicate records | Same row twice |
| Corrupted files | Truncated data |
| Incorrect data types | String instead of int |
| Range violations | Age = -5 → Reject |

---

## 3. Data Cleaning

- Remove duplicates
- Fill missing values
- Normalize formats
- Handle outliers
- Remove corrupted samples

---

## 4. Feature Engineering

Convert raw data into model features:

| Data Type | Transformations |
|---|---|
| Images | Resize, normalize, augment |
| Text | Tokenization, embeddings |
| Tabular | Scaling, encoding, aggregation |

---

## 5. Feature Store

Central repository for reusable features.

### Benefits

- Training and inference consistency (same features, same computation)
- Feature reuse across teams
- Versioning
- Reduced duplication

---

## 6. Model Training

Train using: Feature Store + Labels + Hyperparameters → Model weights

---

## 7. Model Evaluation

### Offline Metrics

| Domain | Metrics |
|---|---|
| Classification | Accuracy, Precision, Recall, F1, AUC |
| Regression | RMSE, MAE |
| LLMs | BLEU, ROUGE, Perplexity, Groundedness, Hallucination Rate |

---

## 8. Model Registry

Stores: model versions, metadata, metrics, training configuration, dataset version

**Purpose:** Enable reproducibility and rollback.

---

## 9. Deployment

| Strategy | Description |
|---|---|
| Batch Inference | Pre-compute predictions offline |
| Online Inference | Real-time serving |
| Shadow Deployment | New model runs in parallel, no user impact |
| Canary Deployment | Gradual rollout to subset of traffic |
| Blue-Green Deployment | Instant switch between two environments |

---

## 10. Monitoring

| Category | Metrics |
|---|---|
| Model Metrics | Latency, throughput, error rate |
| Prediction Metrics | Accuracy, confidence, drift |
| Business Metrics | CTR, conversion, user satisfaction |

---

## 11. Data Drift

Production data changes over time.

```
Training: Mostly English → Production: Mostly Hindi → Performance drops
```

### Types

| Type | What Changes |
|---|---|
| Data Drift | Input distribution changes (P(X)) |
| Concept Drift | Relationship between input and output changes (P(Y\|X)) |
| Label Drift | Output distribution changes (P(Y)) |

---

## 12. Retraining

### Triggers

- Scheduled retraining
- Data drift detected
- Performance degradation
- New data availability

---

## Interview Summary

A production ML pipeline manages the complete lifecycle of data and models: ingestion, validation, feature engineering, training, evaluation, deployment, monitoring, drift detection, and retraining. The objective is to ensure reliable, reproducible, and continuously improving ML systems.
