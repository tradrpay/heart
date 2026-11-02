# Heart Disease MLOps Pipeline on Databricks

An end-to-end MLOps project that trains a binary heart-disease classifier using **Databricks (Unity Catalog + Asset Bundles)** for compute and orchestration, and **GitHub Actions** for CI/CD.

## Overview

This project takes the UCI Heart Disease dataset and builds a reproducible training pipeline:

1. **Raw data** is stored as a table in Unity Catalog (`raw_data_catalog.raw_schema.raw_heart`).
2. **Feature engineering** (`preprocess.py`) cleans the raw table and produces a features table + a fitted scikit-learn `ColumnTransformer`.
3. **Training** (`train.py`) consumes the features and preprocessor, trains a classifier, and logs the model to MLflow.
4. **CI/CD** is driven by GitHub Actions using Databricks Asset Bundles (DAB) for deployment.

The ML problem is binary classification: predict presence of heart disease (`num > 0` → 1, else 0).

## Architecture

```
┌──────────────────┐      ┌──────────────────────┐      ┌──────────────────┐
│  GitHub Repo     │─────▶│  GitHub Actions      │─────▶│  Databricks      │
│  (source code)   │      │  (CI/CD workflows)   │      │  Workspace       │
└──────────────────┘      └──────────────────────┘      └──────────────────┘
                                 │                             │
                                 │ OIDC (Workload Identity)    │
                                 ▼                             ▼
                          ┌──────────────┐          ┌──────────────────────┐
                          │ Databricks   │          │ Unity Catalog        │
                          │ Service      │◀─────────│  - raw_heart         │
                          │ Principal    │          │  - features_heart    │
                          └──────────────┘          │  - model registry    │
                                                    └──────────────────────┘
```

**Key components:**

| Component | Purpose |
|-----------|---------|
| Unity Catalog | Stores raw and feature tables; enforces access control |
| Databricks Asset Bundle | Declarative definition of jobs, clusters, and environments |
| MLflow | Model tracking, registry, and lineage |
| GitHub Actions | CI on PRs, CD on merge to `main` |
| Workload Identity Federation | Passwordless auth from GitHub to Databricks |
