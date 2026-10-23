"""
Train a heart disease classifier on Databricks.

Reads the raw dataset from a Unity Catalog table, writes engineered
features to another Unity Catalog table, trains a single XGBoost model,
and registers it in Unity Catalog via MLflow.

Run as a Databricks notebook / job task (needs an active Spark session
and a Unity Catalog-enabled cluster).
"""

import warnings
from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.preprocess import (
    clean_data,
    prepare_features,
    write_features_to_catalog,
    FEATURES_TABLE,
)

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.20

RAW_TABLE = "heart_disease_catalog.ml_data.heart_raw"
REGISTERED_MODEL_NAME = "heart_disease_catalog.ml_data.heart_model"
EXPERIMENT_NAME = "/Shared/heart_disease_experiment"


# --------------------------------------------------------------------------- #
# 1. Load + clean
# --------------------------------------------------------------------------- #
def load_training_data() -> pd.DataFrame:
    print(f"[1/5] Loading data from Unity Catalog: {RAW_TABLE}")
    df_raw = spark.read.table(RAW_TABLE).toPandas()
    print(f"      raw rows: {len(df_raw)}")

    df = clean_data(df_raw)
    print(f"      clean rows: {len(df)}  (dropped {len(df_raw) - len(df)})")
    return df


# --------------------------------------------------------------------------- #
# 2. Build single model pipeline
# --------------------------------------------------------------------------- #
def build_model(preprocessor, scale_pos_weight: float) -> Pipeline:
    return Pipeline([
        ("pre", preprocessor),
        ("clf", XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


# --------------------------------------------------------------------------- #
# 3. Evaluate
# --------------------------------------------------------------------------- #
def evaluate_model(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }, y_pred, y_proba


# --------------------------------------------------------------------------- #
# 4. Plots (returned as artifact paths)
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(y_test, y_pred, out_dir: Path) -> Path:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"], ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    path = out_dir / "confusion_matrix.png"
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def plot_roc_curve(y_test, y_proba, roc_auc: float, out_dir: Path) -> Path:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"XGBoost (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Heart Disease Model")
    plt.legend(loc="lower right")
    plt.tight_layout()
    path = out_dir / "roc_curve.png"
    plt.savefig(path, dpi=120)
    plt.close()
    return path


# --------------------------------------------------------------------------- #
# 5. Main
# --------------------------------------------------------------------------- #
def main():
    # --- data ---
    df = load_training_data()
    X, y, preprocessor = prepare_features(df)
    print(f"[2/5] Features: {X.shape}  |  positives: {y.mean():.1%}")

    # --- write engineered features to Unity Catalog ---
    write_features_to_catalog(X, y, preprocessor, FEATURES_TABLE)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"      train: {len(X_train)}  test: {len(X_test)}")

    # balance ratio for XGBoost
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"      scale_pos_weight = {scale_pos_weight:.2f}")

    # --- MLflow setup ---
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("[3/5] Training model...")
    model = build_model(preprocessor, scale_pos_weight)

    with mlflow.start_run(run_name="xgboost_heart_disease") as run:
        model.fit(X_train, y_train)

        # metrics
        metrics, y_pred, y_proba = evaluate_model(model, X_test, y_test)

        # 5-fold CV on AUC for a robust number
        cv_auc = cross_val_score(
            model, X_train, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            scoring="roc_auc", n_jobs=-1,
        )
        metrics["cv_auc_mean"] = cv_auc.mean()
        metrics["cv_auc_std"] = cv_auc.std()

        mlflow.log_metrics(metrics)
        mlflow.log_params({
            "n_estimators": 400,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "scale_pos_weight": round(scale_pos_weight, 4),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
        })

        print(f"[4/5] AUC={metrics['roc_auc']:.3f}  "
              f"Acc={metrics['accuracy']:.3f}  "
              f"Recall={metrics['recall']:.3f}  "
              f"CV-AUC={metrics['cv_auc_mean']:.3f}±{metrics['cv_auc_std']:.3f}")
        print(classification_report(y_test, y_pred,
                                    target_names=["No Disease", "Disease"]))

        # --- artifacts ---
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            cm_path = plot_confusion_matrix(y_test, y_pred, tmp_dir)
            roc_path = plot_roc_curve(y_test, y_proba, metrics["roc_auc"], tmp_dir)
            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(roc_path))

        # --- register model in Unity Catalog ---
        print("[5/5] Registering model in Unity Catalog...")
        xgb_clf = model.named_steps["clf"]
        X_test_pre = model.named_steps["pre"].transform(X_test)
        model_signature = infer_signature(X_test_pre, xgb_clf.predict(X_test_pre))
        mlflow.xgboost.log_model(
            xgb_model=xgb_clf,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            signature=model_signature,
        )


if __name__ == "__main__":
    main()