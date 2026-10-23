import pandas as pd
import numpy as np
from databricks.sdk.runtime import spark
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Columns to be dropped entirely
DROP_COLS = ["id", "dataset"]

# Columns stored as TRUE/FALSE strings -> convert to 0/1
BOOL_COLS = ["fbs", "exang"]

# Categorical string columns -> one-hot encoded
CAT_COLS = ["sex", "cp", "restecg", "slope", "thal"]

# Numerical columns
NUM_COLS = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]

TARGET_COL = "num"

# Unity Catalog table that will hold engineered features + target
FEATURES_TABLE = "heart_disease_catalog.ml_data.heart_features"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Drop identifier and site columns
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # 2. Normalise blank strings -> NaN
    df = df.replace(r"^\s*$", np.nan, regex=True)

    # 3. Convert TRUE/FALSE -> 1/0 (only if still string-like)
    for col in BOOL_COLS:
        if col in df.columns and df[col].dtype == object:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.upper()
                .map({"TRUE": 1, "FALSE": 0, "1": 1, "0": 0})
            )

    # 4. Switzerland rows encode missing cholesterol as 0 - treat as missing
    if "chol" in df.columns:
        df.loc[df["chol"] == 0, "chol"] = np.nan

    # 5. Strip whitespace from remaining string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # 6. Coerce numeric columns to numbers
    for col in NUM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 7. Target must be numeric; drop rows without a label
    if TARGET_COL in df.columns:
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
        df = df.dropna(subset=[TARGET_COL])

    return df


def prepare_features(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    # Binarize: 0 = no disease, 1-4 = disease present
    y = (y > 0).astype(int)

    # Only keep columns that actually exist in the dataframe
    num_features = [c for c in NUM_COLS if c in X.columns]
    cat_features = [c for c in CAT_COLS if c in X.columns]

    # Numeric pipeline: median impute + scale
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Categorical pipeline: mode impute + one-hot
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, num_features),
        ("cat", categorical_pipeline, cat_features),
    ])

    return X, y, preprocessor


def write_features_to_catalog(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    table_name: str = FEATURES_TABLE,
) -> None:
    """
    Fit the preprocessor on X, transform it, and write the engineered
    feature matrix + target to a Unity Catalog table.

    The preprocessor is fitted here so the persisted table matches exactly
    what the model sees during training (the same fitted object is later
    reused inside the training pipeline — see train.py).
    """
    X_transformed = preprocessor.fit_transform(X)

    # Retrieve expanded feature names (OneHotEncoder creates several columns)
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"f{i}" for i in range(X_transformed.shape[1])]

        feature_names = [name.replace(" ", "_") for name in feature_names]

    features_df = pd.DataFrame(
        X_transformed.toarray() if hasattr(X_transformed, "toarray") else X_transformed,
        columns=feature_names,
    )
    features_df["target"] = y.values

    sdf = spark.createDataFrame(features_df)
    sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name)
    print(f"[preprocess] Features written to {table_name} "
          f"({features_df.shape[0]} rows, {features_df.shape[1]} cols)")