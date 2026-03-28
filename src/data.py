from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_classification_frame(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series, str]:
    """Load X, y, target name. Prefers KAGGLE_DATA_PATH CSV if set."""
    import os

    kaggle_path = os.environ.get("KAGGLE_DATA_PATH")
    data_cfg = cfg["data"]
    target = data_cfg["target_column"]

    if kaggle_path and Path(kaggle_path).exists():
        df = pd.read_csv(kaggle_path)
        if target not in df.columns:
            raise ValueError(f"Target column {target!r} not in {kaggle_path}")
        y = df[target]
        X = df.drop(columns=[target])
        return X, y, target

    from sklearn.datasets import fetch_openml

    name = data_cfg["openml_name"]
    version = data_cfg.get("openml_version", 1)
    bunch = fetch_openml(name=name, version=version, as_frame=True, parser="auto")
    X = bunch.data
    y = bunch.target
    if not isinstance(y, pd.Series):
        y = pd.Series(y, name=target)
    return X, y, target


def build_preprocess_and_split(
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    X, y, _ = load_classification_frame(cfg)
    rs = cfg["data"]["random_state"]
    test_size = cfg["data"]["test_size"]

    # Encode string labels for XGBoost
    if y.dtype == object or str(y.dtype).startswith("category"):
        y_enc = pd.Series(pd.Categorical(y).codes, index=y.index, name=y.name)
    else:
        y_enc = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=rs, stratify=y_enc
    )

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    transformers: list[tuple[str, Pipeline, Any]] = []
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            )
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return X_train, X_test, y_train, y_test, preprocessor


def save_reference_stats(X: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "columns": list(X.columns),
        "mean": X.select_dtypes(include=[np.number]).mean().to_dict(),
        "std": X.select_dtypes(include=[np.number]).std().replace(0, np.nan).to_dict(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
