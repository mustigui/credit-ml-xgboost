from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import load_config
from src.data import build_preprocess_and_split, save_reference_stats


def _build_pipeline(preprocessor, random_state: int) -> Pipeline:
    clf = XGBClassifier(
        random_state=random_state,
        eval_metric="logloss",
        enable_categorical=False,
    )
    return Pipeline([("prep", preprocessor), ("clf", clf)])


def _param_grid(cfg: dict[str, Any]) -> dict[str, list]:
    p = cfg["training"]["param_distributions"]
    return {f"clf__{k}": v for k, v in p.items()}


def run_training(config_path: str | Path | None = None) -> dict[str, Any]:
    cfg = load_config(config_path)
    mlflow_cfg = cfg["mlflow"]
    train_cfg = cfg["training"]

    X_train, X_test, y_train, y_test, preprocessor = build_preprocess_and_split(cfg)
    ref_path = cfg["paths"]["reference_stats_path"]
    save_reference_stats(X_train, ref_path)
    data_dir = Path(cfg["paths"]["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    ref_sample = X_train.sample(n=min(2000, len(X_train)), random_state=0)
    ref_sample.to_csv(data_dir / "reference_sample.csv", index=False)

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    pipe = _build_pipeline(preprocessor, train_cfg["random_state"])
    param_dist = _param_grid(cfg)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=train_cfg["n_iter"],
        scoring="roc_auc",
        cv=train_cfg["cv_folds"],
        random_state=train_cfg["random_state"],
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    with mlflow.start_run(run_name="random_search_xgb") as run:
        mlflow.log_params(
            {
                "cv_folds": train_cfg["cv_folds"],
                "n_iter": train_cfg["n_iter"],
                "dataset": cfg["data"]["openml_name"],
            }
        )
        mlflow.log_artifact(ref_path, artifact_path="reference")
        mlflow.log_artifact(str(data_dir / "reference_sample.csv"), artifact_path="reference")

        search.fit(X_train, y_train)
        best = search.best_estimator_
        proba = best.predict_proba(X_test)[:, 1]
        pred = best.predict(X_test)

        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1": float(f1_score(y_test, pred)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "best_cv_roc_auc": float(search.best_score_),
        }
        mlflow.log_metrics(metrics)
        mlflow.log_params({f"best__{k}": v for k, v in search.best_params_.items()})

        mlflow.sklearn.log_model(best, artifact_path="model")
        model_uri = f"runs:/{run.info.run_id}/model"

        try:
            mlflow.register_model(model_uri=model_uri, name=mlflow_cfg["registered_model_name"])
        except Exception:
            # Registry may be unavailable (e.g. file store only); training still succeeds
            pass

        summary = {
            "run_id": run.info.run_id,
            "model_uri": model_uri,
            "metrics": metrics,
            "best_params": search.best_params_,
        }
        Path(cfg["paths"]["data_dir"]).mkdir(parents=True, exist_ok=True)
        with open(Path(cfg["paths"]["data_dir"]) / "last_train_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary


if __name__ == "__main__":
    run_training()
    print("Training finished.")
