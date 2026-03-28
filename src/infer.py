from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from src.config import load_config
from src.data import load_classification_frame


def _load_model(cfg: dict[str, Any]):
    mlflow_cfg = cfg["mlflow"]
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    name = mlflow_cfg["registered_model_name"]

    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(name, stages=["Production"])
        if versions:
            uri = f"models:/{name}/Production"
            return mlflow.sklearn.load_model(uri)
        versions = client.search_model_versions(f"name='{name}'")
        if versions:
            latest = max(versions, key=lambda v: int(v.version))
            uri = f"models:/{name}/{latest.version}"
            return mlflow.sklearn.load_model(uri)
    except Exception:
        pass

    exp = mlflow.get_experiment_by_name(mlflow_cfg["experiment_name"])
    if exp is None:
        raise RuntimeError("No experiment found. Run training first.")
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError("No runs found. Run training first.")
    run_id = runs.iloc[0]["run_id"]
    return mlflow.sklearn.load_model(f"runs:/{run_id}/model")


def run_inference(
    cfg: dict[str, Any] | None = None,
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    cfg = cfg or load_config()
    out_dir = Path(cfg["paths"]["inference_output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if input_path:
        df = pd.read_csv(Path(input_path))
    else:
        X, _, _ = load_classification_frame(cfg)
        df = X.sample(n=min(500, len(X)), random_state=cfg["data"]["random_state"])

    model = _load_model(cfg)
    proba_full = model.predict_proba(df)
    proba = proba_full[:, 1] if proba_full.shape[1] == 2 else proba_full
    pred = model.predict(df)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(output_path) if output_path else out_dir / f"predictions_{ts}.csv"
    out_df = df.copy()
    out_df["pred"] = pred
    out_df["pred_proba"] = proba
    out_df.to_csv(out_path, index=False)

    meta = {
        "output": str(out_path),
        "n_rows": int(len(out_df)),
        "utc_time": ts,
    }
    with open(out_dir / f"inference_meta_{ts}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return out_path


if __name__ == "__main__":
    run_inference()
