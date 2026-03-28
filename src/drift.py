from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config import load_config
from src.data import load_classification_frame


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index for 1D arrays."""
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) < 2 or len(actual) < 2:
        return 0.0
    breakpoints = np.unique(np.percentile(expected, np.linspace(0, 100, bins + 1)))
    if len(breakpoints) < 2:
        breakpoints = np.array([expected.min(), expected.max()])

    e_counts, _ = np.histogram(expected, bins=breakpoints)
    a_counts, _ = np.histogram(actual, bins=breakpoints)
    e_pct = e_counts / max(e_counts.sum(), 1)
    a_pct = a_counts / max(a_counts.sum(), 1)
    eps = 1e-6
    e_pct = np.clip(e_pct, eps, 1.0)
    a_pct = np.clip(a_pct, eps, 1.0)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def run_drift_check(
    cfg: dict[str, Any] | None = None,
    current_sample_path: str | Path | None = None,
) -> dict[str, Any]:
    cfg = cfg or load_config()
    ref_path = Path(cfg["paths"]["reference_stats_path"])
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference stats missing: {ref_path}. Train the model first.")

    with open(ref_path, encoding="utf-8") as f:
        ref = json.load(f)

    drift_cfg = cfg["drift"]
    bins = drift_cfg["psi_bins"]
    psi_thr = drift_cfg["psi_alert_threshold"]
    ks_p = drift_cfg["ks_alert_pvalue"]

    if current_sample_path:
        current = pd.read_csv(current_sample_path)
    else:
        X, _, _ = load_classification_frame(cfg)
        current = X.sample(n=min(2000, len(X)), random_state=cfg["data"]["random_state"] + 1)

    report_dir = Path(cfg["paths"]["drift_report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)

    ref_sample_path = Path(cfg["paths"]["data_dir"]) / "reference_sample.csv"
    ref_sample = pd.read_csv(ref_sample_path) if ref_sample_path.exists() else None

    ref_means = ref.get("mean", {})
    results: list[dict[str, Any]] = []
    alerts = 0

    for col in ref_means:
        if col not in current.columns:
            continue
        s = pd.to_numeric(current[col], errors="coerce")
        actual = s.dropna().values
        if len(actual) < 10:
            continue

        if ref_sample is not None and col in ref_sample.columns:
            ref_vals = pd.to_numeric(ref_sample[col], errors="coerce").dropna().values
            if len(ref_vals) < 10:
                continue
            psi_val = _psi(ref_vals, actual, bins=bins)
            ks_stat, ks_pvalue = ks_2samp(ref_vals, actual)
        else:
            ref_mean = float(ref_means.get(col) or 0.0)
            ref_std = ref.get("std", {}).get(col) or 1.0
            if ref_std != ref_std or ref_std == 0:
                ref_std = 1.0
            synthetic_ref = np.random.normal(ref_mean, ref_std, size=min(len(actual), 5000))
            psi_val = _psi(synthetic_ref, actual, bins=bins)
            ks_stat, ks_pvalue = ks_2samp(synthetic_ref, actual[: len(synthetic_ref)])
        flag = bool(psi_val > psi_thr or ks_pvalue < ks_p)
        if flag:
            alerts += 1
        results.append(
            {
                "column": col,
                "psi": float(psi_val),
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pvalue),
                "alert": flag,
            }
        )

    summary = {
        "n_features_checked": len(results),
        "n_alerts": alerts,
        "features": results,
    }

    out_json = report_dir / "latest_drift_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    mlflow_cfg = cfg["mlflow"]
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])
    with mlflow.start_run(run_name="drift_monitoring") as run:
        mlflow.log_metrics(
            {
                "drift_n_alerts": float(alerts),
                "drift_max_psi": max((r["psi"] for r in results), default=0.0),
            }
        )
        mlflow.log_artifact(str(out_json), artifact_path="drift")

    return summary


if __name__ == "__main__":
    print(json.dumps(run_drift_check(), indent=2))
