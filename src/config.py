from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    root = project_root()
    cfg_path = Path(path) if path else root / "config" / "default.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    tracking = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking:
        cfg.setdefault("mlflow", {})["tracking_uri"] = tracking

    # Resolve paths relative to project root
    paths = cfg.get("paths", {})
    for key in (
        "data_dir",
        "reference_stats_path",
        "inference_output_dir",
        "drift_report_dir",
        "explainability_dir",
    ):
        if key in paths and paths[key]:
            p = Path(paths[key])
            if not p.is_absolute():
                paths[key] = str(root / p)
    return cfg
