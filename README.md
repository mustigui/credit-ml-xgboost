# Credit classification (XGBoost)

## Business problem

**What we’re trying to solve:** lenders need to **prioritize applications** and understand **what drives estimated repayment risk**. This project builds a **credit outcome classifier**: given applicant and loan attributes, the model estimates which outcome bucket the case falls into (e.g. **“good” vs “bad”** credit on the classic German Credit / `credit-g` benchmark from OpenML).

**Why it matters:** risk teams and executives care about **which factors the model leans on** (liquidity, loan size, history, etc.), not only a single accuracy number—so they can align with policy, explain trends, and plan reviews (always alongside **human judgment** and **compliance**).

**Scope:** this repo is a **technical learning / portfolio pipeline** on **public benchmark data**. It is **not** a production underwriting system. Real credit decisions need legal review, fair-lending controls, monitoring, and documented governance.

---

## Technical overview

End-to-end pipeline using **XGBoost**, **scikit-learn**, and **MLflow** for experiment tracking. Default data: OpenML **`credit-g`**. Includes **SHAP explainability** (technical plots plus an **`executive_brief.md`** for stakeholder-friendly wording), batch inference, drift checks, optional scheduling, **Docker**, and **Kubernetes** CronJob examples.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Or use Conda: `conda env create -f environment.yml`.

Copy `.env.example` to `.env` if you use environment-specific values (optional).

## Train

```bash
python -m src.train
```

Metrics and the model are logged to MLflow. With the default config, tracking uses `file:./mlruns`. A JSON summary is written to `data/last_train_summary.json` after a successful run.

**Explainability (same run):** under the `explainability/` artifacts you get SHAP plots, `explainability_summary.json`, and **`executive_brief.md`** (plain-language links between top factors and the predicted outcome).

## MLflow UI

```bash
python -m mlflow ui
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) from the project root (so `./mlruns` is found).

Docker MLflow server (separate backend—use if you point `MLFLOW_TRACKING_URI` at it):

```bash
docker compose up -d mlflow
```

## Other commands

| Command | Purpose |
|--------|---------|
| `python -m src.infer` | Batch inference from registered or latest model |
| `python -m src.drift` | Drift report vs. reference stats |
| `python -m src.scheduler` | APScheduler: retrain / inference on cron (see `config/default.yaml`) |

Compose (after `mlflow` is healthy):

```bash
docker compose --profile manual run --rm pipeline-train
docker compose --profile manual run --rm pipeline-infer
docker compose --profile manual run --rm pipeline-drift
```

## Configuration

Edit `config/default.yaml` (paths, OpenML dataset, hyperparameter search, drift thresholds, schedules, explainability sample size).

## Kubernetes

See `deploy/k8s/cronjobs.yaml` — replace the container image and wire `MLFLOW_TRACKING_URI` and storage for your cluster.

## License

MIT — see [LICENSE](LICENSE).
