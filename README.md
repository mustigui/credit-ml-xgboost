# Credit classification (XGBoost)

End-to-end pipeline for credit default / classification using **XGBoost**, **scikit-learn**, and **MLflow** for experiment tracking. Data comes from OpenML (`credit-g` by default). Includes batch inference, drift checks, optional scheduling, **Docker**, and **Kubernetes** CronJob examples.

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

## MLflow UI (Docker)

```bash
docker compose up -d mlflow
```

Open [http://localhost:5000](http://localhost:5000). Set `MLFLOW_TRACKING_URI=http://localhost:5000` when running training on the host, or use the compose services below.

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

Edit `config/default.yaml` (paths, OpenML dataset, hyperparameter search, drift thresholds, schedules).

## Kubernetes

See `deploy/k8s/cronjobs.yaml` — replace the container image and wire `MLFLOW_TRACKING_URI` and storage for your cluster.

## License

MIT — see [LICENSE](LICENSE).
