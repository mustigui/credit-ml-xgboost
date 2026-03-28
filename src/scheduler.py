"""
Long-running scheduler: retrains twice per year and runs batch inference monthly.
Use in Docker with `command: python -m src.scheduler` or run locally for testing.
"""

from __future__ import annotations

import logging
import os
import sys

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from src.config import load_config
from src.drift import run_drift_check
from src.infer import run_inference
from src.train import run_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def job_retrain() -> None:
    log.info("Starting scheduled retrain")
    run_training()
    log.info("Retrain finished")


def job_inference() -> None:
    log.info("Starting scheduled inference")
    run_inference()
    log.info("Inference finished")


def job_drift() -> None:
    log.info("Starting drift check")
    run_drift_check()
    log.info("Drift check finished")


def main() -> None:
    cfg = load_config()
    tz = cfg["schedules"].get("timezone", "UTC")

    # Optional: run drift on the same schedule as monthly inference
    run_drift_with_infer = os.environ.get("SCHEDULE_DRIFT_WITH_INFERENCE", "1") == "1"

    sched = BlockingScheduler(timezone=tz)

    # Twice per year: Jan 1 and Jul 1 at 00:00 (matches config comment)
    sched.add_job(
        job_retrain,
        CronTrigger(month="1,7", day=1, hour=0, minute=0, timezone=tz),
    )
    # Monthly on the 1st at 00:00
    sched.add_job(
        job_inference,
        CronTrigger(day=1, hour=0, minute=0, timezone=tz),
    )
    if run_drift_with_infer:
        sched.add_job(
            job_drift,
            CronTrigger(day=1, hour=0, minute=0, timezone=tz),
        )

    log.info(
        "Scheduler started: retrain=Jan/Jul 1 00:00, inference+drift=monthly 1st 00:00 tz=%s",
        tz,
    )
    sched.start()


if __name__ == "__main__":
    main()
