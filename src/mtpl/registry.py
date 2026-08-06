from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import mlflow

from mtpl.config import get_config
from mtpl.models import FitResult

logger = logging.getLogger(__name__)

# mlflow >= 3 refuses the plain ./mlruns file store unless this is opted back in.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
    )
    return result.stdout.strip()


def log_run(
    fit_result: FitResult,
    metrics: dict[str, float],
    plots: list[Path],
    data_hash: str,
    run_name: str,
    register_as: str | None = None,
) -> str:
    cfg = get_config()
    mlflow.set_tracking_uri(cfg.paths.mlflow_tracking_uri)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(fit_result.params)
        mlflow.log_metrics({**fit_result.metrics, **metrics})
        mlflow.set_tags({"gold_content_sha256": data_hash, "git_sha": _git_sha()})
        for path in plots:
            mlflow.log_artifact(str(path), artifact_path="plots")
        for path in fit_result.artifacts.values():
            mlflow.log_artifact(str(path), artifact_path="artifacts")
        mlflow.sklearn.log_model(
            fit_result.pipeline,
            artifact_path="model",
            registered_model_name=register_as,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        run_id = run.info.run_id
    logger.info("logged run %s: %d metrics, register_as=%s", run_id, len(metrics), register_as)
    return run_id


def load_model(name: str, version: str = "latest") -> Any:
    cfg = get_config()
    mlflow.set_tracking_uri(cfg.paths.mlflow_tracking_uri)
    resolved_version = version
    if version == "latest":
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{name}'")
        if not versions:
            raise ValueError(f"no registered versions for model {name!r}")
        resolved_version = str(max(int(v.version) for v in versions))
    uri = f"models:/{name}/{resolved_version}"
    return mlflow.sklearn.load_model(uri)
