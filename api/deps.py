from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import mlflow
import pandas as pd

from mtpl.config import Settings, get_config
from mtpl.registry import load_model

logger = logging.getLogger(__name__)


@dataclass
class AppState:
    freq_model: Any
    sev_model: Any
    freq_version: str
    sev_version: str
    config: Settings
    reference_distribution: pd.DataFrame
    trained_at: str
    gold_data_hash: str
    large_loss_load: float

    @property
    def model_version(self) -> str:
        return f"freq:{self.freq_version}|sev:{self.sev_version}"


def _latest_version(client: Any, name: str) -> Any:
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise RuntimeError(f"no registered versions for model {name!r}")
    return max(versions, key=lambda v: int(v.version))


def load_app_state() -> AppState:
    cfg = get_config()
    mlflow.set_tracking_uri(cfg.paths.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()

    freq_latest = _latest_version(client, cfg.pricing.frequency_model_name)
    sev_latest = _latest_version(client, cfg.pricing.severity_model_name)
    freq_run = client.get_run(freq_latest.run_id)

    reference_path = cfg.paths.artifacts_dir / "reference_distribution.parquet"
    reference_distribution = pd.read_parquet(reference_path)
    large_loss_load = json.loads(
        (cfg.paths.artifacts_dir / "large_loss_load.json").read_text()
    )["large_loss_load"]

    return AppState(
        freq_model=load_model(cfg.pricing.frequency_model_name, freq_latest.version),
        sev_model=load_model(cfg.pricing.severity_model_name, sev_latest.version),
        freq_version=freq_latest.version,
        sev_version=sev_latest.version,
        config=cfg,
        reference_distribution=reference_distribution,
        trained_at=str(freq_run.info.start_time),
        gold_data_hash=freq_run.data.tags.get("gold_content_sha256", ""),
        large_loss_load=large_loss_load,
    )
