from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PathConfig(BaseModel):
    # cwd-relative, not derived from __file__: that would resolve into
    # site-packages once mtpl is installed non-editably (the docker image).
    raw_dir: Path = Field(default_factory=lambda: Path.cwd() / "data" / "raw")
    bronze_dir: Path = Field(default_factory=lambda: Path.cwd() / "data" / "bronze")
    silver_dir: Path = Field(default_factory=lambda: Path.cwd() / "data" / "silver")
    gold_dir: Path = Field(default_factory=lambda: Path.cwd() / "data" / "gold")
    artifacts_dir: Path = Field(default_factory=lambda: Path.cwd() / "artifacts")
    levels_path: Path = Field(
        default_factory=lambda: Path.cwd() / "artifacts" / "feature_levels.json"
    )
    mlflow_tracking_uri: str = Field(default_factory=lambda: f"file:{Path.cwd() / 'mlruns'}")


class DataConfig(BaseModel):
    openml_ids: dict[str, int] = Field(default_factory=lambda: {"freq": 41214, "sev": 41215})
    exposure_cap: float = 1.0
    claim_nb_cap: int = 4
    large_loss_threshold: float = 50_000.0
    large_loss_quantile: float = 0.995
    large_loss_method: Literal["fixed", "quantile"] = "fixed"
    min_claim_amount: float = 1.0


class SplitConfig(BaseModel):
    seed: int = 42
    test_size: float = 0.2
    val_size: float = 0.2
    n_seeds: int = 5


class FeatureConfig(BaseModel):
    driv_age_bins: list[float] = Field(
        default_factory=lambda: [18.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 100.0]
    )
    veh_age_bins: list[float] = Field(
        default_factory=lambda: [0.0, 1.0, 3.0, 5.0, 10.0, 20.0, 100.0]
    )
    veh_power_bins: list[float] = Field(default_factory=lambda: [0.0, 4.0, 6.0, 8.0, 10.0, 15.0])
    area_levels: list[str] = Field(default_factory=lambda: list("ABCDEF"))
    veh_gas_levels: list[str] = Field(default_factory=lambda: ["Regular", "Diesel"])
    region_levels: list[str] | None = None
    veh_brand_levels: list[str] | None = None
    spline_n_knots: int = 5
    use_interactions: bool = False


class ModelConfig(BaseModel):
    poisson_alpha_grid: list[float] = Field(
        default_factory=lambda: [0.0001, 0.001, 0.01, 0.1, 1.0]
    )
    gamma_alpha_grid: list[float] = Field(
        default_factory=lambda: [0.0001, 0.001, 0.01, 0.1, 1.0]
    )
    tweedie_power_grid: list[float] = Field(
        default_factory=lambda: [round(1.1 + 0.1 * i, 1) for i in range(9)]
    )
    cv_folds: int = 5
    poisson_max_iter: int = 1000
    fit_statsmodels_glm: bool = False
    lgbm_n_trials: int = 50
    lgbm_timeout_s: int = 600
    lgbm_tweedie_variance_power: float = 1.5
    lgbm_num_leaves_range: tuple[int, int] = (7, 255)
    lgbm_min_child_samples_range: tuple[int, int] = (5, 200)
    lgbm_learning_rate_range: tuple[float, float] = (0.01, 0.3)
    lgbm_n_estimators_range: tuple[int, int] = (50, 600)
    lgbm_reg_lambda_range: tuple[float, float] = (1e-4, 10.0)
    lgbm_colsample_bytree_range: tuple[float, float] = (0.5, 1.0)
    lgbm_subsample_range: tuple[float, float] = (0.5, 1.0)


class MonitorConfig(BaseModel):
    psi_bins: int = 10
    psi_watch: float = 0.1
    psi_alert: float = 0.25
    psi_epsilon: float = 1e-4


class PricingConfig(BaseModel):
    expense_load: float = 0.15
    profit_load: float = 0.05
    latency_buckets_ms: list[float] = Field(
        default_factory=lambda: [5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
    )
    frequency_model_name: str = "mtpl_frequency"
    severity_model_name: str = "mtpl_severity"
    pure_premium_model_name: str = "mtpl_pure_premium"


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 7860


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MTPL_", env_nested_delimiter="__")

    paths: PathConfig = Field(default_factory=PathConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)
    pricing: PricingConfig = Field(default_factory=PricingConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)


@lru_cache(maxsize=1)
def get_config() -> Settings:
    return Settings()


def write_levels(levels: dict[str, list[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(levels, indent=2, sort_keys=True) + "\n")


def read_levels(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"feature levels file not found at {path}; run bronze ingestion first"
        )
    return json.loads(path.read_text())
