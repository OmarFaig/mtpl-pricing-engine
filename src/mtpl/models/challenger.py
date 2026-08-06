from __future__ import annotations

import logging
from typing import Literal

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.pipeline import Pipeline

from mtpl.config import Settings, get_config
from mtpl.evaluate import poisson_deviance, tweedie_deviance
from mtpl.features import build_preprocessor
from mtpl.models import FitResult

logger = logging.getLogger(__name__)

MONOTONE_INCREASING_FEATURE = "bonus_malus"


def _monotone_constraints(feature_names: list[str]) -> list[int]:
    if MONOTONE_INCREASING_FEATURE not in feature_names:
        raise ValueError(f"{MONOTONE_INCREASING_FEATURE!r} not found in {feature_names}")
    return [1 if name == MONOTONE_INCREASING_FEATURE else 0 for name in feature_names]


def build(cfg: Settings, use_case: Literal["frequency", "pure_premium"]) -> Pipeline:
    preprocessor = build_preprocessor("gbm")
    objective = "poisson" if use_case == "frequency" else "tweedie"
    model = lgb.LGBMRegressor(
        objective=objective,
        tweedie_variance_power=cfg.models.lgbm_tweedie_variance_power,
        verbosity=-1,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def _offset(train: pd.DataFrame, use_case: str) -> np.ndarray | None:
    return np.log(train["exposure"].to_numpy()) if use_case == "frequency" else None


def _score(pipeline: Pipeline, val: pd.DataFrame, use_case: str, power: float) -> float:
    yhat = pipeline.predict(val)
    y = val["y"].to_numpy()
    w = val["sample_weight"].to_numpy()
    if use_case == "frequency":
        return poisson_deviance(y, yhat, w)
    return tweedie_deviance(y, yhat, w, power)


def _objective(
    trial: optuna.Trial,
    pipeline: Pipeline,
    train: pd.DataFrame,
    val: pd.DataFrame,
    use_case: str,
    cfg: Settings,
) -> float:
    m = cfg.models
    params = {
        "num_leaves": trial.suggest_int("num_leaves", *m.lgbm_num_leaves_range),
        "min_child_samples": trial.suggest_int(
            "min_child_samples", *m.lgbm_min_child_samples_range
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", *m.lgbm_learning_rate_range, log=True
        ),
        "n_estimators": trial.suggest_int("n_estimators", *m.lgbm_n_estimators_range),
        "reg_lambda": trial.suggest_float("reg_lambda", *m.lgbm_reg_lambda_range, log=True),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", *m.lgbm_colsample_bytree_range
        ),
        "subsample": trial.suggest_float("subsample", *m.lgbm_subsample_range),
    }
    pipeline.set_params(**{f"model__{k}": v for k, v in params.items()})

    feature_names = pipeline.named_steps["preprocess"].fit_transform(train).columns.tolist()
    pipeline.set_params(model__monotone_constraints=_monotone_constraints(feature_names))

    fit_kwargs = {"model__sample_weight": train["sample_weight"]}
    offset = _offset(train, use_case)
    if offset is not None:
        fit_kwargs["model__init_score"] = offset
    pipeline.fit(train, train["y"], **fit_kwargs)

    power = cfg.models.lgbm_tweedie_variance_power
    return _score(pipeline, val, use_case, power)


def fit(pipeline: Pipeline, train: pd.DataFrame, val: pd.DataFrame, use_case: str) -> FitResult:
    cfg = get_config()
    study_path = cfg.paths.artifacts_dir / f"optuna_{use_case}.db"
    study_path.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=f"challenger_{use_case}",
        storage=f"sqlite:///{study_path}",
        load_if_exists=True,
        direction="minimize",
    )
    study.optimize(
        lambda trial: _objective(trial, pipeline, train, val, use_case, cfg),
        n_trials=cfg.models.lgbm_n_trials,
        timeout=cfg.models.lgbm_timeout_s,
    )

    best_params = study.best_params
    pipeline.set_params(**{f"model__{k}": v for k, v in best_params.items()})
    feature_names = pipeline.named_steps["preprocess"].fit_transform(train).columns.tolist()
    pipeline.set_params(model__monotone_constraints=_monotone_constraints(feature_names))

    fit_kwargs = {"model__sample_weight": train["sample_weight"]}
    offset = _offset(train, use_case)
    if offset is not None:
        fit_kwargs["model__init_score"] = offset
    pipeline.fit(train, train["y"], **fit_kwargs)

    val_score = _score(pipeline, val, use_case, cfg.models.lgbm_tweedie_variance_power)
    metric_name = "val_poisson_deviance" if use_case == "frequency" else "val_tweedie_deviance"

    return FitResult(
        pipeline=pipeline,
        params=best_params,
        metrics={metric_name: val_score},
        artifacts={"optuna_study": study_path},
    )


def predict(pipeline: Pipeline, df: pd.DataFrame) -> np.ndarray:
    return pipeline.predict(df)
