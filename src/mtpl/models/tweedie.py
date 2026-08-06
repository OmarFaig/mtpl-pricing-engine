from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import TweedieRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from mtpl.config import Settings, get_config
from mtpl.evaluate import tweedie_deviance
from mtpl.features import build_preprocessor
from mtpl.models import FitResult

logger = logging.getLogger(__name__)


def build(cfg: Settings) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor("glm")),
            (
                "model",
                TweedieRegressor(
                    power=cfg.models.tweedie_power_grid[0],
                    link="log",
                    max_iter=cfg.models.poisson_max_iter,
                ),
            ),
        ]
    )


def _cv_score(pipeline: Pipeline, train: pd.DataFrame, power: float, cfg: Settings) -> float:
    gkf = GroupKFold(n_splits=cfg.models.cv_folds)
    y = train["y"]
    w = train["sample_weight"]
    groups = train["policy_id"]

    pipeline.set_params(model__power=power)
    fold_scores = []
    for tr_idx, va_idx in gkf.split(train, groups=groups):
        pipeline.fit(train.iloc[tr_idx], y.iloc[tr_idx], model__sample_weight=w.iloc[tr_idx])
        yhat = pipeline.predict(train.iloc[va_idx])
        fold_scores.append(
            tweedie_deviance(y.iloc[va_idx].to_numpy(), yhat, w.iloc[va_idx].to_numpy(), power)
        )
    return float(np.mean(fold_scores))


def fit(pipeline: Pipeline, train: pd.DataFrame, val: pd.DataFrame) -> FitResult:
    cfg = get_config()

    best_power, best_score = cfg.models.tweedie_power_grid[0], np.inf
    for power in cfg.models.tweedie_power_grid:
        score = _cv_score(pipeline, train, power, cfg)
        if score < best_score:
            best_power, best_score = power, score

    pipeline.set_params(model__power=best_power)
    pipeline.fit(train, train["y"], model__sample_weight=train["sample_weight"])
    yhat_val = predict(pipeline, val)
    val_deviance = tweedie_deviance(
        val["y"].to_numpy(), yhat_val, val["sample_weight"].to_numpy(), best_power
    )

    return FitResult(
        pipeline=pipeline,
        params={"power": best_power},
        metrics={"cv_tweedie_deviance": best_score, "val_tweedie_deviance": val_deviance},
        artifacts={},
    )


def predict(pipeline: Pipeline, df: pd.DataFrame) -> np.ndarray:
    return pipeline.predict(df)


def combine_freq_sev(freq_pred: np.ndarray, sev_pred: np.ndarray) -> np.ndarray:
    """Multiplicative pure premium: frequency (claims per unit exposure) times
    average severity per claim, so route A (freq x sev) and route B (direct
    Tweedie) both land on the same units."""
    return freq_pred * sev_pred
