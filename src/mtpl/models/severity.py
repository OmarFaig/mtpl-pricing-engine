from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import GammaRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from mtpl.config import Settings, get_config
from mtpl.evaluate import gamma_deviance
from mtpl.features import build_preprocessor
from mtpl.models import FitResult

logger = logging.getLogger(__name__)


def build(cfg: Settings) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor("glm")),
            ("model", GammaRegressor(max_iter=cfg.models.poisson_max_iter)),
        ]
    )


def _assert_claims_only(df: pd.DataFrame) -> None:
    if (df["claim_nb"] <= 0).any():
        raise ValueError("severity model requires claim_nb > 0 on every row")


def _cv_select_alpha(pipeline: Pipeline, train: pd.DataFrame, cfg: Settings) -> tuple[float, float]:
    gkf = GroupKFold(n_splits=cfg.models.cv_folds)
    y = train["y"]
    w = train["sample_weight"]
    groups = train["policy_id"]

    best_alpha, best_score = cfg.models.gamma_alpha_grid[0], np.inf
    for alpha in cfg.models.gamma_alpha_grid:
        pipeline.set_params(model__alpha=alpha)
        fold_scores = []
        for tr_idx, va_idx in gkf.split(train, groups=groups):
            pipeline.fit(train.iloc[tr_idx], y.iloc[tr_idx], model__sample_weight=w.iloc[tr_idx])
            yhat = pipeline.predict(train.iloc[va_idx])
            fold_scores.append(
                gamma_deviance(y.iloc[va_idx].to_numpy(), yhat, w.iloc[va_idx].to_numpy())
            )
        mean_score = float(np.mean(fold_scores))
        if mean_score < best_score:
            best_alpha, best_score = alpha, mean_score
    return best_alpha, best_score


def fit(pipeline: Pipeline, train: pd.DataFrame, val: pd.DataFrame) -> FitResult:
    cfg = get_config()
    _assert_claims_only(train)
    _assert_claims_only(val)

    alpha, cv_deviance = _cv_select_alpha(pipeline, train, cfg)
    pipeline.set_params(model__alpha=alpha)
    pipeline.fit(train, train["y"], model__sample_weight=train["sample_weight"])

    yhat_val = predict(pipeline, val)
    val_deviance = gamma_deviance(val["y"].to_numpy(), yhat_val, val["sample_weight"].to_numpy())

    return FitResult(
        pipeline=pipeline,
        params={"alpha": alpha},
        metrics={"cv_gamma_deviance": cv_deviance, "val_gamma_deviance": val_deviance},
        artifacts={},
    )


def predict(pipeline: Pipeline, df: pd.DataFrame) -> np.ndarray:
    return pipeline.predict(df)
