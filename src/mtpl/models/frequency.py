from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from mtpl.config import Settings, get_config
from mtpl.evaluate import poisson_deviance
from mtpl.features import build_preprocessor
from mtpl.models import FitResult

logger = logging.getLogger(__name__)


def build(cfg: Settings) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor("glm")),
            ("model", PoissonRegressor(max_iter=cfg.models.poisson_max_iter)),
        ]
    )


def _cv_select_alpha(pipeline: Pipeline, train: pd.DataFrame, cfg: Settings) -> tuple[float, float]:
    gkf = GroupKFold(n_splits=cfg.models.cv_folds)
    y_rate = train["claim_nb"] / train["exposure"]
    w = train["exposure"]
    groups = train["policy_id"]

    best_alpha, best_score = cfg.models.poisson_alpha_grid[0], np.inf
    for alpha in cfg.models.poisson_alpha_grid:
        pipeline.set_params(model__alpha=alpha)
        fold_scores = []
        for tr_idx, va_idx in gkf.split(train, groups=groups):
            pipeline.fit(
                train.iloc[tr_idx], y_rate.iloc[tr_idx], model__sample_weight=w.iloc[tr_idx]
            )
            yhat = pipeline.predict(train.iloc[va_idx])
            fold_scores.append(
                poisson_deviance(y_rate.iloc[va_idx].to_numpy(), yhat, w.iloc[va_idx].to_numpy())
            )
        mean_score = float(np.mean(fold_scores))
        if mean_score < best_score:
            best_alpha, best_score = alpha, mean_score
    return best_alpha, best_score


def fit(pipeline: Pipeline, train: pd.DataFrame, val: pd.DataFrame) -> FitResult:
    cfg = get_config()
    alpha, cv_deviance = _cv_select_alpha(pipeline, train, cfg)
    pipeline.set_params(model__alpha=alpha)

    # PoissonRegressor has no offset param. Fitting the rate with exposure as
    # sample_weight is equivalent for a log link.
    y_rate = train["claim_nb"] / train["exposure"]
    pipeline.fit(train, y_rate, model__sample_weight=train["exposure"])

    yhat_val = predict(pipeline, val)
    y_val_rate = val["claim_nb"] / val["exposure"]
    val_deviance = poisson_deviance(y_val_rate.to_numpy(), yhat_val, val["exposure"].to_numpy())

    artifacts: dict[str, Path] = {}
    if cfg.models.fit_statsmodels_glm:
        artifacts["statsmodels_summary"] = _fit_statsmodels_variant(train, cfg)

    return FitResult(
        pipeline=pipeline,
        params={"alpha": alpha},
        metrics={"cv_poisson_deviance": cv_deviance, "val_poisson_deviance": val_deviance},
        artifacts=artifacts,
    )


def predict(pipeline: Pipeline, df: pd.DataFrame) -> np.ndarray:
    return pipeline.predict(df)


def _fit_statsmodels_variant(train: pd.DataFrame, cfg: Settings) -> Path:
    preprocessor = build_preprocessor("glm")
    X = preprocessor.fit_transform(train)
    X = sm.add_constant(X)
    offset = np.log(train["exposure"].to_numpy())
    model = sm.GLM(train["claim_nb"], X, family=sm.families.Poisson(), offset=offset)
    res = model.fit()
    summary_df = pd.DataFrame(
        {
            "param": res.params.index.astype(str),
            "coef": res.params.to_numpy(),
            "std_err": res.bse.to_numpy(),
            "p_value": res.pvalues.to_numpy(),
        }
    )
    out_path = cfg.paths.artifacts_dir / "statsmodels_frequency_glm.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_parquet(out_path, index=False)
    return out_path
