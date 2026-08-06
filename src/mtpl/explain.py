from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


@dataclass
class ReasonCode:
    feature: str
    value: Any
    factor: float
    direction: Literal["increases", "decreases"]


def _feature_groups(ct: ColumnTransformer) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for _name, transformer, columns in ct.transformers_:
        if transformer in ("drop", None):
            continue
        if transformer == "passthrough":
            mapping.update({col: col for col in columns})
            continue
        for out_name in transformer.get_feature_names_out(columns):
            source = max((c for c in columns if out_name.startswith(c)), key=len, default=out_name)
            mapping[out_name] = source
    return mapping


def glm_contributions(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Exact per-row multiplicative contribution of each source feature to a
    log-link GLM prediction: exp(beta_j * x_j) for every encoded column j,
    grouped back to the raw feature it came from (one-hot levels and spline
    basis functions collapse into their parent feature) and multiplied together."""
    ct = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    X_enc = ct.transform(X)
    coefs = pd.Series(model.coef_, index=X_enc.columns)
    groups = _feature_groups(ct)

    log_contrib = X_enc.mul(coefs, axis=1)
    log_contrib.columns = [groups[c] for c in log_contrib.columns]
    grouped_log = log_contrib.T.groupby(level=0).sum().T
    factors = np.exp(grouped_log)
    factors.index = X.index
    return factors


def top_reason_codes(contributions: pd.Series, raw_row: pd.Series, k: int = 3) -> list[ReasonCode]:
    ranked = contributions.map(lambda f: abs(np.log(f))).sort_values(ascending=False)
    codes = []
    for feature in ranked.index[:k]:
        factor = float(contributions[feature])
        value = raw_row[feature]
        codes.append(
            ReasonCode(
                feature=feature,
                value=value.item() if isinstance(value, np.generic) else value,
                factor=factor,
                direction="increases" if factor > 1 else "decreases",
            )
        )
    return codes


def shap_contributions(model: Any, X: pd.DataFrame) -> pd.DataFrame:
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    return pd.DataFrame(values, columns=X.columns, index=X.index)
