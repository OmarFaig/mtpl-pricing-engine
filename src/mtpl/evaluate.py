from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import mean_gamma_deviance, mean_poisson_deviance, mean_tweedie_deviance


def poisson_deviance(y: np.ndarray, yhat: np.ndarray, w: np.ndarray) -> float:
    return float(mean_poisson_deviance(y, yhat, sample_weight=w))


def gamma_deviance(y: np.ndarray, yhat: np.ndarray, w: np.ndarray) -> float:
    return float(mean_gamma_deviance(y, yhat, sample_weight=w))


def tweedie_deviance(y: np.ndarray, yhat: np.ndarray, w: np.ndarray, power: float) -> float:
    return float(mean_tweedie_deviance(y, yhat, sample_weight=w, power=power))


def deviance_explained(
    y: np.ndarray,
    yhat: np.ndarray,
    w: np.ndarray,
    baseline_yhat: np.ndarray,
    kind: Literal["poisson", "gamma", "tweedie"],
    power: float | None = None,
) -> float:
    if kind == "tweedie":
        if power is None:
            raise ValueError("power is required for tweedie deviance_explained")
        d_model = tweedie_deviance(y, yhat, w, power)
        d_base = tweedie_deviance(y, baseline_yhat, w, power)
    elif kind == "poisson":
        d_model = poisson_deviance(y, yhat, w)
        d_base = poisson_deviance(y, baseline_yhat, w)
    else:
        d_model = gamma_deviance(y, yhat, w)
        d_base = gamma_deviance(y, baseline_yhat, w)
    return 1 - d_model / d_base


def lorenz_gini(y: np.ndarray, order_by: np.ndarray, w: np.ndarray) -> float:
    order = np.argsort(order_by, kind="mergesort")
    y_o, w_o = y[order], w[order]
    cum_w = np.concatenate([[0.0], np.cumsum(w_o) / w_o.sum()])
    cum_y = np.concatenate([[0.0], np.cumsum(y_o * w_o) / np.sum(y_o * w_o)])
    area = np.sum((cum_w[1:] - cum_w[:-1]) * (cum_y[1:] + cum_y[:-1]) / 2)
    return 1 - 2 * float(area)


def normalized_gini(y: np.ndarray, yhat: np.ndarray, w: np.ndarray) -> float:
    """Exposure-weighted Gini of the ordered Lorenz curve (sorted by yhat), normalized
    by the perfect model's Gini (sorted by y) so scores are comparable across portfolios."""
    model_gini = lorenz_gini(y, yhat, w)
    perfect_gini = lorenz_gini(y, y, w)
    return model_gini / perfect_gini


def _weighted_bins(yhat: np.ndarray, w: np.ndarray, n_bins: int) -> np.ndarray:
    order = np.argsort(yhat, kind="mergesort")
    cum_w = np.cumsum(w[order])
    bin_of_sorted = np.minimum((cum_w / cum_w[-1] * n_bins).astype(int), n_bins - 1)
    bins = np.empty_like(bin_of_sorted)
    bins[order] = bin_of_sorted
    return bins


def lift_table(y: np.ndarray, yhat: np.ndarray, w: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    bins = _weighted_bins(yhat, w, n_bins)
    df = pd.DataFrame({"bin": bins, "y": y, "yhat": yhat, "w": w})
    out = df.groupby("bin").apply(
        lambda d: pd.Series(
            {
                "n": len(d),
                "exposure": d["w"].sum(),
                "mean_pred": np.average(d["yhat"], weights=d["w"]),
                "mean_actual": np.average(d["y"], weights=d["w"]),
            }
        ),
        include_groups=False,
    ).reset_index()
    out["ratio"] = out["mean_actual"] / out["mean_pred"]
    return out


def calibration_table(
    y: np.ndarray, yhat: np.ndarray, w: np.ndarray, n_bins: int = 20
) -> pd.DataFrame:
    return lift_table(y, yhat, w, n_bins)


def balance_check(y: np.ndarray, yhat: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(yhat * w) / np.sum(y * w))


def segment_ae(df: pd.DataFrame, yhat: np.ndarray, by: list[str]) -> pd.DataFrame:
    rows = []
    for col in by:
        d = df[[col, "y", "sample_weight"]].assign(yhat=yhat)
        g = d.groupby(col, observed=True).apply(
            lambda d: pd.Series(
                {
                    "n": len(d),
                    "exposure": d["sample_weight"].sum(),
                    "mean_actual": np.average(d["y"], weights=d["sample_weight"]),
                    "mean_pred": np.average(d["yhat"], weights=d["sample_weight"]),
                }
            ),
            include_groups=False,
        ).reset_index(names="segment_value")
        g.insert(0, "segment", col)
        rows.append(g)
    out = pd.concat(rows, ignore_index=True)
    out["ae_ratio"] = out["mean_actual"] / out["mean_pred"]
    return out


def stability_report(
    fit_fn: Callable[[pd.DataFrame, int], dict[str, float]], data: pd.DataFrame, seeds: list[int]
) -> pd.DataFrame:
    records = [fit_fn(data, seed) for seed in seeds]
    df = pd.DataFrame(records)
    return df.agg(["mean", "std"]).T.reset_index(names="metric")
