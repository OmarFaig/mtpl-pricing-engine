from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from mtpl.config import get_config
from mtpl.evaluate import segment_ae

logger = logging.getLogger(__name__)


def psi(reference: np.ndarray, current: np.ndarray, bins: int) -> float:
    cfg = get_config()
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_frac = np.clip(ref_counts / ref_counts.sum(), cfg.monitor.psi_epsilon, None)
    cur_frac = np.clip(cur_counts / cur_counts.sum(), cfg.monitor.psi_epsilon, None)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def _status(score: float) -> str:
    cfg = get_config()
    if score >= cfg.monitor.psi_alert:
        return "alert"
    if score >= cfg.monitor.psi_watch:
        return "watch"
    return "stable"


def psi_report(
    reference_df: pd.DataFrame, current_df: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    cfg = get_config()
    rows = []
    for feature in features:
        ref = reference_df[feature].to_numpy()
        cur = current_df[feature].to_numpy()
        score = psi(ref, cur, cfg.monitor.psi_bins)
        rows.append({"feature": feature, "psi": score, "status": _status(score)})
    return pd.DataFrame(rows)


def psi_from_reference_bins(
    bin_edges: list[float], bin_counts: list[int], current: np.ndarray
) -> float:
    cfg = get_config()
    edges = np.array(bin_edges)
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.array(bin_counts, dtype=float)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_frac = np.clip(ref_counts / ref_counts.sum(), cfg.monitor.psi_epsilon, None)
    cur_frac = np.clip(cur_counts / cur_counts.sum(), cfg.monitor.psi_epsilon, None)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def prediction_drift(reference_preds: np.ndarray, current_preds: np.ndarray) -> float:
    return psi(reference_preds, current_preds, get_config().monitor.psi_bins)


def actual_vs_expected(df: pd.DataFrame, yhat: np.ndarray, by: list[str]) -> pd.DataFrame:
    return segment_ae(df, yhat, by)


def save_reference_distribution(df: pd.DataFrame, features: list[str]) -> Path:
    cfg = get_config()
    rows = []
    for feature in features:
        values = df[feature].to_numpy()
        edges = np.quantile(values, np.linspace(0, 1, cfg.monitor.psi_bins + 1))
        counts, _ = np.histogram(values, bins=edges)
        rows.append(
            {"feature": feature, "bin_edges": edges.tolist(), "bin_counts": counts.tolist()}
        )
    out = pd.DataFrame(rows)
    path = cfg.paths.artifacts_dir / "reference_distribution.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    logger.info("reference distribution written -> %s", path)
    return path
