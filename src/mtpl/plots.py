from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mtpl.config import get_config
from mtpl.evaluate import lorenz_gini

logger = logging.getLogger(__name__)


def _out_path(name: str) -> Path:
    plots_dir = get_config().paths.artifacts_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir / name


def plot_lorenz_curves(models: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, (y, yhat, w) in models.items():
        order = np.argsort(yhat, kind="mergesort")
        y_o, w_o = y[order], w[order]
        cum_w = np.concatenate([[0.0], np.cumsum(w_o) / w_o.sum()])
        cum_y = np.concatenate([[0.0], np.cumsum(y_o * w_o) / np.sum(y_o * w_o)])
        gini = lorenz_gini(y, yhat, w)
        ax.plot(cum_w, cum_y, label=f"{name} (gini={gini:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="random")
    ax.set_xlabel("cumulative exposure")
    ax.set_ylabel("cumulative claims")
    ax.set_title("Lorenz curve")
    ax.legend()
    path = _out_path("lorenz.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_lift(lift_table: pd.DataFrame) -> Path:
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.bar(lift_table["bin"], lift_table["exposure"], color="lightgray", label="exposure")
    ax1.set_xlabel("predicted risk decile")
    ax1.set_ylabel("exposure")
    ax2 = ax1.twinx()
    ax2.plot(lift_table["bin"], lift_table["mean_actual"], marker="o", label="actual")
    ax2.plot(lift_table["bin"], lift_table["mean_pred"], marker="s", label="predicted")
    ax2.set_ylabel("mean value")
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    ax1.set_title("lift by decile")
    path = _out_path("lift.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_calibration(calibration_table: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(calibration_table["mean_pred"], calibration_table["mean_actual"])
    lo = min(calibration_table["mean_pred"].min(), calibration_table["mean_actual"].min())
    hi = max(calibration_table["mean_pred"].max(), calibration_table["mean_actual"].max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_xlabel("mean predicted")
    ax.set_ylabel("mean actual")
    ax.set_title("calibration")
    path = _out_path("calibration.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_spline_effect(x_grid: np.ndarray, effect: np.ndarray, feature: str) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_grid, effect)
    ax.set_xlabel(feature)
    ax.set_ylabel("multiplicative effect")
    ax.set_title(f"{feature} spline effect")
    path = _out_path(f"spline_{feature}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_segment_ae(segment_table: pd.DataFrame, segment: str) -> Path:
    sub = segment_table.loc[segment_table["segment"] == segment]
    fig, ax = plt.subplots(figsize=(7, 4))
    widths = sub["exposure"] / sub["exposure"].max()
    ax.bar(sub["segment_value"].astype(str), sub["ae_ratio"], width=widths)
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_ylabel("actual / expected")
    ax.set_title(f"A/E by {segment}")
    fig.autofmt_xdate(rotation=45)
    path = _out_path(f"segment_ae_{segment}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_psi_bars(psi_table: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    status_colors = {"stable": "tab:green", "watch": "tab:orange", "alert": "tab:red"}
    colors = psi_table["status"].map(status_colors)
    ax.bar(psi_table["feature"], psi_table["psi"], color=colors)
    ax.set_ylabel("PSI")
    ax.set_title("population stability index by feature")
    fig.autofmt_xdate(rotation=45)
    path = _out_path("psi.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
