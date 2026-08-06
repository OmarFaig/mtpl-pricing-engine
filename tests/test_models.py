from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import PoissonRegressor

from mtpl import clean, features
from mtpl.cli import _fit_for_seed
from mtpl.config import Settings, get_config
from mtpl.evaluate import stability_report
from mtpl.models import frequency, severity, tweedie


def test_poisson_offset_equivalence() -> None:
    rng = np.random.default_rng(42)
    n = 2000
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    exposure = rng.uniform(0.1, 1.0, n)
    mu = exposure * np.exp(-1.0 + 0.3 * x1 - 0.2 * x2)
    claim_nb = rng.poisson(mu)
    X = np.column_stack([x1, x2])

    offset = np.log(exposure)
    sm_res = sm.GLM(claim_nb, sm.add_constant(X), family=sm.families.Poisson(), offset=offset).fit()

    # the same equivalence models/frequency.py relies on: rate as target,
    # exposure as sample_weight, no offset parameter needed.
    y_rate = claim_nb / exposure
    skl = PoissonRegressor(alpha=0.0, max_iter=10000, tol=1e-12)
    skl.fit(X, y_rate, sample_weight=exposure)

    np.testing.assert_allclose(skl.coef_, sm_res.params[1:], atol=1e-4)
    np.testing.assert_allclose(skl.intercept_, sm_res.params[0], atol=1e-4)


def _gold_splits(cfg: Settings, use_case: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    gold_dir = cfg.paths.gold_dir / use_case
    train = pd.read_parquet(gold_dir / "train.parquet")
    val = pd.read_parquet(gold_dir / "val.parquet")
    return train, val


def test_frequency_fit_predict(bronze_pair: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    cfg = get_config()
    silver, _ = clean.build_silver()
    features.build_gold(silver, "frequency")
    train, val = _gold_splits(cfg, "frequency")

    pipeline = frequency.build(cfg)
    result = frequency.fit(pipeline, train, val)
    yhat = frequency.predict(result.pipeline, val)
    assert (yhat > 0).all()
    assert "alpha" in result.params


def test_severity_fit_only_on_claims(bronze_pair: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    cfg = get_config()
    silver, _ = clean.build_silver()
    gold = features.build_gold(silver, "severity")
    assert (gold["claim_nb"] > 0).all()
    train, val = _gold_splits(cfg, "severity")

    pipeline = severity.build(cfg)
    result = severity.fit(pipeline, train, val)
    yhat = severity.predict(result.pipeline, val)
    assert (yhat > 0).all()


def test_tweedie_selects_power_from_grid(bronze_pair: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    cfg = get_config()
    silver, _ = clean.build_silver()
    features.build_gold(silver, "pure_premium")
    train, val = _gold_splits(cfg, "pure_premium")

    pipeline = tweedie.build(cfg)
    result = tweedie.fit(pipeline, train, val)
    assert result.params["power"] in cfg.models.tweedie_power_grid


def test_stability_report_has_mean_and_std(bronze_pair: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    cfg = get_config()
    silver, _ = clean.build_silver()
    report = stability_report(
        lambda df, seed: _fit_for_seed("frequency", df, seed, cfg), silver, seeds=[1, 2]
    )
    assert set(report["metric"]) == {"gini", "balance"}
    assert {"mean", "std"}.issubset(report.columns)


def test_combine_freq_sev_is_multiplicative() -> None:
    freq_pred = np.array([0.1, 0.2])
    sev_pred = np.array([1000.0, 2000.0])
    combined = tweedie.combine_freq_sev(freq_pred, sev_pred)
    np.testing.assert_allclose(combined, freq_pred * sev_pred)
