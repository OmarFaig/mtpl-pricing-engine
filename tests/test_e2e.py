from __future__ import annotations

import pandas as pd

from mtpl import clean, features
from mtpl.config import get_config
from mtpl.evaluate import balance_check, normalized_gini, poisson_deviance
from mtpl.models import frequency, severity, tweedie


def test_full_pipeline_on_fixture_lands_in_plausible_range(
    bronze_pair: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    cfg = get_config()
    silver, quarantine = clean.build_silver()
    assert len(silver) > 0
    assert len(silver) + len(quarantine) == len(bronze_pair[0])

    features.build_gold(silver, "frequency")
    features.build_gold(silver, "severity")
    features.build_gold(silver, "pure_premium")

    freq_train = pd.read_parquet(cfg.paths.gold_dir / "frequency" / "train.parquet")
    freq_val = pd.read_parquet(cfg.paths.gold_dir / "frequency" / "val.parquet")
    freq_test = pd.read_parquet(cfg.paths.gold_dir / "frequency" / "test.parquet")

    freq_result = frequency.fit(frequency.build(cfg), freq_train, freq_val)
    freq_yhat_test = frequency.predict(freq_result.pipeline, freq_test)
    y_test_rate = (freq_test["claim_nb"] / freq_test["exposure"]).to_numpy()
    w_test = freq_test["exposure"].to_numpy()

    test_deviance = poisson_deviance(y_test_rate, freq_yhat_test, w_test)
    test_gini = normalized_gini(y_test_rate, freq_yhat_test, w_test)
    assert test_deviance > 0
    assert -0.2 <= test_gini <= 1.0

    # the exact balance property (test_evaluate.py) only holds for an unregularized
    # fit; the CV-selected alpha here is generally nonzero, so allow a small slack.
    freq_yhat_train = frequency.predict(freq_result.pipeline, freq_train)
    y_train_rate = (freq_train["claim_nb"] / freq_train["exposure"]).to_numpy()
    w_train = freq_train["exposure"].to_numpy()
    assert abs(balance_check(y_train_rate, freq_yhat_train, w_train) - 1.0) < 1e-2

    sev_train = pd.read_parquet(cfg.paths.gold_dir / "severity" / "train.parquet")
    sev_val = pd.read_parquet(cfg.paths.gold_dir / "severity" / "val.parquet")
    sev_result = severity.fit(severity.build(cfg), sev_train, sev_val)
    sev_yhat = severity.predict(sev_result.pipeline, sev_val)
    assert (sev_yhat > 0).all()

    pp_train = pd.read_parquet(cfg.paths.gold_dir / "pure_premium" / "train.parquet")
    pp_val = pd.read_parquet(cfg.paths.gold_dir / "pure_premium" / "val.parquet")
    pp_result = tweedie.fit(tweedie.build(cfg), pp_train, pp_val)
    assert pp_result.params["power"] in cfg.models.tweedie_power_grid
