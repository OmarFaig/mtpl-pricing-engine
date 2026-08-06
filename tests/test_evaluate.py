from __future__ import annotations

import numpy as np
from sklearn.linear_model import PoissonRegressor

from mtpl.evaluate import balance_check, lift_table, normalized_gini


def test_gini_of_perfect_predictor_is_one() -> None:
    rng = np.random.default_rng(0)
    y = rng.exponential(1.0, 500)
    w = rng.uniform(0.1, 1.0, 500)
    assert normalized_gini(y, y, w) == 1.0


def test_gini_of_random_predictor_is_near_zero() -> None:
    rng = np.random.default_rng(0)
    y = rng.exponential(1.0, 20_000)
    w = rng.uniform(0.1, 1.0, 20_000)
    yhat = rng.uniform(0, 1, 20_000)
    assert abs(normalized_gini(y, yhat, w)) < 0.05


def test_balance_check_holds_for_poisson_glm_with_intercept() -> None:
    rng = np.random.default_rng(0)
    n = 3000
    x = rng.normal(size=n)
    exposure = rng.uniform(0.1, 1.0, n)
    mu = exposure * np.exp(0.2 + 0.4 * x)
    claim_nb = rng.poisson(mu)
    y_rate = claim_nb / exposure

    model = PoissonRegressor(alpha=0.0, max_iter=10000, tol=1e-12)
    model.fit(x.reshape(-1, 1), y_rate, sample_weight=exposure)
    yhat = model.predict(x.reshape(-1, 1))

    assert abs(balance_check(y_rate, yhat, exposure) - 1.0) < 1e-6


def test_lift_table_bins_sum_to_total_exposure() -> None:
    rng = np.random.default_rng(0)
    n = 5000
    y = rng.exponential(1.0, n)
    yhat = y + rng.normal(0, 0.5, n)
    w = rng.uniform(0.1, 1.0, n)

    table = lift_table(y, yhat, w, n_bins=10)
    assert abs(table["exposure"].sum() - w.sum()) < 1e-8
