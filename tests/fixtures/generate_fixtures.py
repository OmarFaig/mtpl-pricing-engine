from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FIXTURE_SEED = 42
N_POLICIES = 1000

FIXTURES_DIR = Path(__file__).resolve().parent


def _build_freq(rng: np.random.Generator) -> pd.DataFrame:
    n = N_POLICIES
    driv_age = rng.integers(18, 90, n)
    bonus_malus = rng.integers(50, 200, n)
    exposure = rng.uniform(0.05, 1.3, n)

    # bias claim rate with age and bonus_malus so the fixture carries a real signal
    # for the offset-equivalence and gini tests, instead of pure noise.
    risk = 0.05 + 0.08 * (driv_age < 25) + 0.15 * (bonus_malus - 50) / 150
    claim_nb = rng.poisson(risk * exposure)

    return pd.DataFrame(
        {
            "IDpol": np.arange(1, n + 1, dtype="float64"),
            "ClaimNb": claim_nb.astype("int64"),
            "Exposure": exposure,
            "Area": rng.choice(list("ABCDEF"), n),
            "VehPower": rng.integers(4, 15, n),
            "VehAge": rng.integers(0, 20, n),
            "DrivAge": driv_age,
            "BonusMalus": bonus_malus,
            "VehBrand": rng.choice(["B1", "B2", "B3", "B4"], n),
            "VehGas": rng.choice(["Regular", "Diesel"], n),
            "Density": rng.uniform(1, 5000, n),
            "Region": rng.choice(["R11", "R24", "R93"], n),
        }
    )


def _build_sev(freq: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for pid, claim_nb in zip(freq["IDpol"], freq["ClaimNb"], strict=True):
        for _ in range(int(claim_nb)):
            rows.append({"IDpol": pid, "ClaimAmount": float(rng.lognormal(7.0, 1.2))})
    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(FIXTURE_SEED)
    freq = _build_freq(rng)
    sev = _build_sev(freq, rng)
    freq.to_parquet(FIXTURES_DIR / "raw_freq.parquet", index=False)
    sev.to_parquet(FIXTURES_DIR / "raw_sev.parquet", index=False)


if __name__ == "__main__":
    main()
