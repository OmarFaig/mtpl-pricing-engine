from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mtpl.config import Settings

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def raw_freq_df() -> pd.DataFrame:
    return pd.read_parquet(FIXTURES_DIR / "raw_freq.parquet")


@pytest.fixture
def raw_sev_df() -> pd.DataFrame:
    return pd.read_parquet(FIXTURES_DIR / "raw_sev.parquet")


@pytest.fixture
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    from mtpl.config import get_config

    env = {
        "MTPL_PATHS__RAW_DIR": tmp_path / "raw",
        "MTPL_PATHS__BRONZE_DIR": tmp_path / "bronze",
        "MTPL_PATHS__SILVER_DIR": tmp_path / "silver",
        "MTPL_PATHS__GOLD_DIR": tmp_path / "gold",
        "MTPL_PATHS__ARTIFACTS_DIR": tmp_path / "artifacts",
        "MTPL_PATHS__LEVELS_PATH": tmp_path / "artifacts" / "feature_levels.json",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    monkeypatch.setenv("MTPL_PATHS__MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}")
    get_config.cache_clear()
    yield get_config()
    get_config.cache_clear()


@pytest.fixture
def bronze_pair(
    isolated_config: Settings, raw_freq_df: pd.DataFrame, raw_sev_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from mtpl import to_bronze
    from mtpl.manifest import write_manifest

    cfg = isolated_config
    cfg.paths.raw_dir.mkdir(parents=True, exist_ok=True)
    entries = {}
    for name, df in (("freq", raw_freq_df), ("sev", raw_sev_df)):
        path = cfg.paths.raw_dir / f"{name}.parquet"
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)
        entries[name] = (path, df)
    write_manifest("raw", entries, upstream_hash=None)

    to_bronze.build_bronze()
    freq = pd.read_parquet(cfg.paths.bronze_dir / "freq.parquet")
    sev = pd.read_parquet(cfg.paths.bronze_dir / "sev.parquet")
    return freq, sev


@pytest.fixture
def small_freq_df(rng: np.random.Generator) -> pd.DataFrame:
    n = 200
    return pd.DataFrame(
        {
            "IDpol": np.arange(1, n + 1, dtype="float64"),
            "ClaimNb": rng.poisson(0.1, n).astype("int64"),
            "Exposure": rng.uniform(0.05, 1.3, n),
            "Area": rng.choice(list("ABCDEF"), n),
            "VehPower": rng.integers(4, 15, n),
            "VehAge": rng.integers(0, 20, n),
            "DrivAge": rng.integers(18, 90, n),
            "BonusMalus": rng.integers(50, 200, n),
            "VehBrand": rng.choice(["B1", "B2", "B3"], n),
            "VehGas": rng.choice(["Regular", "Diesel"], n),
            "Density": rng.uniform(1, 5000, n),
            "Region": rng.choice(["R11", "R24", "R93"], n),
        }
    )


@pytest.fixture
def small_sev_df(small_freq_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    claim_ids = small_freq_df.loc[small_freq_df["ClaimNb"] > 0, "IDpol"]
    rows = []
    for pid in claim_ids:
        n_claims = int(small_freq_df.loc[small_freq_df["IDpol"] == pid, "ClaimNb"].iloc[0])
        for _ in range(n_claims):
            rows.append({"IDpol": pid, "ClaimAmount": float(rng.lognormal(7, 1.2))})
    return pd.DataFrame(rows)
