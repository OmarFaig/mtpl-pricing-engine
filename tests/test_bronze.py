from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from mtpl.to_bronze import (
    BRONZE_FREQ_SCHEMA,
    BRONZE_SEV_SCHEMA,
    _cast_freq,
    _cast_sev,
    profile_bronze,
)

INGESTED_AT = datetime.now(UTC)


def test_bronze_freq_row_count_and_schema(small_freq_df: pd.DataFrame, tmp_path: Path) -> None:
    df = _cast_freq(small_freq_df, INGESTED_AT, "deadbeef")
    out = tmp_path / "freq.parquet"
    pq.write_table(pa.Table.from_pandas(df, schema=BRONZE_FREQ_SCHEMA, preserve_index=False), out)

    assert len(df) == len(small_freq_df)
    assert pq.read_schema(out).equals(BRONZE_FREQ_SCHEMA)


def test_bronze_freq_id_and_claim_nb(small_freq_df: pd.DataFrame) -> None:
    df = _cast_freq(small_freq_df, INGESTED_AT, "deadbeef")
    assert df["IDpol"].is_unique
    assert df["IDpol"].notna().all()
    assert pd.api.types.is_integer_dtype(df["ClaimNb"])
    assert df["ClaimNb"].sum() >= 0


def test_bronze_sev_schema(small_sev_df: pd.DataFrame, tmp_path: Path) -> None:
    df = _cast_sev(small_sev_df, INGESTED_AT, "deadbeef")
    out = tmp_path / "sev.parquet"
    pq.write_table(pa.Table.from_pandas(df, schema=BRONZE_SEV_SCHEMA, preserve_index=False), out)

    assert len(df) == len(small_sev_df)
    assert pq.read_schema(out).equals(BRONZE_SEV_SCHEMA)


def test_profile_bronze_columns(small_freq_df: pd.DataFrame) -> None:
    df = _cast_freq(small_freq_df, INGESTED_AT, "deadbeef")
    profile = profile_bronze(df)
    assert set(profile["column"]) == set(df.columns)
    assert (profile["n_null"] >= 0).all()
