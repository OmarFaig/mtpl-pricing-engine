from __future__ import annotations

import pandas as pd
import pytest

from mtpl import clean
from mtpl.manifest import read_manifest


def test_silver_plus_quarantine_equals_bronze(
    bronze_pair: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    freq, _ = bronze_pair
    silver, quarantine = clean.build_silver()
    assert len(silver) + len(quarantine) == len(freq)


def test_clean_stats_recorded_for_every_step(
    bronze_pair: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    clean.build_silver()
    stats = read_manifest("silver")["extra"]["clean_stats"]
    steps = {s["step"] for s in stats}
    assert steps == {
        "cap_exposure",
        "cap_claim_nb",
        "flag_duplicates",
        "aggregate_severity",
        "cap_large_losses",
        "join_frequency_severity",
        "rename_snake_case",
        "assert_levels",
    }
    for s in stats:
        assert s["rows_affected"] >= 0


def test_cap_exposure_idempotent(raw_freq_df: pd.DataFrame) -> None:
    once, _ = clean.cap_exposure(raw_freq_df, cap=1.0)
    twice, _ = clean.cap_exposure(once, cap=1.0)
    pd.testing.assert_frame_equal(once, twice)


def test_cap_claim_nb_idempotent(raw_freq_df: pd.DataFrame) -> None:
    once, _ = clean.cap_claim_nb(raw_freq_df, cap=4)
    twice, _ = clean.cap_claim_nb(once, cap=4)
    pd.testing.assert_frame_equal(once, twice)


def test_assert_levels_raises_on_unseen_value(raw_freq_df: pd.DataFrame) -> None:
    df = raw_freq_df.rename(columns=clean.RENAME_MAP)
    df.loc[df.index[0], "area"] = "Z"
    levels = {
        "area": list("ABCDEF"),
        "veh_gas": ["Regular", "Diesel"],
        "region": sorted(df["region"].unique()),
        "veh_brand": sorted(df["veh_brand"].unique()),
    }
    with pytest.raises(ValueError, match="area"):
        clean.assert_levels(df, levels)
