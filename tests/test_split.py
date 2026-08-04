from __future__ import annotations

import pandas as pd
import pytest

from mtpl import clean
from mtpl.split import assert_no_leakage, grouped_split


@pytest.fixture
def silver_df(bronze_pair: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    silver, _ = clean.build_silver()
    return silver


def test_no_policy_id_in_multiple_splits(silver_df: pd.DataFrame) -> None:
    df = silver_df.copy()
    df["split"] = grouped_split(df, seed=42, test_size=0.2, val_size=0.2)
    assert_no_leakage(df, "split")


def test_assert_no_leakage_raises_on_duplicated_id(silver_df: pd.DataFrame) -> None:
    df = silver_df.copy()
    df["split"] = grouped_split(df, seed=42, test_size=0.2, val_size=0.2)
    df.loc[df.index[0], "split"] = "train"
    df.loc[df.index[1], "policy_id"] = df.loc[df.index[0], "policy_id"]
    df.loc[df.index[1], "split"] = "test"
    with pytest.raises(ValueError, match="policy_id"):
        assert_no_leakage(df, "split")


def test_claim_rate_roughly_stable_across_splits(silver_df: pd.DataFrame) -> None:
    df = silver_df.copy()
    df["split"] = grouped_split(df, seed=42, test_size=0.2, val_size=0.2)
    rates = df.groupby("split")["claim_nb"].apply(lambda s: (s > 0).mean())
    assert rates.max() - rates.min() < 0.1
