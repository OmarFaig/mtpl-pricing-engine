from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

logger = logging.getLogger(__name__)


def _two_way_group_split(
    df: pd.DataFrame, group_col: str, test_size: float, seed: int
) -> tuple[pd.Index, pd.Index]:
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=df[group_col]))
    return df.index[train_idx], df.index[test_idx]


def grouped_split(df: pd.DataFrame, seed: int, test_size: float, val_size: float) -> pd.Series:
    has_claim = df["claim_nb"] > 0
    labels = pd.Series(index=df.index, dtype=object)

    for stratum in (has_claim, ~has_claim):
        sub = df.loc[stratum]
        if sub.empty:
            continue
        train_val_idx, test_idx = _two_way_group_split(sub, "policy_id", test_size, seed)
        train_val = sub.loc[train_val_idx]
        val_frac_of_train_val = val_size / (1 - test_size)
        train_idx, val_idx = _two_way_group_split(
            train_val, "policy_id", val_frac_of_train_val, seed
        )
        labels.loc[train_idx] = "train"
        labels.loc[val_idx] = "val"
        labels.loc[test_idx] = "test"

    return labels


def assert_no_leakage(df: pd.DataFrame, label_col: str) -> None:
    counts = df.groupby("policy_id")[label_col].nunique()
    leaked = counts[counts > 1]
    if not leaked.empty:
        raise ValueError(f"policy_id appears under multiple splits: {sorted(leaked.index)[:20]}")
