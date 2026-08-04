from __future__ import annotations

import pandas as pd
import pytest

from mtpl import clean, features


@pytest.fixture
def engineered_df(bronze_pair: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    silver, _ = clean.build_silver()
    return features.add_engineered_columns(silver)


def test_preprocessor_column_count_matches_across_splits(engineered_df: pd.DataFrame) -> None:
    n = len(engineered_df)
    train = engineered_df.iloc[: int(n * 0.7)]
    test = engineered_df.iloc[int(n * 0.7) :]

    ct = features.build_preprocessor("glm")
    train_out = ct.fit_transform(train)
    test_out = ct.transform(test)
    assert train_out.shape[1] == test_out.shape[1]


def test_unknown_category_raises(engineered_df: pd.DataFrame) -> None:
    n = len(engineered_df)
    train = engineered_df.iloc[: int(n * 0.7)].copy()
    test = engineered_df.iloc[int(n * 0.7) :].copy()

    ct = features.build_preprocessor("glm")
    ct.fit(train)
    test.loc[test.index[0], "veh_brand"] = "UNSEEN_BRAND"
    with pytest.raises(ValueError):
        ct.transform(test)


def test_gbm_preprocessor_passes_categories_as_pandas_category(engineered_df: pd.DataFrame) -> None:
    ct = features.build_preprocessor("gbm")
    out = ct.fit_transform(engineered_df)
    for col in features.CATEGORICAL_COLS:
        assert out[col].dtype.name == "category"
