from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    SplineTransformer,
    StandardScaler,
)

from mtpl.config import get_config
from mtpl.manifest import read_manifest, write_manifest
from mtpl.split import assert_no_leakage, grouped_split

logger = logging.getLogger(__name__)

NUMERIC_COLS = ["driv_age", "veh_age", "veh_power", "bonus_malus", "log_density"]
CATEGORICAL_COLS = [
    "area",
    "veh_brand",
    "veh_gas",
    "region",
    "veh_power_band",
    "driv_age_band",
    "veh_age_band",
]
ID_COLS = ["policy_id", "exposure", "split"]
_GOLD_TAIL = ["is_new_vehicle", "claim_nb", "y", "sample_weight"]

GOLD_COLUMNS = {
    "frequency": [*ID_COLS, *NUMERIC_COLS, *CATEGORICAL_COLS, *_GOLD_TAIL],
    "severity": [*ID_COLS, *NUMERIC_COLS, *CATEGORICAL_COLS, *_GOLD_TAIL],
    "pure_premium": [*ID_COLS, *NUMERIC_COLS, *CATEGORICAL_COLS, *_GOLD_TAIL],
}

UseCase = Literal["frequency", "severity", "pure_premium"]


def add_engineered_columns(df: pd.DataFrame) -> pd.DataFrame:
    cfg = get_config()
    df = df.copy()
    df["log_density"] = np.log(df["density"])
    df["driv_age_band"] = pd.cut(
        df["driv_age"], bins=cfg.features.driv_age_bins, right=False
    ).astype(str)
    df["veh_age_band"] = pd.cut(
        df["veh_age"], bins=cfg.features.veh_age_bins, right=False
    ).astype(str)
    df["veh_power_band"] = pd.cut(
        df["veh_power"], bins=cfg.features.veh_power_bins, right=False
    ).astype(str)
    df["is_new_vehicle"] = (df["veh_age"] <= 1).astype(int)
    return df


def add_target(df: pd.DataFrame, use_case: UseCase) -> pd.DataFrame:
    df = df.copy()
    if use_case == "frequency":
        df["y"] = df["claim_nb"] / df["exposure"]
        df["sample_weight"] = df["exposure"]
    elif use_case == "severity":
        df["y"] = df["avg_claim_amount"]
        df["sample_weight"] = df["claim_nb"]
    else:
        df["y"] = df["pure_premium"]
        df["sample_weight"] = df["exposure"]
    return df


def build_gold(silver: pd.DataFrame, use_case: UseCase) -> pd.DataFrame:
    cfg = get_config()
    df = silver
    if use_case == "severity":
        df = df.loc[df["claim_nb"] > 0]

    df = add_engineered_columns(df)
    df = add_target(df, use_case)
    df["split"] = grouped_split(df, cfg.split.seed, cfg.split.test_size, cfg.split.val_size)
    assert_no_leakage(df, "split")

    out = df[GOLD_COLUMNS[use_case]].reset_index(drop=True)

    gold_dir = cfg.paths.gold_dir / use_case
    gold_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, tuple] = {}
    for split_name, sub in out.groupby("split"):
        path = gold_dir / f"{split_name}.parquet"
        sub_out = sub.drop(columns="split").reset_index(drop=True)
        sub_out.to_parquet(path, index=False)
        written[split_name] = (path, sub_out)
        logger.info("gold %s/%s: %d rows -> %s", use_case, split_name, len(sub_out), path)

    silver_manifest = read_manifest("silver")
    write_manifest(
        "gold",
        written,
        upstream_hash=silver_manifest["layer_bundle_sha256"],
        extra={"use_case": use_case},
        manifest_dir=gold_dir,
    )
    return out


def build_preprocessor(spec: Literal["glm", "gbm"]) -> ColumnTransformer:
    cfg = get_config()

    if spec == "gbm":
        to_category = FunctionTransformer(lambda X: X.astype("category"))
        ct = ColumnTransformer(
            [
                ("numeric", "passthrough", NUMERIC_COLS),
                ("categorical", to_category, CATEGORICAL_COLS),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        return ct.set_output(transform="pandas")

    age_splines = SplineTransformer(
        degree=3, n_knots=cfg.features.spline_n_knots, extrapolation="constant"
    )
    transformers = [
        ("age_splines", age_splines, ["driv_age", "veh_age"]),
        ("scale", StandardScaler(), ["log_density", "bonus_malus"]),
        ("area", OrdinalEncoder(categories=[cfg.features.area_levels]), ["area"]),
        (
            "onehot",
            OneHotEncoder(drop="first", handle_unknown="error", sparse_output=False),
            ["veh_brand", "region", "veh_gas", "veh_power_band"],
        ),
    ]
    if cfg.features.use_interactions:
        # TODO: veh_age x veh_brand (numeric x categorical) needs a stateful
        # per-brand-slope encoding; only the numeric x numeric term is wired up here.
        transformers.append(
            (
                "driv_age_x_veh_power",
                FunctionTransformer(lambda X: (X[:, [0]] * X[:, [1]])),
                ["driv_age", "veh_power"],
            )
        )
    ct = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    return ct.set_output(transform="pandas")
