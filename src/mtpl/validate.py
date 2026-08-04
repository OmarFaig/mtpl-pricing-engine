from __future__ import annotations

import logging

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from mtpl.config import get_config

logger = logging.getLogger(__name__)

cfg = get_config()

BRONZE_FREQ = pa.DataFrameSchema(
    {
        "IDpol": pa.Column(int, unique=True, nullable=False),
        "ClaimNb": pa.Column(int, pa.Check.ge(0)),
        # raw Exposure is uncapped; the exposure_cap bound only applies after clean.cap_exposure.
        "Exposure": pa.Column(float, pa.Check.gt(0)),
        "Area": pa.Column(str, pa.Check.isin(cfg.features.area_levels)),
        "VehPower": pa.Column(int),
        "VehAge": pa.Column(int),
        "DrivAge": pa.Column(int, pa.Check.in_range(18, 100)),
        "BonusMalus": pa.Column(int, pa.Check.in_range(50, 350)),
        "VehBrand": pa.Column(str),
        "VehGas": pa.Column(str, pa.Check.isin(cfg.features.veh_gas_levels)),
        "Density": pa.Column(float),
        "Region": pa.Column(str),
    },
    strict=False,
    coerce=False,
)

BRONZE_SEV = pa.DataFrameSchema(
    {
        "IDpol": pa.Column(int, nullable=False),
        "ClaimAmount": pa.Column(float, pa.Check.ge(cfg.data.min_claim_amount)),
    },
    strict=False,
    coerce=False,
)

SILVER_POLICIES = pa.DataFrameSchema(
    {
        "policy_id": pa.Column(int, unique=True, nullable=False),
        "claim_nb": pa.Column(int, pa.Check.ge(0)),
        "exposure": pa.Column(
            float, pa.Check.in_range(0, cfg.data.exposure_cap, include_min=False)
        ),
        "area": pa.Column(str, pa.Check.isin(cfg.features.area_levels)),
        "driv_age": pa.Column(int, pa.Check.in_range(18, 100)),
        "bonus_malus": pa.Column(int, pa.Check.in_range(50, 350)),
        "veh_gas": pa.Column(str, pa.Check.isin(cfg.features.veh_gas_levels)),
    },
    strict=False,
    coerce=False,
)


def validate_strict(df: pd.DataFrame, schema: pa.DataFrameSchema) -> pd.DataFrame:
    return schema.validate(df, lazy=False)


def validate_lenient(
    df: pd.DataFrame, schema: pa.DataFrameSchema
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        valid = schema.validate(df, lazy=True)
        return valid, df.iloc[0:0].assign(_reason=pd.Series(dtype=str))
    except SchemaErrors as err:
        failure_cases = err.failure_cases
        bad_idx = failure_cases["index"].dropna().astype(int).unique()
        quarantined = df.loc[df.index.isin(bad_idx)].copy()
        reasons = (
            failure_cases.dropna(subset=["index"])
            .astype({"index": int})
            .groupby("index")["check"]
            .apply(lambda s: ", ".join(sorted(set(s.astype(str)))))
        )
        quarantined["_reason"] = quarantined.index.map(reasons)
        valid = df.loc[~df.index.isin(bad_idx)].copy()
        logger.info(
            "validate_lenient: %d valid, %d quarantined out of %d",
            len(valid),
            len(quarantined),
            len(df),
        )
        return valid, quarantined
