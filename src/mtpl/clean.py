from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mtpl.config import get_config, read_levels
from mtpl.manifest import read_manifest, write_manifest
from mtpl.validate import BRONZE_FREQ, BRONZE_SEV, SILVER_POLICIES, validate_strict

logger = logging.getLogger(__name__)

RENAME_MAP = {
    "IDpol": "policy_id",
    "ClaimNb": "claim_nb",
    "Exposure": "exposure",
    "VehPower": "veh_power",
    "VehAge": "veh_age",
    "DrivAge": "driv_age",
    "BonusMalus": "bonus_malus",
    "VehBrand": "veh_brand",
    "VehGas": "veh_gas",
    "Density": "density",
    "Region": "region",
    "Area": "area",
}

COVARIATE_COLS = [
    "Area",
    "VehPower",
    "VehAge",
    "DrivAge",
    "BonusMalus",
    "VehBrand",
    "VehGas",
    "Density",
    "Region",
]


@dataclass
class CleanStat:
    step: str
    rows_in: int
    rows_out: int
    rows_affected: int
    note: str


def cap_exposure(df: pd.DataFrame, cap: float) -> tuple[pd.DataFrame, CleanStat]:
    df = df.copy()
    affected = int((df["Exposure"] > cap).sum())
    df["Exposure"] = df["Exposure"].clip(upper=cap)
    stat = CleanStat("cap_exposure", len(df), len(df), affected, f"clipped Exposure to {cap}")
    return df, stat


def cap_claim_nb(df: pd.DataFrame, cap: int) -> tuple[pd.DataFrame, CleanStat]:
    df = df.copy()
    affected = int((df["ClaimNb"] > cap).sum())
    df["ClaimNb"] = df["ClaimNb"].clip(upper=cap)
    stat = CleanStat("cap_claim_nb", len(df), len(df), affected, f"clipped ClaimNb to {cap}")
    return df, stat


def flag_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, CleanStat]:
    df = df.copy()
    covariate_hash = pd.util.hash_pandas_object(df[COVARIATE_COLS], index=False)
    group_sizes = covariate_hash.map(covariate_hash.value_counts())
    df["n_covariate_duplicates"] = group_sizes.to_numpy()
    df["is_duplicate_group"] = df["n_covariate_duplicates"] > 1
    affected = int(df["is_duplicate_group"].sum())
    stat = CleanStat(
        "flag_duplicates", len(df), len(df), affected, "flagged rows sharing identical covariates"
    )
    return df, stat


def aggregate_severity(sev: pd.DataFrame) -> tuple[pd.DataFrame, CleanStat]:
    agg = sev.groupby("IDpol").agg(
        total_claim_amount=("ClaimAmount", "sum"),
        n_severity_rows=("ClaimAmount", "count"),
        max_claim_amount=("ClaimAmount", "max"),
    ).reset_index()
    stat = CleanStat(
        "aggregate_severity", len(sev), len(agg), len(sev), "grouped severity rows by IDpol"
    )
    return agg, stat


def cap_large_losses(
    sev: pd.DataFrame, threshold: float, quantile: float, method: str
) -> tuple[pd.DataFrame, CleanStat, float]:
    cap = sev["ClaimAmount"].quantile(quantile) if method == "quantile" else threshold
    per_claim = sev.copy()
    per_claim["capped_claim_amount"] = per_claim["ClaimAmount"].clip(upper=cap)
    per_claim["excess_claim_amount"] = per_claim["ClaimAmount"] - per_claim["capped_claim_amount"]
    affected = int((per_claim["excess_claim_amount"] > 0).sum())

    agg = per_claim.groupby("IDpol").agg(
        capped_claim_amount=("capped_claim_amount", "sum"),
        excess_claim_amount=("excess_claim_amount", "sum"),
    ).reset_index()
    note = f"capped {affected} claims at {cap:.2f} ({method})"
    stat = CleanStat("cap_large_losses", len(sev), len(agg), affected, note)
    return agg, stat, float(cap)


def join_frequency_severity(
    freq: pd.DataFrame, sev_agg: pd.DataFrame, loss_agg: pd.DataFrame
) -> tuple[pd.DataFrame, CleanStat]:
    df = freq.merge(sev_agg, on="IDpol", how="left").merge(loss_agg, on="IDpol", how="left")
    df["has_severity_record"] = df["n_severity_rows"].notna()
    df["n_severity_rows"] = df["n_severity_rows"].fillna(0).astype(int)
    df["claim_count_mismatch"] = df["n_severity_rows"] != df["ClaimNb"]

    zero_claim = df["ClaimNb"] == 0
    zero_fill_cols = [
        "total_claim_amount",
        "capped_claim_amount",
        "excess_claim_amount",
        "max_claim_amount",
    ]
    for col in zero_fill_cols:
        df.loc[zero_claim, col] = df.loc[zero_claim, col].fillna(0.0)

    missing_severity = (df["ClaimNb"] > 0) & ~df["has_severity_record"]
    affected = int(missing_severity.sum())
    stat = CleanStat(
        "join_frequency_severity",
        len(freq),
        len(df),
        affected,
        f"{affected} policies have claims but no severity record",
    )
    return df, stat


def rename_snake_case(df: pd.DataFrame) -> tuple[pd.DataFrame, CleanStat]:
    df = df.rename(columns=RENAME_MAP)
    stat = CleanStat("rename_snake_case", len(df), len(df), 0, "renamed columns to snake_case")
    return df, stat


def assert_levels(df: pd.DataFrame, levels: dict[str, list[str]]) -> tuple[pd.DataFrame, CleanStat]:
    checks = {"area": "area", "veh_gas": "veh_gas", "region": "region", "veh_brand": "veh_brand"}
    for col, level_key in checks.items():
        allowed = set(levels[level_key])
        seen = set(df[col].unique())
        unknown = seen - allowed
        if unknown:
            raise ValueError(f"unseen {col} levels: {sorted(unknown)}")
    stat = CleanStat("assert_levels", len(df), len(df), 0, "all categorical levels within config")
    return df, stat


def build_silver() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = get_config()
    bronze_manifest = read_manifest("bronze")
    silver_dir = cfg.paths.silver_dir
    silver_dir.mkdir(parents=True, exist_ok=True)
    quarantine_dir = silver_dir / "_quarantine"
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    freq = pd.read_parquet(cfg.paths.bronze_dir / "freq.parquet")
    sev = pd.read_parquet(cfg.paths.bronze_dir / "sev.parquet")
    freq = validate_strict(freq, BRONZE_FREQ)
    sev = validate_strict(sev, BRONZE_SEV)

    stats: list[CleanStat] = []

    freq, stat = cap_exposure(freq, cfg.data.exposure_cap)
    stats.append(stat)
    freq, stat = cap_claim_nb(freq, cfg.data.claim_nb_cap)
    stats.append(stat)
    freq, stat = flag_duplicates(freq)
    stats.append(stat)

    sev_agg, stat = aggregate_severity(sev)
    stats.append(stat)

    loss_agg, stat, large_loss_cap = cap_large_losses(
        sev, cfg.data.large_loss_threshold, cfg.data.large_loss_quantile, cfg.data.large_loss_method
    )
    stats.append(stat)

    joined, stat = join_frequency_severity(freq, sev_agg, loss_agg)
    stats.append(stat)

    renamed, stat = rename_snake_case(joined)
    stats.append(stat)

    levels = read_levels(cfg.paths.levels_path)
    silver, stat = assert_levels(renamed, levels)
    stats.append(stat)

    silver["claim_frequency"] = silver["claim_nb"] / silver["exposure"]
    silver["avg_claim_amount"] = np.where(
        silver["claim_nb"] > 0, silver["capped_claim_amount"] / silver["claim_nb"], np.nan
    )
    silver["pure_premium"] = silver["capped_claim_amount"] / silver["exposure"]
    large_loss_load = float(silver["excess_claim_amount"].sum() / silver["exposure"].sum())

    missing_severity_mask = (silver["claim_nb"] > 0) & ~silver["has_severity_record"]
    quarantine = silver.loc[missing_severity_mask].copy()
    quarantine["_reason"] = "claim_nb positive but no severity record"
    silver = silver.loc[~missing_severity_mask].copy()
    silver = validate_strict(silver, SILVER_POLICIES)

    silver_path = silver_dir / "policies.parquet"
    silver.to_parquet(silver_path, index=False)
    quarantine_path = quarantine_dir / "policies.parquet"
    quarantine.to_parquet(quarantine_path, index=False)

    logger.info("silver: %d policies kept, %d quarantined", len(silver), len(quarantine))

    write_manifest(
        "silver",
        {"policies": (silver_path, silver)},
        upstream_hash=bronze_manifest["layer_bundle_sha256"],
        extra={
            "clean_stats": [stat.__dict__ for stat in stats],
            "large_loss_load": large_loss_load,
            "large_loss_cap": large_loss_cap,
            "quarantine_rows": len(quarantine),
        },
    )

    large_loss_path = cfg.paths.artifacts_dir / "large_loss_load.json"
    large_loss_path.parent.mkdir(parents=True, exist_ok=True)
    large_loss_path.write_text(json.dumps({"large_loss_load": large_loss_load}) + "\n")
    logger.info("silver manifest written")
    return silver, quarantine
