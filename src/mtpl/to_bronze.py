from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from mtpl.config import get_config, write_levels
from mtpl.manifest import read_manifest, sha256_file, write_manifest

logger = logging.getLogger(__name__)

BRONZE_FREQ_SCHEMA = pa.schema(
    [
        ("IDpol", pa.int64()),
        ("ClaimNb", pa.int64()),
        ("Exposure", pa.float64()),
        ("Area", pa.string()),
        ("VehPower", pa.int64()),
        ("VehAge", pa.int64()),
        ("DrivAge", pa.int64()),
        ("BonusMalus", pa.int64()),
        ("VehBrand", pa.string()),
        ("VehGas", pa.string()),
        ("Density", pa.float64()),
        ("Region", pa.string()),
        ("_ingested_at", pa.timestamp("us", tz="UTC")),
        ("_source_hash", pa.string()),
    ]
)

BRONZE_SEV_SCHEMA = pa.schema(
    [
        ("IDpol", pa.int64()),
        ("ClaimAmount", pa.float64()),
        ("_ingested_at", pa.timestamp("us", tz="UTC")),
        ("_source_hash", pa.string()),
    ]
)


def _cast_freq(df: pd.DataFrame, ingested_at: datetime, source_hash: str) -> pd.DataFrame:
    claim_nb_dtype = "int64" if df["ClaimNb"].isna().sum() == 0 else "Int64"
    out = pd.DataFrame(
        {
            "IDpol": df["IDpol"].astype("float64").astype("int64"),
            "ClaimNb": df["ClaimNb"].astype(claim_nb_dtype),
            "Exposure": df["Exposure"].astype("float64"),
            "Area": df["Area"].astype(str),
            "VehPower": df["VehPower"].astype("int64"),
            "VehAge": df["VehAge"].astype("int64"),
            "DrivAge": df["DrivAge"].astype("int64"),
            "BonusMalus": df["BonusMalus"].astype("int64"),
            "VehBrand": df["VehBrand"].astype(str),
            "VehGas": df["VehGas"].astype(str),
            "Density": df["Density"].astype("float64"),
            "Region": df["Region"].astype(str),
        }
    )
    out["_ingested_at"] = ingested_at
    out["_source_hash"] = source_hash
    return out


def _cast_sev(df: pd.DataFrame, ingested_at: datetime, source_hash: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "IDpol": df["IDpol"].astype("float64").astype("int64"),
            "ClaimAmount": df["ClaimAmount"].astype("float64"),
        }
    )
    out["_ingested_at"] = ingested_at
    out["_source_hash"] = source_hash
    return out


def profile_bronze(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        n_unique = s.nunique(dropna=True)
        top = s.value_counts(dropna=True)
        top_value = str(top.index[0]) if len(top) else None
        top_share = float(top.iloc[0]) / n if len(top) and n else 0.0
        is_numeric = pd.api.types.is_numeric_dtype(s)
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "n_unique": int(n_unique),
                "cardinality_ratio": float(n_unique) / n if n else 0.0,
                "n_null": int(s.isna().sum()),
                "min": str(s.min()) if is_numeric else None,
                "max": str(s.max()) if is_numeric else None,
                "top_value": top_value,
                "top_value_share": top_share,
            }
        )
    return pd.DataFrame(rows)


def build_bronze() -> Path:
    cfg = get_config()
    raw_manifest = read_manifest("raw")
    bronze_dir = cfg.paths.bronze_dir
    bronze_dir.mkdir(parents=True, exist_ok=True)

    ingested_at = datetime.now(UTC)
    specs = {
        "freq": (_cast_freq, BRONZE_FREQ_SCHEMA),
        "sev": (_cast_sev, BRONZE_SEV_SCHEMA),
    }

    entries: dict[str, tuple[Path, pd.DataFrame]] = {}
    for name, (cast_fn, schema) in specs.items():
        raw_path = cfg.paths.raw_dir / f"{name}.parquet"
        source_hash = sha256_file(raw_path)
        raw_df = pq.read_table(raw_path).to_pandas()
        df = cast_fn(raw_df, ingested_at, source_hash)

        out_path = bronze_dir / f"{name}.parquet"
        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
        pq.write_table(table, out_path)
        entries[name] = (out_path, df)
        logger.info("wrote bronze %s: %d rows -> %s", name, len(df), out_path)

        profile = profile_bronze(df)
        profile_path = bronze_dir / f"_profile_{name}.parquet"
        profile.to_parquet(profile_path, index=False)
        logger.info("wrote profile %s -> %s", name, profile_path)

    manifest_path = write_manifest(
        "bronze",
        entries,
        upstream_hash=raw_manifest["layer_bundle_sha256"],
    )
    logger.info("bronze manifest written")

    freq_df = entries["freq"][1]
    levels = {
        "region": sorted(freq_df["Region"].unique().tolist()),
        "veh_brand": sorted(freq_df["VehBrand"].unique().tolist()),
        "area": cfg.features.area_levels,
        "veh_gas": cfg.features.veh_gas_levels,
    }
    write_levels(levels, cfg.paths.levels_path)
    logger.info("feature levels written -> %s", cfg.paths.levels_path)
    return manifest_path
