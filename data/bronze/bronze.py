import logging
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from pipeline.manifest import load_upstream_hashes, write_manifest

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

RAW_DIR = Path(__file__).resolve().parent.parent / "raw"
BRONZE_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = BRONZE_DIR / "_manifest.json"

FREQ_SQL = """
    SELECT
        CAST(IDpol AS BIGINT)      AS id_pol,
        CAST(ClaimNb AS INTEGER)   AS claim_nb,
        CAST(Exposure AS DOUBLE)   AS exposure,
        CAST(Area AS VARCHAR)      AS area,
        CAST(VehPower AS INTEGER)  AS veh_power,
        CAST(VehAge AS INTEGER)    AS veh_age,
        CAST(DrivAge AS INTEGER)   AS driv_age,
        CAST(BonusMalus AS INTEGER) AS bonus_malus,
        CAST(VehBrand AS VARCHAR)  AS veh_brand,
        CAST(VehGas AS VARCHAR)    AS veh_gas,
        CAST(Density AS DOUBLE)    AS density,
        CAST(Region AS VARCHAR)    AS region
    FROM read_parquet(?)
    ORDER BY id_pol
"""

SEV_SQL = """
    SELECT
        CAST(IDpol AS BIGINT)        AS id_pol,
        CAST(ClaimAmount AS DOUBLE)  AS claim_amount
    FROM read_parquet(?)
    ORDER BY id_pol, claim_amount
"""


def main() -> None:
    raw_manifest_path = RAW_DIR / "_manifest.json"
    if not raw_manifest_path.exists():
        raise SystemExit("bronze - no raw manifest found; run data/raw/raw.py first")

    con = duckdb.connect()

    freq_in = str(RAW_DIR / "freq.parquet")
    freq_out = BRONZE_DIR / "freq.parquet"
    con.execute(f"COPY ({FREQ_SQL}) TO '{freq_out}' (FORMAT parquet)", [freq_in])
    logging.info("bronze - wrote freq -> %s", freq_out)

    sev_in = str(RAW_DIR / "sev.parquet")
    sev_out = BRONZE_DIR / "sev.parquet"
    con.execute(f"COPY ({SEV_SQL}) TO '{sev_out}' (FORMAT parquet)", [sev_in])
    logging.info("bronze - wrote sev -> %s", sev_out)

    write_manifest(
        MANIFEST_PATH,
        layer="bronze",
        command="python data/bronze/bronze.py",
        outputs={"freq": freq_out, "sev": sev_out},
        upstream_hashes=load_upstream_hashes(raw_manifest_path),
    )
    logging.info("bronze - manifest written -> %s", MANIFEST_PATH)


if __name__ == "__main__":
    main()
