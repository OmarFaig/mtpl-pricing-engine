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

BRONZE_DIR = Path(__file__).resolve().parent.parent / "bronze"
SILVER_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = SILVER_DIR / "_manifest.json"

EXPOSURE_CAP = 1.0
CLAIM_NB_CAP = 4

POLICIES_SQL = """
    WITH capped_freq AS (
        SELECT
            id_pol,
            LEAST(claim_nb, ?)   AS claim_nb,
            LEAST(exposure, ?)   AS exposure,
            area, veh_power, veh_age, driv_age, bonus_malus,
            veh_brand, veh_gas, density, region
        FROM read_parquet(?)
    ),
    sev_agg AS (
        SELECT id_pol,
               SUM(claim_amount) AS claim_amount,
               COUNT(*)          AS claim_count
        FROM read_parquet(?)
        GROUP BY id_pol
    )
    SELECT
        f.id_pol, f.claim_nb, f.exposure, f.area, f.veh_power, f.veh_age,
        f.driv_age, f.bonus_malus, f.veh_brand, f.veh_gas, f.density, f.region,
        COALESCE(s.claim_amount, 0.0) AS claim_amount,
        COALESCE(s.claim_count, 0)    AS claim_count
    FROM capped_freq f
    LEFT JOIN sev_agg s USING (id_pol)
    ORDER BY f.id_pol
"""


def main() -> None:
    bronze_manifest_path = BRONZE_DIR / "_manifest.json"
    if not bronze_manifest_path.exists():
        raise SystemExit("silver - no bronze manifest found; run data/bronze/bronze.py first")

    freq_path = str(BRONZE_DIR / "freq.parquet")
    sev_path = str(BRONZE_DIR / "sev.parquet")
    policies_out = SILVER_DIR / "policies.parquet"

    con = duckdb.connect()

    exposure_capped = con.execute(
        f"SELECT count(*) FROM read_parquet(?) WHERE exposure > {EXPOSURE_CAP}", [freq_path]
    ).fetchone()[0]
    claim_nb_capped = con.execute(
        f"SELECT count(*) FROM read_parquet(?) WHERE claim_nb > {CLAIM_NB_CAP}", [freq_path]
    ).fetchone()[0]
    orphan_claims = con.execute(
        """
        SELECT count(*) FROM read_parquet(?) s
        WHERE NOT EXISTS (SELECT 1 FROM read_parquet(?) f WHERE f.id_pol = s.id_pol)
        """,
        [sev_path, freq_path],
    ).fetchone()[0]

    con.execute(
        f"COPY ({POLICIES_SQL}) TO '{policies_out}' (FORMAT parquet)",
        [CLAIM_NB_CAP, EXPOSURE_CAP, freq_path, sev_path],
    )
    logging.info("silver - wrote policies -> %s", policies_out)
    logging.info(
        "silver - capped exposure on %d rows, claim_nb on %d rows, dropped %d orphan claims",
        exposure_capped, claim_nb_capped, orphan_claims,
    )

    write_manifest(
        MANIFEST_PATH,
        layer="silver",
        command="python data/silver/silver.py",
        outputs={"policies": policies_out},
        upstream_hashes=load_upstream_hashes(bronze_manifest_path),
        extra={
            "exposure_cap": EXPOSURE_CAP,
            "claim_nb_cap": CLAIM_NB_CAP,
            "rows_exposure_capped": exposure_capped,
            "rows_claim_nb_capped": claim_nb_capped,
            "orphan_claims_dropped": orphan_claims,
        },
    )
    logging.info("silver - manifest written -> %s", MANIFEST_PATH)


if __name__ == "__main__":
    main()
