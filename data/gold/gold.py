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

SILVER_DIR = Path(__file__).resolve().parent.parent / "silver"
GOLD_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = GOLD_DIR / "_manifest.json"

TEST_SHARE_PCT = 20  # deterministic hash(id_pol) % 100 < TEST_SHARE_PCT -> test

TRAINING_SQL = """
    SELECT
        id_pol, exposure, claim_nb, claim_amount, claim_count,
        claim_amount / exposure AS pure_premium,
        area, veh_power, veh_age, driv_age, bonus_malus,
        veh_brand, veh_gas, density, region,
        CASE WHEN hash(id_pol) % 100 < ? THEN 'test' ELSE 'train' END AS split
    FROM read_parquet(?)
    ORDER BY id_pol
"""


def main() -> None:
    silver_manifest_path = SILVER_DIR / "_manifest.json"
    if not silver_manifest_path.exists():
        raise SystemExit("gold - no silver manifest found; run data/silver/silver.py first")

    policies_path = str(SILVER_DIR / "policies.parquet")
    training_out = GOLD_DIR / "training.parquet"

    con = duckdb.connect()
    con.execute(
        f"COPY ({TRAINING_SQL}) TO '{training_out}' (FORMAT parquet)",
        [TEST_SHARE_PCT, policies_path],
    )
    logging.info("gold - wrote training -> %s", training_out)

    split_counts = con.execute(
        f"SELECT split, count(*) FROM read_parquet('{training_out}') GROUP BY split"
    ).fetchall()
    split_summary = {split: count for split, count in split_counts}
    logging.info("gold - split sizes: %s", split_summary)

    write_manifest(
        MANIFEST_PATH,
        layer="gold",
        command="python data/gold/gold.py",
        outputs={"training": training_out},
        upstream_hashes=load_upstream_hashes(silver_manifest_path),
        extra={"test_share_pct": TEST_SHARE_PCT, "split_counts": split_summary},
    )
    logging.info("gold - manifest written -> %s", MANIFEST_PATH)


if __name__ == "__main__":
    main()
