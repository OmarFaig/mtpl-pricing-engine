import logging
import sys
from pathlib import Path

import openml
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from pipeline.manifest import write_manifest

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

RAW_DIR = Path(__file__).resolve().parent
FREQ_PATH = RAW_DIR / "freq.parquet"
SEV_PATH = RAW_DIR / "sev.parquet"
MANIFEST_PATH = RAW_DIR / "_manifest.json"

DATASETS = {
    "freq": {"openml_id": 41214, "path": FREQ_PATH},  # freMTPL2freq
    "sev": {"openml_id": 41215, "path": SEV_PATH},  # freMTPL2sev
}


def fetch(openml_id: int) -> tuple[pa.Table, openml.OpenMLDataset]:
    dataset = openml.datasets.get_dataset(openml_id)
    df, *_ = dataset.get_data(dataset_format="dataframe")
    return pa.Table.from_pandas(df, preserve_index=False), dataset


def main() -> None:
    sources = {}
    for name, spec in DATASETS.items():
        logging.info("raw - fetching OpenML dataset %s (%s)", spec["openml_id"], name)
        table, dataset = fetch(spec["openml_id"])
        pq.write_table(table, spec["path"])
        sources[name] = {
            "openml_id": spec["openml_id"],
            "openml_name": dataset.name,
            "openml_version": dataset.version,
            "openml_md5_checksum": dataset.md5_checksum,
        }
        logging.info("raw - wrote %s (%d rows) -> %s", name, table.num_rows, spec["path"])

    write_manifest(
        MANIFEST_PATH,
        layer="raw",
        command="python data/raw/raw.py",
        outputs={"freq": FREQ_PATH, "sev": SEV_PATH},
        extra={"sources": sources},
    )
    logging.info("raw - manifest written -> %s", MANIFEST_PATH)


if __name__ == "__main__":
    main()
