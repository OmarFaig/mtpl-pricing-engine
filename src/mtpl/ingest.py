from __future__ import annotations

import logging
from pathlib import Path

import openml
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from mtpl.config import get_config
from mtpl.manifest import write_manifest

logger = logging.getLogger(__name__)


def fetch(openml_id: int) -> tuple[pd.DataFrame, openml.OpenMLDataset]:
    dataset = openml.datasets.get_dataset(openml_id)
    df, *_ = dataset.get_data(dataset_format="dataframe")
    return df, dataset


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    cfg = get_config()
    raw_dir = cfg.paths.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    entries: dict[str, tuple[Path, pd.DataFrame]] = {}
    sources: dict[str, dict] = {}
    for name, openml_id in cfg.data.openml_ids.items():
        logger.info("fetching openml dataset %s (%s)", openml_id, name)
        df, dataset = fetch(openml_id)
        out_path = raw_dir / f"{name}.parquet"
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out_path)
        entries[name] = (out_path, df)
        sources[name] = {
            "openml_id": openml_id,
            "openml_name": dataset.name,
            "openml_version": dataset.version,
            "openml_md5_checksum": dataset.md5_checksum,
        }
        logger.info("wrote %s: %d rows -> %s", name, len(df), out_path)

    write_manifest("raw", entries, upstream_hash=None, extra={"sources": sources})
    logger.info("raw manifest written")


if __name__ == "__main__":
    main()
