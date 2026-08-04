from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_hash(df: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(df, index=False).values
    return hashlib.sha256(values.tobytes()).hexdigest()


def _table_entry(path: Path, df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "cols": len(df.columns),
        "file_sha256": sha256_file(path),
        "content_sha256": content_hash(df),
        "schema": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }


def _manifest_path(layer: str) -> Path:
    from mtpl.config import get_config

    layer_dirs = {
        "raw": get_config().paths.raw_dir,
        "bronze": get_config().paths.bronze_dir,
        "silver": get_config().paths.silver_dir,
        "gold": get_config().paths.gold_dir,
    }
    return layer_dirs[layer] / "_manifest.json"


def write_manifest(
    layer: str,
    entries: dict[str, tuple[Path, pd.DataFrame]],
    upstream_hash: str | None,
    extra: dict | None = None,
    manifest_dir: Path | None = None,
) -> Path:
    tables = {name: _table_entry(path, df) for name, (path, df) in entries.items()}
    bundle = hashlib.sha256(
        "".join(sorted(t["content_sha256"] for t in tables.values())).encode()
    ).hexdigest()
    manifest = {
        "layer": layer,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "upstream_hash": upstream_hash,
        "tables": tables,
        "layer_bundle_sha256": bundle,
    }
    if extra:
        manifest["extra"] = extra
    path = (manifest_dir / "_manifest.json") if manifest_dir else _manifest_path(layer)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return path


def read_manifest(layer: str, manifest_dir: Path | None = None) -> dict:
    path = (manifest_dir / "_manifest.json") if manifest_dir else _manifest_path(layer)
    if not path.exists():
        raise FileNotFoundError(f"no manifest for layer {layer!r} at {path}")
    return json.loads(path.read_text())


def upstream_hash_of(layer: str) -> str:
    return read_manifest(layer)["layer_bundle_sha256"]


def parquet_row_count(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows
