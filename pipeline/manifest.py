from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parquet_row_count(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def describe_outputs(outputs: dict[str, Path]) -> dict:
    return {
        name: {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(path),
            "rows": parquet_row_count(path),
            "bytes": path.stat().st_size,
        }
        for name, path in outputs.items()
    }


def load_upstream_hashes(manifest_path: Path) -> dict[str, str]:
    """Pull {output_name: sha256} out of an upstream layer's manifest."""
    manifest = json.loads(manifest_path.read_text())
    return {name: info["sha256"] for name, info in manifest["outputs"].items()}


def write_manifest(
    manifest_path: Path,
    *,
    layer: str,
    command: str,
    outputs: dict[str, Path],
    upstream_hashes: dict[str, str] | None = None,
    extra: dict | None = None,
) -> dict:
    manifest = {
        "layer": layer,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "git_commit": _git_commit(),
        "upstream_hashes": upstream_hashes or {},
        "outputs": describe_outputs(outputs),
    }
    if extra:
        manifest["extra"] = extra
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
