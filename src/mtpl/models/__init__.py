from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sklearn.pipeline import Pipeline


@dataclass
class FitResult:
    pipeline: Pipeline
    params: dict[str, Any]
    metrics: dict[str, float]
    artifacts: dict[str, Path] = field(default_factory=dict)
