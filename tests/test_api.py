from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from mtpl import clean, features, to_bronze
from mtpl.manifest import write_manifest
from mtpl.models import frequency, severity
from mtpl.monitor import save_reference_distribution

VALID_PAYLOAD = {
    "policy_id": 1,
    "exposure": 0.5,
    "area": "A",
    "veh_power": 6,
    "veh_age": 3,
    "driv_age": 40,
    "bonus_malus": 100,
    "veh_brand": "B1",
    "veh_gas": "Regular",
    "density": 1000.0,
    "region": "R11",
}

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="module")
def fake_app_state(tmp_path_factory: pytest.TempPathFactory):
    from api.deps import AppState
    from mtpl.config import get_config as _get_config

    tmp_path = tmp_path_factory.mktemp("api_state")
    mp = pytest.MonkeyPatch()
    for key, sub in (
        ("RAW_DIR", "raw"),
        ("BRONZE_DIR", "bronze"),
        ("SILVER_DIR", "silver"),
        ("GOLD_DIR", "gold"),
        ("ARTIFACTS_DIR", "artifacts"),
    ):
        mp.setenv(f"MTPL_PATHS__{key}", str(tmp_path / sub))
    mp.setenv("MTPL_PATHS__LEVELS_PATH", str(tmp_path / "artifacts" / "feature_levels.json"))
    mp.setenv("MTPL_PATHS__MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}")
    # a single alpha is enough for a contract test of the API surface; the CV
    # grid search itself is exercised in test_models.py.
    mp.setenv("MTPL_MODELS__POISSON_ALPHA_GRID", "[0.1]")
    mp.setenv("MTPL_MODELS__GAMMA_ALPHA_GRID", "[0.1]")
    _get_config.cache_clear()

    cfg = _get_config()
    cfg.paths.raw_dir.mkdir(parents=True, exist_ok=True)
    raw_freq = pd.read_parquet(FIXTURES_DIR / "raw_freq.parquet")
    raw_sev = pd.read_parquet(FIXTURES_DIR / "raw_sev.parquet")
    entries = {}
    for name, df in (("freq", raw_freq), ("sev", raw_sev)):
        path = cfg.paths.raw_dir / f"{name}.parquet"
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)
        entries[name] = (path, df)
    write_manifest("raw", entries, upstream_hash=None)
    to_bronze.build_bronze()

    silver, _ = clean.build_silver()
    features.build_gold(silver, "frequency")
    features.build_gold(silver, "severity")

    gold_dir = cfg.paths.gold_dir
    freq_train = pd.read_parquet(gold_dir / "frequency" / "train.parquet")
    freq_val = pd.read_parquet(gold_dir / "frequency" / "val.parquet")
    freq_result = frequency.fit(frequency.build(cfg), freq_train, freq_val)

    sev_train = pd.read_parquet(gold_dir / "severity" / "train.parquet")
    sev_val = pd.read_parquet(gold_dir / "severity" / "val.parquet")
    sev_result = severity.fit(severity.build(cfg), sev_train, sev_val)

    ref_path = save_reference_distribution(freq_train, features.NUMERIC_COLS)
    large_loss_load = json.loads(
        (cfg.paths.artifacts_dir / "large_loss_load.json").read_text()
    )["large_loss_load"]

    state = AppState(
        freq_model=freq_result.pipeline,
        sev_model=sev_result.pipeline,
        freq_version="1",
        sev_version="1",
        config=cfg,
        reference_distribution=pd.read_parquet(ref_path),
        trained_at="2026-01-01T00:00:00",
        gold_data_hash="testhash",
        large_loss_load=large_loss_load,
    )
    yield state
    _get_config.cache_clear()
    mp.undo()


@pytest.fixture(scope="module")
def client(fake_app_state: object) -> Iterator[TestClient]:
    import api.main as main_module

    original = main_module.load_app_state
    main_module.load_app_state = lambda: fake_app_state
    try:
        with TestClient(main_module.app) as c:
            yield c
    finally:
        main_module.load_app_state = original


def test_price_valid_request_returns_three_reason_codes(client: TestClient) -> None:
    resp = client.post("/price", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["reason_codes"]) == 3
    assert body["currency"] == "EUR"
    assert body["premium_pure"] > 0


def test_price_out_of_range_driv_age_returns_422(client: TestClient) -> None:
    payload = {**VALID_PAYLOAD, "driv_age": 200}
    resp = client.post("/price", json=payload)
    assert resp.status_code == 422


def test_version_matches_loaded_model(client: TestClient, fake_app_state: object) -> None:
    resp = client.get("/version")
    assert resp.status_code == 200
    assert resp.json()["model_version"] == fake_app_state.model_version


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_price_batch(client: TestClient) -> None:
    resp = client.post("/price/batch", json=[VALID_PAYLOAD, VALID_PAYLOAD])
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_metrics_endpoint_is_prometheus_text(client: TestClient) -> None:
    client.post("/price", json=VALID_PAYLOAD)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "mtpl_request_count_total" in resp.text
