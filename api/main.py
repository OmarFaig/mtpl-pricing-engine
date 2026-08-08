from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse

from api.deps import AppState, load_app_state
from api.schemas import DriftRequest, PricingRequest, PricingResponse, ReasonCode
from mtpl.explain import glm_contributions, top_reason_codes
from mtpl.features import ID_COLS, NUMERIC_COLS, add_engineered_columns
from mtpl.models.tweedie import combine_freq_sev
from mtpl.monitor import psi_from_reference_bins

logger = logging.getLogger(__name__)

RAW_FIELDS = [
    "policy_id",
    "exposure",
    "area",
    "veh_power",
    "veh_age",
    "driv_age",
    "bonus_malus",
    "veh_brand",
    "veh_gas",
    "density",
    "region",
]


class Metrics:
    def __init__(self) -> None:
        self.request_count = 0
        self.latencies_ms: list[float] = []
        self.predictions: list[float] = []

    def record(self, latency_ms: float, prediction: float | None = None) -> None:
        self.request_count += 1
        self.latencies_ms.append(latency_ms)
        if prediction is not None:
            self.predictions.append(prediction)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.mtpl = load_app_state()
    app.state.metrics = Metrics()
    logger.info("model loaded: %s", app.state.mtpl.model_version)
    yield


app = FastAPI(title="mtpl-pricing-engine", lifespan=lifespan)


@app.middleware("http")
async def logging_middleware(request: Request, call_next: Callable) -> Response:
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000

    model_version = getattr(getattr(request.app.state, "mtpl", None), "model_version", "unknown")
    response.headers["X-Model-Version"] = model_version
    response.headers["X-Request-ID"] = request_id

    logger.info(
        json.dumps(
            {
                "request_id": request_id,
                "path": request.url.path,
                "latency_ms": round(latency_ms, 2),
                "status": response.status_code,
                "model_version": model_version,
            }
        )
    )
    return response


def _request_frame(req: PricingRequest) -> pd.DataFrame:
    row = {field: getattr(req, field) for field in RAW_FIELDS}
    df = pd.DataFrame([row])
    return add_engineered_columns(df)


def _price_one(state: AppState, req: PricingRequest) -> PricingResponse:
    frame = _request_frame(req)
    freq_pred = float(state.freq_model.predict(frame)[0])
    sev_pred = float(state.sev_model.predict(frame)[0])
    pure_rate = float(combine_freq_sev(np.array([freq_pred]), np.array([sev_pred]))[0])

    premium_pure = pure_rate * req.exposure
    cfg = state.config
    premium_technical = (
        premium_pure * (1 + cfg.pricing.expense_load) * (1 + cfg.pricing.profit_load)
        + state.large_loss_load * req.exposure
    )

    # TODO: assumes the registered frequency model is the GLM route (linear coef_);
    # a promoted LightGBM challenger would need shap_contributions here instead.
    contributions = glm_contributions(state.freq_model, frame).iloc[0]
    raw_row = frame.iloc[0]
    reason_codes = [
        ReasonCode(feature=r.feature, value=r.value, factor=r.factor, direction=r.direction)
        for r in top_reason_codes(contributions, raw_row, k=3)
    ]

    return PricingResponse(
        premium_pure=premium_pure,
        premium_technical=premium_technical,
        frequency=freq_pred,
        severity=sev_pred,
        model_version=state.model_version,
        reason_codes=reason_codes,
    )


@app.post("/price", response_model=PricingResponse)
def price(req: PricingRequest, request: Request) -> PricingResponse:
    state: AppState = request.app.state.mtpl
    metrics: Metrics = request.app.state.metrics
    start = time.perf_counter()
    result = _price_one(state, req)
    metrics.record((time.perf_counter() - start) * 1000, result.premium_pure)
    return result


@app.post("/price/batch", response_model=list[PricingResponse])
def price_batch(reqs: list[PricingRequest], request: Request) -> list[PricingResponse]:
    state: AppState = request.app.state.mtpl
    metrics: Metrics = request.app.state.metrics
    start = time.perf_counter()
    results = [_price_one(state, req) for req in reqs]
    elapsed = (time.perf_counter() - start) * 1000
    for result in results:
        metrics.record(elapsed / len(results), result.premium_pure)
    return results


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request) -> dict[str, bool]:
    state: AppState | None = getattr(request.app.state, "mtpl", None)
    if state is None:
        return {"ready": False, "model_loaded": False, "registry_reachable": False}

    registry_reachable = True
    try:
        mlflow.set_tracking_uri(state.config.paths.mlflow_tracking_uri)
        mlflow.tracking.MlflowClient().search_registered_models(max_results=1)
    except mlflow.exceptions.MlflowException:
        registry_reachable = False

    return {
        "ready": registry_reachable,
        "model_loaded": True,
        "registry_reachable": registry_reachable,
    }


def _git_sha() -> str:
    # the runtime image has no git binary or .git dir; MTPL_GIT_SHA is set at build time.
    env_sha = os.environ.get("MTPL_GIT_SHA")
    if env_sha:
        return env_sha
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
    )
    return result.stdout.strip()


@app.get("/version")
def version(request: Request) -> dict[str, str]:
    state: AppState = request.app.state.mtpl
    return {
        "model_version": state.model_version,
        "gold_data_hash": state.gold_data_hash,
        "git_sha": _git_sha(),
        "trained_at": state.trained_at,
    }


@app.get("/metrics")
def metrics_endpoint(request: Request) -> PlainTextResponse:
    m: Metrics = request.app.state.metrics
    cfg = request.app.state.mtpl.config
    lines = [
        "# HELP mtpl_request_count_total Total number of pricing requests",
        "# TYPE mtpl_request_count_total counter",
        f"mtpl_request_count_total {m.request_count}",
        "# HELP mtpl_request_latency_ms Request latency in milliseconds",
        "# TYPE mtpl_request_latency_ms histogram",
    ]
    buckets = cfg.pricing.latency_buckets_ms
    for bucket in buckets:
        count = sum(1 for latency in m.latencies_ms if latency <= bucket)
        lines.append(f'mtpl_request_latency_ms_bucket{{le="{bucket}"}} {count}')
    lines.append(f'mtpl_request_latency_ms_bucket{{le="+Inf"}} {len(m.latencies_ms)}')
    lines.append(f"mtpl_request_latency_ms_sum {sum(m.latencies_ms)}")
    lines.append(f"mtpl_request_latency_ms_count {len(m.latencies_ms)}")
    lines.append("# HELP mtpl_prediction_mean Mean predicted pure premium")
    lines.append("# TYPE mtpl_prediction_mean gauge")
    mean_pred = float(np.mean(m.predictions)) if m.predictions else 0.0
    lines.append(f"mtpl_prediction_mean {mean_pred}")
    return PlainTextResponse("\n".join(lines) + "\n")


@app.post("/monitor/drift")
def monitor_drift(payload: DriftRequest, request: Request) -> dict[str, dict]:
    state: AppState = request.app.state.mtpl
    frames = [_request_frame(req) for req in payload.records]
    current = pd.concat(frames, ignore_index=True)

    ref = state.reference_distribution.set_index("feature")
    features = [f for f in [*NUMERIC_COLS, *ID_COLS] if f in ref.index and f in current.columns]
    report = {}
    for feature in features:
        row = ref.loc[feature]
        score = psi_from_reference_bins(
            list(row["bin_edges"]), list(row["bin_counts"]), current[feature].to_numpy()
        )
        report[feature] = {"psi": score}
    return report
