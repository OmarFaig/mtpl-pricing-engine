from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import pandas as pd
import typer
import uvicorn

from mtpl import clean, ingest, to_bronze
from mtpl.config import Settings, get_config
from mtpl.evaluate import balance_check, normalized_gini, poisson_deviance, stability_report
from mtpl.features import NUMERIC_COLS, add_engineered_columns, add_target, build_gold
from mtpl.manifest import content_hash
from mtpl.models import challenger as challenger_model
from mtpl.models import frequency, severity, tweedie
from mtpl.monitor import psi_report, save_reference_distribution
from mtpl.registry import log_run
from mtpl.split import grouped_split
from mtpl.validate import BRONZE_FREQ, BRONZE_SEV, validate_strict

_GLM_MODELS = {"frequency": frequency, "severity": severity, "tweedie": tweedie}

app = typer.Typer(add_completion=False)
logger = logging.getLogger(__name__)


@app.command("ingest")
def ingest_cmd() -> None:
    ingest.main()


@app.command()
def bronze() -> None:
    manifest_path = to_bronze.build_bronze()
    typer.echo(f"bronze written -> {manifest_path}")


@app.command()
def validate() -> None:
    cfg = get_config()
    freq = pd.read_parquet(cfg.paths.bronze_dir / "freq.parquet")
    sev = pd.read_parquet(cfg.paths.bronze_dir / "sev.parquet")
    validate_strict(freq, BRONZE_FREQ)
    validate_strict(sev, BRONZE_SEV)
    typer.echo(f"bronze/freq: {len(freq)} rows valid")
    typer.echo(f"bronze/sev: {len(sev)} rows valid")


@app.command()
def silver() -> None:
    kept, quarantined = clean.build_silver()
    typer.echo(f"silver: {len(kept)} kept, {len(quarantined)} quarantined")


@app.command()
def gold(use_case: str = typer.Option("pure_premium", "--use-case")) -> None:
    cfg = get_config()
    silver_df = pd.read_parquet(cfg.paths.silver_dir / "policies.parquet")
    out = build_gold(silver_df, use_case)  # type: ignore[arg-type]
    counts = out["split"].value_counts().to_dict()
    typer.echo(f"gold/{use_case}: {counts}")

    if use_case == "frequency":
        train_rows = out.loc[out["split"] == "train"]
        ref_path = save_reference_distribution(train_rows, NUMERIC_COLS)
        typer.echo(f"reference distribution -> {ref_path}")


def _load_gold_splits(use_case: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gold_dir = get_config().paths.gold_dir / use_case
    train = pd.read_parquet(gold_dir / "train.parquet")
    val = pd.read_parquet(gold_dir / "val.parquet")
    test = pd.read_parquet(gold_dir / "test.parquet")
    return train, val, test


@app.command()
def train(
    model: str,
    use_case: str = typer.Option("frequency", "--use-case"),
    register_as: str = typer.Option(None, "--register-as"),
) -> None:
    cfg = get_config()

    if model == "frequency":
        train_df, val_df, test_df = _load_gold_splits("frequency")
        pipeline = frequency.build(cfg)
        result = frequency.fit(pipeline, train_df, val_df)
        yhat_test = frequency.predict(result.pipeline, test_df)
        y_test = test_df["claim_nb"] / test_df["exposure"]
        w_test = test_df["exposure"].to_numpy()
        test_deviance = poisson_deviance(y_test.to_numpy(), yhat_test, w_test)
    elif model == "severity":
        train_df, val_df, test_df = _load_gold_splits("severity")
        pipeline = severity.build(cfg)
        result = severity.fit(pipeline, train_df, val_df)
        yhat_test = severity.predict(result.pipeline, test_df)
        y_test = test_df["y"]
        w_test = test_df["sample_weight"].to_numpy()
        test_deviance = None
    elif model == "tweedie":
        train_df, val_df, test_df = _load_gold_splits("pure_premium")
        pipeline = tweedie.build(cfg)
        result = tweedie.fit(pipeline, train_df, val_df)
        yhat_test = tweedie.predict(result.pipeline, test_df)
        y_test = test_df["y"]
        w_test = test_df["sample_weight"].to_numpy()
        test_deviance = None
    elif model == "challenger":
        train_df, val_df, test_df = _load_gold_splits(use_case)
        pipeline = challenger_model.build(cfg, use_case)  # type: ignore[arg-type]
        result = challenger_model.fit(pipeline, train_df, val_df, use_case)
        yhat_test = challenger_model.predict(result.pipeline, test_df)
        y_test = test_df["y"]
        w_test = test_df["sample_weight"].to_numpy()
        test_deviance = None
    else:
        raise typer.BadParameter(f"unknown model {model!r}")

    gini = normalized_gini(y_test.to_numpy(), yhat_test, w_test)
    balance = balance_check(y_test.to_numpy(), yhat_test, w_test)
    metrics = {"test_gini": gini, "test_balance": balance}
    if test_deviance is not None:
        metrics["test_deviance"] = test_deviance

    run_id = log_run(
        result,
        metrics,
        plots=[],
        data_hash=content_hash(train_df),
        run_name=f"{model}_{use_case}",
        register_as=register_as,
    )
    typer.echo(f"run {run_id}: {metrics}")


@app.command()
def evaluate(run_id: str = typer.Option(..., "--run-id")) -> None:
    cfg = get_config()
    mlflow.set_tracking_uri(cfg.paths.mlflow_tracking_uri)
    run = mlflow.get_run(run_id)
    for name, value in run.data.metrics.items():
        typer.echo(f"{name}: {value:.6f}")


@app.command()
def serve() -> None:
    cfg = get_config()
    uvicorn.run("api.main:app", host=cfg.server.host, port=cfg.server.port)


_MODEL_USE_CASE = {"frequency": "frequency", "severity": "severity", "tweedie": "pure_premium"}


def _fit_for_seed(model: str, silver: pd.DataFrame, seed: int, cfg: Settings) -> dict[str, float]:
    use_case = _MODEL_USE_CASE[model]
    df = silver.loc[silver["claim_nb"] > 0] if use_case == "severity" else silver
    df = add_engineered_columns(df)
    df = add_target(df, use_case)  # type: ignore[arg-type]
    df["split"] = grouped_split(df, seed, cfg.split.test_size, cfg.split.val_size)
    train_df, val_df, test_df = (df.loc[df["split"] == s] for s in ("train", "val", "test"))

    module = _GLM_MODELS[model]
    result = module.fit(module.build(cfg), train_df, val_df)
    yhat = module.predict(result.pipeline, test_df)
    y, w = test_df["y"].to_numpy(), test_df["sample_weight"].to_numpy()
    return {"gini": normalized_gini(y, yhat, w), "balance": balance_check(y, yhat, w)}


@app.command()
def stability(model: str = typer.Argument("frequency")) -> None:
    cfg = get_config()
    silver_df = pd.read_parquet(cfg.paths.silver_dir / "policies.parquet")
    seeds = [cfg.split.seed + i for i in range(cfg.split.n_seeds)]
    report = stability_report(
        lambda df, seed: _fit_for_seed(model, df, seed, cfg), silver_df, seeds
    )
    typer.echo(report.to_string(index=False))


@app.command()
def drift(input_path: Path = typer.Option(..., "--input")) -> None:
    cfg = get_config()
    reference = pd.read_parquet(cfg.paths.gold_dir / "frequency" / "train.parquet")
    current = pd.read_parquet(input_path)
    features = [f for f in NUMERIC_COLS if f in reference.columns and f in current.columns]
    report = psi_report(reference, current, features)
    typer.echo(report.to_string(index=False))


if __name__ == "__main__":
    app()
