#!/usr/bin/env python
"""Turn a NeuralHydrology run into things a human can read: predictions CSV, metrics, hydrographs.

    python scripts/evaluate.py                        # latest LSTM and Transformer runs, test period
    python scripts/evaluate.py --model lstm
    python scripts/evaluate.py --run-dir runs/lstm_6basins_0810_120000 --model lstm
    python scripts/evaluate.py --period validation

Writes
    outputs/predictions/<model>_<period>.csv        basin, date, obs, sim  (mm/day)
    outputs/metrics/<model>_<period>_metrics.csv    our metrics + NeuralHydrology's, per basin
    outputs/figures/hydrograph_<model>_<basin>_*.png, scatter_<model>.png, fdc_<model>_<basin>.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nhplay.data.camels import load_basin  # noqa: E402
from nhplay.evaluation import metrics as M  # noqa: E402
from nhplay.models.run import evaluate, latest_run, load_nh_metrics, load_results, results_dir  # noqa: E402
from nhplay.paths import FIGURE_DIR, METRIC_DIR, PREDICTION_DIR, RUNS_DIR, ensure_dirs  # noqa: E402
from nhplay.visualization.plots import plot_flow_duration_curve, plot_hydrograph, plot_scatter  # noqa: E402


def resolve_run(model: str, run_dir: str | None) -> Path:
    if run_dir:
        return Path(run_dir)
    pointer = RUNS_DIR / f"latest_{model}.txt"
    if pointer.exists():
        p = Path(pointer.read_text().strip())
        if p.exists():
            return p
    p = latest_run(f"{model}_6basins")
    if p is None:
        raise SystemExit(f"No run found for {model}. Train first: python scripts/train.py --model {model}")
    return p


def evaluate_model(model: str, run_dir: Path, period: str, zoom_year: int) -> pd.DataFrame:
    ensure_dirs()
    if not (results_dir(run_dir, period) / f"{period}_results.p").exists():
        print(f"running NeuralHydrology evaluation for {period} ...")
        evaluate(run_dir, period=period)
    results = load_results(run_dir, period)
    nh_metrics = load_nh_metrics(run_dir, period)

    long_rows, metric_rows = [], []
    for basin, df in results.items():
        df = df.copy()
        df["basin"] = basin
        long_rows.append(df.reset_index())

        sc = M.scorecard(df["obs"].values, df["sim"].values).as_dict()
        row = {"basin": basin, "model": model, "period": period, **sc}
        # NeuralHydrology's numbers for the same series (should agree with ours to ~1e-6)
        if basin in nh_metrics.index:
            row["nh_NSE"] = float(nh_metrics.loc[basin, "NSE"])
            row["nh_KGE"] = float(nh_metrics.loc[basin, "KGE"])
            for c in ("FHV", "FLV", "FMS", "Alpha-NSE", "Beta-NSE", "Pearson-r"):
                if c in nh_metrics.columns:
                    row[f"nh_{c}"] = float(nh_metrics.loc[basin, c])
        metric_rows.append(row)

        # figures per basin
        bd = load_basin(basin)
        rain = bd.timeseries["prcp(mm/day)"]
        name = str(bd.attributes.get("gauge_name", ""))
        plot_hydrograph(df["obs"], {model: df["sim"]}, f"{basin} {name} - {period} period, {model.upper()}",
                        rain=rain, savepath=FIGURE_DIR / f"hydrograph_{model}_{basin}_{period}_full.png")
        plot_hydrograph(df["obs"], {model: df["sim"]}, f"{basin} {name} - water year {zoom_year}, {model.upper()}",
                        rain=rain, start=f"{zoom_year - 1}-10-01", end=f"{zoom_year}-09-30",
                        savepath=FIGURE_DIR / f"hydrograph_{model}_{basin}_wy{zoom_year}.png")
        plot_flow_duration_curve(df["obs"], {model: df["sim"]}, f"{basin} flow duration curve ({period})",
                                 savepath=FIGURE_DIR / f"fdc_{model}_{basin}_{period}.png")

    preds = pd.concat(long_rows, ignore_index=True)[["basin", "date", "obs", "sim"]]
    preds.to_csv(PREDICTION_DIR / f"{model}_{period}.csv", index=False, float_format="%.4f")

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(METRIC_DIR / f"{model}_{period}_metrics.csv", index=False, float_format="%.4f")

    # one scatter figure with all basins pooled
    plot_scatter(preds.set_index("date")["obs"], {model: preds.set_index("date")["sim"]},
                 f"{model.upper()} - observed vs predicted, all basins, {period}",
                 savepath=FIGURE_DIR / f"scatter_{model}_{period}.png")

    if "nh_NSE" in metrics:
        gap = (metrics["nse"] - metrics["nh_NSE"]).abs().max()
        print(f"sanity: max |our NSE - NeuralHydrology NSE| = {gap:.2e}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["lstm", "transformer", "all"], default="all")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--period", choices=["train", "validation", "test"], default="test")
    ap.add_argument("--zoom-year", type=int, default=2011, help="water year to zoom the hydrograph into")
    args = ap.parse_args()
    models = ["lstm", "transformer"] if args.model == "all" else [args.model]
    pd.set_option("display.width", 160)
    for model in models:
        run_dir = resolve_run(model, args.run_dir if len(models) == 1 else None)
        print(f"\n=== {model}: {run_dir} ({args.period} period) ===")
        metrics = evaluate_model(model, run_dir, args.period, args.zoom_year)
        cols = ["basin", "nse", "kge", "r", "alpha", "beta", "pbias", "fhv"]
        print(metrics[cols].round(3).to_string(index=False))
        print(f"median NSE {metrics['nse'].median():.3f}   median KGE {metrics['kge'].median():.3f}")
        print(f"-> {PREDICTION_DIR / f'{model}_{args.period}.csv'}, {METRIC_DIR / f'{model}_{args.period}_metrics.csv'}")


if __name__ == "__main__":
    main()
