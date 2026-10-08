#!/usr/bin/env python
"""Phase 3 - LSTM vs Transformer on exactly the same basins, inputs, target and test period.

    python scripts/compare.py                 # uses runs/latest_lstm.txt and runs/latest_transformer.txt
    python scripts/compare.py --zoom-year 2011

Writes
    outputs/metrics/comparison_test.csv          per basin x model
    outputs/metrics/comparison_summary.csv       median over basins per model
    outputs/metrics/comparison_peaks.csv         the biggest observed flood per basin and what each model predicted
    outputs/metrics/comparison.md                markdown tables for the README
    outputs/figures/compare_hydrograph_<basin>_wy<year>.png, compare_nse.png, compare_kge.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nhplay.data.camels import load_basin  # noqa: E402
from nhplay.evaluation.compare import comparison_table, markdown_table, summarise  # noqa: E402
from nhplay.models.run import describe_run, load_results  # noqa: E402
from nhplay.paths import FIGURE_DIR, METRIC_DIR, ensure_dirs  # noqa: E402
from nhplay.visualization.plots import plot_hydrograph, plot_metric_bars  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import resolve_run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoom-year", type=int, default=2011)
    ap.add_argument("--period", default="test")
    args = ap.parse_args()
    ensure_dirs()

    runs = {m: resolve_run(m, None) for m in ("lstm", "transformer")}
    table = comparison_table(runs, period=args.period)
    summary = summarise(table)
    table.to_csv(METRIC_DIR / f"comparison_{args.period}.csv", index=False, float_format="%.4f")
    summary.to_csv(METRIC_DIR / "comparison_summary.csv", float_format="%.4f")

    # hydrographs with both models, and the biggest flood of the test period per basin
    results = {m: load_results(p, args.period) for m, p in runs.items()}
    peak_rows = []
    for basin in results["lstm"]:
        obs = results["lstm"][basin]["obs"]
        sims = {m: results[m][basin]["sim"] for m in results}
        bd = load_basin(basin)
        name = str(bd.attributes.get("gauge_name", ""))
        rain = bd.timeseries["prcp(mm/day)"]
        plot_hydrograph(obs, sims, f"{basin} {name} - water year {args.zoom_year} (test period)", rain=rain,
                        start=f"{args.zoom_year - 1}-10-01", end=f"{args.zoom_year}-09-30",
                        savepath=FIGURE_DIR / f"compare_hydrograph_{basin}_wy{args.zoom_year}.png")
        plot_hydrograph(obs, sims, f"{basin} {name} - whole test period", rain=None,
                        savepath=FIGURE_DIR / f"compare_hydrograph_{basin}_{args.period}_full.png")
        peak_day = obs.idxmax()
        row = {"basin": basin, "name": name, "peak_date": peak_day.date(), "rain_that_day_mm": round(float(rain.loc[peak_day]), 1),
               "rain_prev_3_days_mm": round(float(rain.loc[peak_day - pd.Timedelta(days=3): peak_day - pd.Timedelta(days=1)].sum()), 1),
               "observed_mm_per_day": round(float(obs.loc[peak_day]), 2)}
        for m in sims:
            row[f"{m}_mm_per_day"] = round(float(sims[m].loc[peak_day]), 2)
        peak_rows.append(row)
        # zoom on the biggest flood: +-30 days
        plot_hydrograph(obs, sims, f"{basin} {name} - biggest observed flood of the test period ({peak_day.date()})",
                        rain=rain, start=peak_day - pd.Timedelta(days=30), end=peak_day + pd.Timedelta(days=30),
                        savepath=FIGURE_DIR / f"compare_peak_{basin}.png")
    peaks = pd.DataFrame(peak_rows)
    peaks.to_csv(METRIC_DIR / "comparison_peaks.csv", index=False)

    plot_metric_bars(table, "nse", f"NSE per basin ({args.period} period, higher is better, 1 = perfect)",
                     savepath=FIGURE_DIR / "compare_nse.png")
    plot_metric_bars(table, "kge", f"KGE per basin ({args.period} period, higher is better, 1 = perfect)",
                     savepath=FIGURE_DIR / "compare_kge.png")

    info = {m: describe_run(p) for m, p in runs.items()}
    md = ["# LSTM vs Transformer (" + args.period + " period: " + "2004-10-01 to 2014-09-30" + ")\n",
          "## Median over the six basins\n", summary.to_markdown(), "\n",
          "## Per basin\n", markdown_table(table),
          "## Biggest observed flood per basin and what each model predicted on that day (mm/day)\n",
          peaks.to_markdown(index=False), "\n",
          "## Run details\n", pd.DataFrame(info).T.to_markdown(), "\n"]
    (METRIC_DIR / "comparison.md").write_text("\n".join(md))

    pd.set_option("display.width", 200)
    print(summary.to_string())
    print()
    print(table[["basin", "model", "nse", "kge", "r", "alpha", "beta", "pbias", "fhv"]].round(3).to_string(index=False))
    print()
    print(peaks.to_string(index=False))
    print(f"\n-> {METRIC_DIR / 'comparison.md'}")


if __name__ == "__main__":
    main()
