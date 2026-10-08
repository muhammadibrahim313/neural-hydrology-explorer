#!/usr/bin/env python
"""Phase 1 - look at the real data before touching any model.

For every basin in data/basins.txt this script prints and saves:
  * a table that labels every variable as DYNAMIC input / STATIC attribute / TARGET,
  * the static attributes (area, elevation, slope, forest fraction, soils, climate),
  * a few example days (date, rain, temperature, observed streamflow),
  * figures: one water year of rain + temperature + streamflow, and the whole record.

Outputs go to outputs/figures/data_*.png and outputs/metrics/basin_summary.csv.

Usage:  python scripts/inspect_data.py [--year 2010] [basin ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nhplay.data.camels import (DISPLAY_ATTRIBUTES, STATIC_ATTRIBUTES, basin_summary, load_basin,  # noqa: E402
                                read_basin_list)
from nhplay.paths import FIGURE_DIR, METRIC_DIR, ensure_dirs  # noqa: E402
from nhplay.visualization.plots import plot_forcing_and_flow  # noqa: E402

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("basins", nargs="*")
    ap.add_argument("--year", type=int, default=2010, help="water year to zoom into (Oct of year-1 .. Sep of year)")
    args = ap.parse_args()
    ensure_dirs()
    basins = args.basins or read_basin_list()

    summaries, static_rows = [], []
    for i, basin in enumerate(basins):
        bd = load_basin(basin)
        ts = bd.timeseries
        print("=" * 100)
        print(f"BASIN {basin}  -  {bd.attributes.get('gauge_name', '')}")
        print("=" * 100)

        if i == 0:
            print("\nWhat each column is (same for every basin):")
            print(bd.variable_table().to_string(index=False))

        print("\nSTATIC attributes (one number per basin, constant in time):")
        statics = bd.static_inputs().rename("value").to_frame()
        statics["meaning"] = [STATIC_ATTRIBUTES[k] for k in statics.index]
        print(statics.to_string())
        statics_row = bd.static_inputs().copy()
        statics_row["gauge_name"] = bd.attributes.get("gauge_name", "")
        statics_row.name = basin
        static_rows.append(statics_row)

        print("\nDYNAMIC inputs + TARGET, a few example days (water year %d):" % args.year)
        start, end = f"{args.year - 1}-10-01", f"{args.year}-09-30"
        window = ts.loc[start:end]
        wettest = window["prcp(mm/day)"].idxmax()
        example_days = window.loc[wettest - pd.Timedelta(days=2): wettest + pd.Timedelta(days=5),
                                  ["prcp(mm/day)", "tmax(C)", "tmin(C)", "srad(W/m2)", "vp(Pa)", "QObs(mm/d)"]]
        print(f"(around the wettest day of that year, {wettest.date()})")
        print(example_days.round(2).to_string())

        summ = basin_summary(bd)
        summaries.append(summ)
        print("\nSummary:")
        print(summ.to_string())

        plot_forcing_and_flow(ts, f"{basin} {bd.attributes.get('gauge_name', '')} - water year {args.year}",
                              start=start, end=end, savepath=FIGURE_DIR / f"data_{basin}_wy{args.year}.png")
        plot_forcing_and_flow(ts, f"{basin} {bd.attributes.get('gauge_name', '')} - full record 1980-2014",
                              savepath=FIGURE_DIR / f"data_{basin}_full.png")
        print(f"\nfigures -> {FIGURE_DIR / f'data_{basin}_wy{args.year}.png'}, {FIGURE_DIR / f'data_{basin}_full.png'}")

    summary_df = pd.DataFrame(summaries).set_index("gauge_id")
    summary_df.to_csv(METRIC_DIR / "basin_summary.csv")
    pd.DataFrame(static_rows).to_csv(METRIC_DIR / "basin_static_attributes.csv", index_label="gauge_id")
    print("\n" + "=" * 100)
    print("ALL BASINS")
    print(summary_df[["name", "area_km2", "mean_elevation_m", "forest_fraction", "aridity_PET_over_P",
                      "snow_fraction", "mean_rain_mm_per_day", "mean_flow_mm_per_day", "max_flow_mm_per_day"]].to_string())
    print(f"\nsaved {METRIC_DIR / 'basin_summary.csv'} and {METRIC_DIR / 'basin_static_attributes.csv'}")


if __name__ == "__main__":
    main()
