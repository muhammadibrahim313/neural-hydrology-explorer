"""Assemble the LSTM-vs-Transformer comparison table from NeuralHydrology run directories."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nhplay.evaluation import metrics as M
from nhplay.models.run import load_results


def score_run(run_dir: Path, model_name: str, period: str = "test", epoch: int | None = None) -> pd.DataFrame:
    """One row per basin with our readable metrics for a single run."""
    rows = []
    for basin, df in load_results(run_dir, period, epoch).items():
        sc = M.scorecard(df["obs"].values, df["sim"].values).as_dict()
        sc.update({"basin": basin, "model": model_name, "period": period,
                   "n_days": int(df.dropna().shape[0]), "run_dir": str(run_dir)})
        rows.append(sc)
    cols = ["basin", "model", "period", "nse", "kge", "r", "alpha", "beta", "rmse", "mae", "pbias", "fhv",
            "n_days", "run_dir"]
    return pd.DataFrame(rows)[cols]


def comparison_table(runs: dict[str, Path], period: str = "test") -> pd.DataFrame:
    """runs = {'lstm': Path(...), 'transformer': Path(...)} -> long table (basin x model)."""
    return pd.concat([score_run(p, name, period) for name, p in runs.items()], ignore_index=True)


def summarise(table: pd.DataFrame) -> pd.DataFrame:
    """Median across basins per model (the usual way hydrology papers summarise many basins)."""
    g = table.groupby("model")[["nse", "kge", "r", "alpha", "beta", "pbias", "fhv"]]
    out = g.median().round(3)
    out.insert(0, "n_basins", g.size())
    return out


def markdown_table(table: pd.DataFrame) -> str:
    """Markdown for the README: one row per basin per model."""
    cols = ["basin", "model", "nse", "kge", "r", "alpha", "beta", "pbias", "fhv"]
    t = table[cols].copy()
    for c in cols[2:]:
        t[c] = t[c].map(lambda v: f"{v:.3f}" if np.isfinite(v) else "nan")
    header = "| " + " | ".join(cols) + " |\n|" + "|".join(["---"] * len(cols)) + "|\n"
    body = "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in t.itertuples(index=False))
    return header + body + "\n"
