"""Matplotlib figures used by the scripts, the notebook and (as static fallbacks) the app.

Style rules kept deliberately simple and consistent:
* Rain is drawn as bars in its *own* panel above the flow panel (no dual y-axes).
* Observed flow is always dark grey/black; LSTM is blue; Transformer is orange.
* Streamflow in mm/day so that rain and flow are in the same unit and comparable by eye.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: scripts save PNGs; the notebook re-enables inline display itself
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fixed colours by *entity* (validated for colour-vision deficiency with the dataviz palette).
COLORS = {
    "observed": "#2b2b2b",
    "lstm": "#2a78d6",         # blue
    "transformer": "#eb6834",  # orange
    "rain": "#1baf7a",         # aqua-green for precipitation bars
    "temperature": "#eda100",  # yellow/amber
    "muted": "#9a9a9a",
}
MODEL_LABELS = {"lstm": "LSTM", "transformer": "Transformer", "observed": "Observed"}

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.titleweight": "bold",
        "font.size": 10,
    }
)


def _finish(fig, savepath: Path | None):
    fig.tight_layout()
    if savepath is not None:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, bbox_inches="tight")
    return fig


def plot_forcing_and_flow(ts: pd.DataFrame, title: str, start=None, end=None, savepath: Path | None = None):
    """Three stacked panels sharing the time axis: rain, temperature, observed streamflow."""
    df = ts.loc[start:end] if (start or end) else ts
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1.6]})

    ax = axes[0]
    if len(df) > 3 * 366:
        # thousands of daily bars would alias into a smear: show monthly totals for long windows
        monthly = df["prcp(mm/day)"].resample("MS").sum()
        ax.bar(monthly.index, monthly.values, width=28, align="edge", color=COLORS["rain"], linewidth=0)
        ax.set_ylabel("Rain\n(mm/month)")
    else:
        ax.bar(df.index, df["prcp(mm/day)"], width=1.0, color=COLORS["rain"], linewidth=0)
        ax.set_ylabel("Rain\n(mm/day)")
    ax.set_title(title)
    ax.invert_yaxis()  # hydrology convention: rain "falls" from the top of the panel

    ax = axes[1]
    ax.plot(df.index, df["tmax(C)"], color=COLORS["temperature"], lw=1.2, label="Tmax")
    ax.plot(df.index, df["tmin(C)"], color=COLORS["lstm"], lw=1.2, label="Tmin")
    ax.axhline(0, color=COLORS["muted"], lw=0.8, ls="--")
    ax.set_ylabel("Temperature\n(°C)")
    ax.legend(loc="upper right", frameon=False, ncol=2)

    ax = axes[2]
    ax.plot(df.index, df["QObs(mm/d)"], color=COLORS["observed"], lw=1.3)
    ax.set_ylabel("Observed streamflow\n(mm/day)")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    return _finish(fig, savepath)


def plot_hydrograph(obs: pd.Series, sims: dict[str, pd.Series], title: str, rain: pd.Series | None = None,
                    start=None, end=None, savepath: Path | None = None):
    """Observed vs predicted streamflow (a *hydrograph*), optionally with the rain panel on top."""
    sl = slice(start, end)
    obs = obs.loc[sl]
    n_panels = 2 if rain is not None else 1
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(12, 5.5 if rain is None else 7), sharex=True,
        gridspec_kw={"height_ratios": [1, 2.6]} if rain is not None else None,
    )
    axes = np.atleast_1d(axes)
    if rain is not None:
        r = rain.loc[sl]
        if len(r) > 3 * 366:
            r = r.resample("MS").sum()
            axes[0].bar(r.index, r.values, width=28, align="edge", color=COLORS["rain"], linewidth=0)
            axes[0].set_ylabel("Rain (mm/month)")
        else:
            axes[0].bar(r.index, r.values, width=1.0, color=COLORS["rain"], linewidth=0)
            axes[0].set_ylabel("Rain (mm/day)")
        axes[0].invert_yaxis()
        axes[0].set_title(title)
    else:
        axes[0].set_title(title)
    ax = axes[-1]
    ax.plot(obs.index, obs, color=COLORS["observed"], lw=1.6, label="Observed (USGS gauge)")
    for key, sim in sims.items():
        ax.plot(sim.loc[sl].index, sim.loc[sl], color=COLORS.get(key, None), lw=1.2, alpha=0.95,
                label=f"{MODEL_LABELS.get(key, key)} prediction")
    ax.set_ylabel("Streamflow (mm/day)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper right", frameon=False)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    return _finish(fig, savepath)


def plot_scatter(obs: pd.Series, sims: dict[str, pd.Series], title: str, savepath: Path | None = None):
    """Observed (x) vs predicted (y). A perfect model would put every day on the 1:1 line."""
    n = len(sims)
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 4.6), squeeze=False)
    for ax, (key, sim) in zip(axes[0], sims.items()):
        df = pd.concat([obs.rename("obs"), sim.rename("sim")], axis=1).dropna()
        hi = float(max(df.max()))
        ax.scatter(df["obs"], df["sim"], s=7, alpha=0.35, color=COLORS.get(key, None), linewidths=0)
        ax.plot([0, hi], [0, hi], color=COLORS["muted"], lw=1, ls="--", label="1:1 line")
        ax.set_xlabel("Observed (mm/day)")
        ax.set_ylabel("Predicted (mm/day)")
        ax.set_title(f"{MODEL_LABELS.get(key, key)}")
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.legend(frameon=False, loc="upper left")
    fig.suptitle(title)
    return _finish(fig, savepath)


def plot_flow_duration_curve(obs: pd.Series, sims: dict[str, pd.Series], title: str, savepath: Path | None = None):
    """Flow duration curve: how often each flow level is exceeded. Reveals bias in high/low flows."""
    fig, ax = plt.subplots(figsize=(7, 4.6))

    def fdc(s):
        v = np.sort(s.dropna().values)[::-1]
        p = 100 * (np.arange(1, len(v) + 1)) / (len(v) + 1)
        return p, v

    p, v = fdc(obs)
    ax.plot(p, v, color=COLORS["observed"], lw=1.8, label="Observed")
    for key, sim in sims.items():
        p, v = fdc(sim.reindex(obs.dropna().index))
        ax.plot(p, v, color=COLORS.get(key, None), lw=1.3, label=MODEL_LABELS.get(key, key))
    ax.set_yscale("log")
    ax.set_xlabel("% of days this flow is exceeded")
    ax.set_ylabel("Streamflow (mm/day, log scale)")
    ax.set_title(title)
    ax.legend(frameon=False)
    return _finish(fig, savepath)


def plot_metric_bars(metrics: pd.DataFrame, metric: str, title: str, savepath: Path | None = None):
    """Grouped bars: one group per basin, one bar per model."""
    pivot = metrics.pivot(index="basin", columns="model", values=metric)
    models = [m for m in ["lstm", "transformer"] if m in pivot.columns] + [
        m for m in pivot.columns if m not in ("lstm", "transformer")]
    x = np.arange(len(pivot.index))
    w = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(pivot.index)), 4.2))
    for i, m in enumerate(models):
        vals = pivot[m].values
        bars = ax.bar(x + (i - (len(models) - 1) / 2) * w, vals, width=w * 0.92, color=COLORS.get(m, None),
                      label=MODEL_LABELS.get(m, m), linewidth=0)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=0)
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.axhline(0, color=COLORS["muted"], lw=0.8)
    ax.legend(frameon=False)
    return _finish(fig, savepath)
