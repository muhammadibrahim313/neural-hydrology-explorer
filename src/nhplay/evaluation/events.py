"""Flood-oriented views of a streamflow record: water years, annual maxima, return levels, exceedances.

Everything here is deliberately simple and *empirical* (no distribution fitting) so that it can be read and
checked by hand. It is meant for teaching, not for setting real flood warning levels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nhplay.evaluation import metrics as M

WY_MONTH_ORDER = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
MONTH_ABBR = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep",
              10: "Oct", 11: "Nov", 12: "Dec"}


def water_year(index: pd.DatetimeIndex) -> np.ndarray:
    """US water year: 1 Oct 2010 .. 30 Sep 2011 is water year 2011."""
    index = pd.DatetimeIndex(index)
    return np.where(index.month >= 10, index.year + 1, index.year)


def annual_maxima(flow: pd.Series, start=None, end=None) -> pd.DataFrame:
    """Largest daily flow in each water year (only complete-ish years with >= 300 days of data)."""
    s = flow.loc[start:end].dropna()
    wy = pd.Series(water_year(s.index), index=s.index, name="water_year")
    rows = []
    for year, grp in s.groupby(wy):
        if len(grp) < 300:
            continue
        rows.append({"water_year": int(year), "peak_date": grp.idxmax().date(), "peak_mm_per_day": float(grp.max())})
    return pd.DataFrame(rows).set_index("water_year")


def empirical_return_levels(maxima: pd.Series, return_periods=(2, 5, 10)) -> dict[int, float]:
    """Return levels from annual maxima with Weibull plotting positions, interpolated in log(T).

    With n annual maxima the largest one has an empirical return period of (n + 1) years, so nothing beyond
    that can be estimated, and anything near it is very uncertain. Returns NaN for those.
    """
    x = np.sort(np.asarray(maxima.dropna(), dtype=float))[::-1]  # largest first
    n = len(x)
    if n < 5:
        return {rp: float("nan") for rp in return_periods}
    T = (n + 1) / np.arange(1, n + 1)  # return period of the i-th largest value
    log_t = np.log(T[::-1])  # increasing
    levels = x[::-1]
    out = {}
    for rp in return_periods:
        out[int(rp)] = float(np.interp(np.log(rp), log_t, levels)) if rp <= T.max() else float("nan")
    return out


def n_episodes(mask: pd.Series) -> int:
    """Number of separate runs of consecutive True days."""
    mask = mask.fillna(False).astype(bool)
    return int((mask & ~mask.shift(1, fill_value=False)).sum())


def exceedance_table(obs: pd.Series, sims: dict[str, pd.Series], levels: dict[str, float]) -> pd.DataFrame:
    """Per threshold: how often observation and each model exceed it, and how often they agree."""
    rows = []
    for name, thr in levels.items():
        if thr is None or np.isnan(thr):
            continue
        o = obs > thr
        row = {"level": name, "threshold_mm_per_day": round(thr, 2), "observed_days_above": int(o.sum()),
               "observed_episodes": n_episodes(o)}
        for m, s in sims.items():
            s = s.reindex(obs.index)
            p = s > thr
            row[f"{m}_days_above"] = int(p.sum())
            row[f"{m}_hits"] = int((p & o).sum())
            row[f"{m}_missed"] = int((o & ~p).sum())
            row[f"{m}_false_alarms"] = int((p & ~o).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def metrics_by_water_year(obs: pd.Series, sims: dict[str, pd.Series]) -> pd.DataFrame:
    """NSE and KGE for every water year of the period, per model, plus the observed peak."""
    wy = pd.Series(water_year(obs.index), index=obs.index)
    rows = []
    for year, idx in obs.groupby(wy).groups.items():
        o = obs.loc[idx]
        if o.notna().sum() < 300:
            continue
        row = {"water_year": int(year), "observed_peak_mm_per_day": float(o.max()), "peak_date": o.idxmax().date(),
               "observed_mean_mm_per_day": float(o.mean())}
        for m, s in sims.items():
            s = s.reindex(o.index)
            row[f"{m}_nse"] = M.nse(o.values, s.values)
            row[f"{m}_kge"] = M.kge(o.values, s.values)
            row[f"{m}_peak_mm_per_day"] = float(s.loc[o.idxmax()])
        rows.append(row)
    return pd.DataFrame(rows).set_index("water_year")


def monthly_regime(obs: pd.Series, sims: dict[str, pd.Series]) -> pd.DataFrame:
    """Mean flow per calendar month (water-year order Oct..Sep): does the model get the seasons right?"""
    df = pd.DataFrame({"observed": obs})
    for m, s in sims.items():
        df[m] = s.reindex(obs.index)
    out = df.groupby(df.index.month).mean()
    out = out.loc[WY_MONTH_ORDER]
    out.index = [MONTH_ABBR[m] for m in out.index]
    out.index.name = "month"
    return out


def mm_per_day_to_m3s(q_mm_day: float, area_m2: float) -> float:
    """Depth over the basin per day -> volume per second at the gauge."""
    return q_mm_day / 1000.0 * area_m2 / 86400.0
