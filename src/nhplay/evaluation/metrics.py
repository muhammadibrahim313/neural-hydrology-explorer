"""Hydrology skill scores written out line by line, for learning.

NeuralHydrology ships its own (well tested) implementations in
``neuralhydrology.evaluation.metrics``. We use *those* for the official numbers in
``outputs/metrics`` and keep this file as the readable, dependency-free version so that
you can follow exactly what each number means. ``scripts/evaluate.py`` checks that both
implementations agree.

Conventions
-----------
* ``obs``: observed streamflow (what the river gauge measured), 1-D array, mm/day.
* ``sim``: simulated / predicted streamflow from the model, same length, mm/day.
* Days where either value is missing (NaN) are dropped from both series before scoring.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _clean(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    if obs.shape != sim.shape:
        raise ValueError(f"obs and sim must have the same shape, got {obs.shape} vs {sim.shape}")
    keep = ~np.isnan(obs) & ~np.isnan(sim)
    return obs[keep], sim[keep]


def nse(obs, sim) -> float:
    """Nash-Sutcliffe Efficiency.

    NSE = 1 - sum((sim - obs)^2) / sum((obs - mean(obs))^2)

    Read it as: "how much better is the model than simply predicting the long-term
    average flow every day?"

    * NSE = 1   perfect match
    * NSE = 0   as good as predicting the mean flow (useless model)
    * NSE < 0   worse than predicting the mean
    * NSE > 0.7 is usually considered "good" for daily streamflow in the literature.

    Because the squared errors are dominated by big numbers, NSE mostly rewards getting
    the *high flows* (floods) right and barely notices errors in low flows.
    """
    obs, sim = _clean(obs, sim)
    denominator = np.sum((obs - obs.mean()) ** 2)
    numerator = np.sum((sim - obs) ** 2)
    return float(1.0 - numerator / denominator)


def kge(obs, sim) -> float:
    """Kling-Gupta Efficiency (Gupta et al., 2009).

    KGE = 1 - sqrt( (r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2 )

    with three ingredients that each have a plain meaning:

    * r      Pearson correlation between obs and sim        -> timing / shape of the hydrograph
    * alpha  std(sim) / std(obs)                            -> variability (are peaks as spiky as observed?)
    * beta   mean(sim) / mean(obs)                          -> bias (too much / too little water overall?)

    KGE = 1 is perfect. KGE > -0.41 means the model beats the "mean flow" benchmark
    (Knoben et al., 2019), so KGE and NSE are *not* on the same scale.
    """
    obs, sim = _clean(obs, sim)
    if len(obs) < 2:
        return float("nan")
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = sim.std(ddof=1) / obs.std(ddof=1)
    beta = sim.mean() / obs.mean()
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def kge_components(obs, sim) -> dict:
    """The three parts of KGE, useful for diagnosing *why* a model scores badly."""
    obs, sim = _clean(obs, sim)
    return {
        "r": float(np.corrcoef(obs, sim)[0, 1]),
        "alpha": float(sim.std(ddof=1) / obs.std(ddof=1)),
        "beta": float(sim.mean() / obs.mean()),
    }


def rmse(obs, sim) -> float:
    """Root mean squared error, in the units of the data (mm/day)."""
    obs, sim = _clean(obs, sim)
    return float(np.sqrt(np.mean((sim - obs) ** 2)))


def mae(obs, sim) -> float:
    """Mean absolute error, in mm/day. Less dominated by floods than RMSE."""
    obs, sim = _clean(obs, sim)
    return float(np.mean(np.abs(sim - obs)))


def pbias(obs, sim) -> float:
    """Percent bias: positive = model produces too much water overall."""
    obs, sim = _clean(obs, sim)
    return float(100.0 * (sim.sum() - obs.sum()) / obs.sum())


def fhv(obs, sim, top_fraction: float = 0.02) -> float:
    """Peak-flow bias (FHV, Yilmaz et al. 2008): % bias over the highest 2 % of flows.

    This is the metric that matters most for floods. Negative = peaks are underestimated.
    """
    obs, sim = _clean(obs, sim)
    obs_sorted = np.sort(obs)[::-1]
    sim_sorted = np.sort(sim)[::-1]
    n = max(1, int(round(top_fraction * len(obs))))
    return float(100.0 * (sim_sorted[:n].sum() - obs_sorted[:n].sum()) / obs_sorted[:n].sum())


@dataclass
class Scorecard:
    nse: float
    kge: float
    r: float
    alpha: float
    beta: float
    rmse: float
    mae: float
    pbias: float
    fhv: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def scorecard(obs, sim) -> Scorecard:
    comps = kge_components(obs, sim)
    return Scorecard(
        nse=nse(obs, sim),
        kge=kge(obs, sim),
        r=comps["r"],
        alpha=comps["alpha"],
        beta=comps["beta"],
        rmse=rmse(obs, sim),
        mae=mae(obs, sim),
        pbias=pbias(obs, sim),
        fhv=fhv(obs, sim),
    )


def interpret(nse_value: float, kge_value: float) -> str:
    """One-sentence plain-English reading of a (NSE, KGE) pair, for the app and notes."""
    if np.isnan(nse_value):
        return "No overlapping observed/predicted days to score."
    if nse_value >= 0.8:
        quality = "very good"
    elif nse_value >= 0.7:
        quality = "good"
    elif nse_value >= 0.5:
        quality = "acceptable"
    elif nse_value >= 0.0:
        quality = "weak (only slightly better than predicting the average flow)"
    else:
        quality = "poor (worse than predicting the average flow every day)"
    msg = f"NSE = {nse_value:.2f}: {quality}."
    if kge_value is not None and not np.isnan(kge_value):
        msg += f" KGE = {kge_value:.2f}"
        msg += " (beats the mean-flow benchmark)." if kge_value > -0.41 else " (does not beat the mean-flow benchmark)."
    return msg
