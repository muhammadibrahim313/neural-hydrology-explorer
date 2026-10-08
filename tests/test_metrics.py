"""Our readable metrics must agree with NeuralHydrology's reference implementation."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from nhplay.evaluation import metrics as M


@pytest.fixture
def series():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2000-01-01", periods=1000, freq="D")
    obs = np.abs(rng.gamma(2.0, 1.0, size=1000))
    sim = obs * rng.normal(1.0, 0.3, size=1000) + 0.1
    sim[::50] = np.nan  # some missing predictions
    obs[::77] = np.nan  # some missing observations
    return dates, obs, sim


def test_nse_kge_match_neuralhydrology(series):
    from neuralhydrology.evaluation import metrics as NH

    dates, obs, sim = series
    o = xr.DataArray(obs, coords={"date": dates}, dims=["date"])
    s = xr.DataArray(sim, coords={"date": dates}, dims=["date"])
    assert M.nse(obs, sim) == pytest.approx(NH.nse(o, s), abs=1e-9)
    assert M.kge(obs, sim) == pytest.approx(NH.kge(o, s), abs=1e-9)
    assert M.fhv(obs, sim) == pytest.approx(NH.fdc_fhv(o, s), abs=1e-6)


def test_perfect_and_mean_predictions():
    obs = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    assert M.nse(obs, obs) == pytest.approx(1.0)
    assert M.kge(obs, obs) == pytest.approx(1.0)
    mean_pred = np.full_like(obs, obs.mean())
    assert M.nse(obs, mean_pred) == pytest.approx(0.0)
    # constant prediction has zero std, so r is undefined -> KGE is nan (same as reference)
    assert np.isnan(M.kge(obs, mean_pred))


def test_interpret_text():
    assert "very good" in M.interpret(0.85, 0.8)
    assert "worse than" in M.interpret(-0.2, -0.5)
