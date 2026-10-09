import numpy as np
import pandas as pd

from nhplay.evaluation import events as E


def test_water_year():
    idx = pd.DatetimeIndex(["2010-09-30", "2010-10-01", "2011-09-30", "2011-10-01"])
    assert list(E.water_year(idx)) == [2010, 2011, 2011, 2012]


def test_return_levels_are_ordered_and_bounded():
    rng = np.random.default_rng(1)
    maxima = pd.Series(rng.gamma(3.0, 5.0, size=24))
    rl = E.empirical_return_levels(maxima, (2, 5, 10, 50))
    assert rl[2] < rl[5] < rl[10]
    assert np.isnan(rl[50])            # 24 years cannot support a 50-year estimate
    assert rl[10] <= maxima.max()


def test_episodes_and_exceedance():
    idx = pd.date_range("2000-01-01", periods=10, freq="D")
    obs = pd.Series([0, 5, 6, 0, 0, 7, 0, 0, 0, 8.0], index=idx)
    sim = pd.Series([0, 5, 0, 0, 0, 7, 7, 0, 0, 0.0], index=idx)
    assert E.n_episodes(obs > 4) == 3
    tbl = E.exceedance_table(obs, {"lstm": sim}, {"L": 4.0}).iloc[0]
    assert tbl["observed_days_above"] == 4 and tbl["lstm_hits"] == 2
    assert tbl["lstm_missed"] == 2 and tbl["lstm_false_alarms"] == 1


def test_monthly_regime_order():
    idx = pd.date_range("2004-10-01", "2006-09-30", freq="D")
    obs = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    reg = E.monthly_regime(obs, {})
    assert list(reg.index) == ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
