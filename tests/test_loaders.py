"""The pure-pandas readers must agree with NeuralHydrology's, and missing flow (-999) must become NaN."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nhplay.data import camels as C

ROOT = Path(__file__).resolve().parents[1]
HAVE_DATA = (ROOT / "data" / "camels_us" / "camels_attributes_v2.0" / "camels_topo.txt").exists()


@pytest.mark.skipif(not (HAVE_DATA and C._nh_available()), reason="needs downloaded data and neuralhydrology")
def test_pandas_readers_match_neuralhydrology():
    basin = "01022500"
    df_nh, area_nh = C.load_forcing(basin)                      # NeuralHydrology path
    df_pd, area_pd = C._read_forcing_pandas(C._find_forcing_file(basin, C.CAMELS_DIR, "daymet"))
    assert area_nh == area_pd
    cols = list(C.DYNAMIC_INPUTS) + list(C.EXTRA_FORCING_COLUMNS)
    pd.testing.assert_frame_equal(df_nh[cols], df_pd[cols])
    q_nh = C.load_discharge_mm(basin, area_nh)
    q_pd = C._read_discharge_pandas(C._find_discharge_file(basin, C.CAMELS_DIR), area_pd)
    np.testing.assert_allclose(q_nh.values, q_pd.values, rtol=1e-12)
    a_nh = C.load_attributes([basin]).loc[basin, list(C.STATIC_ATTRIBUTES)]
    a_pd = C._read_attributes_pandas(C.CAMELS_DIR, [basin]).loc[basin, list(C.STATIC_ATTRIBUTES)]
    np.testing.assert_allclose(a_nh.astype(float).values, a_pd.astype(float).values)


def test_missing_flow_becomes_nan(tmp_path):
    basin = "99999999"
    (tmp_path / "usgs_streamflow" / "01").mkdir(parents=True)
    (tmp_path / "basin_mean_forcing" / "daymet" / "01").mkdir(parents=True)
    flow = tmp_path / "usgs_streamflow" / "01" / f"{basin}_streamflow_qc.txt"
    flow.write_text(f"{basin} 2000 01 01 100.00 A\n{basin} 2000 01 02 -999.00 M\n{basin} 2000 01 03 50.00 A\n")
    forcing = tmp_path / "basin_mean_forcing" / "daymet" / "01" / f"{basin}_lump_cida_forcing_leap.txt"
    forcing.write_text("45.0\n100.0\n1000000000\nYear Mnth Day Hr dayl(s) prcp(mm/day) srad(W/m2) swe(mm) tmax(C) tmin(C) vp(Pa)\n"
                       "2000 01 01 12 30000 1.0 100 0 1.0 -1.0 500\n2000 01 02 12 30000 2.0 100 0 1.0 -1.0 500\n"
                       "2000 01 03 12 30000 0.0 100 0 1.0 -1.0 500\n")
    area = 1_000_000_000  # 1000 km^2
    q_pd = C._read_discharge_pandas(flow, area)
    assert np.isnan(q_pd.iloc[1]) and q_pd.iloc[0] > 0
    # 100 cfs over 1000 km^2 = 100 * 0.0283168 * 86400 / 1e9 * 1000 mm = 0.2447 mm/day
    assert q_pd.iloc[0] == pytest.approx(0.24467, rel=1e-3)
    q_all = C.load_discharge_mm(basin, area, data_dir=tmp_path)  # whichever path is active
    assert np.isnan(q_all.iloc[1]) and q_all.iloc[0] == pytest.approx(q_pd.iloc[0])
    df, a = C._read_forcing_pandas(forcing)
    assert a == area and list(df["prcp(mm/day)"]) == [1.0, 2.0, 0.0]
