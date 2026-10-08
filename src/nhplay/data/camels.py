"""Load a few CAMELS-US basins into tidy pandas objects and say what each column *is*.

We deliberately re-use NeuralHydrology's own CAMELS-US readers so that what you inspect
here is byte-for-byte what the models are trained on. The only things added are:

* human-readable descriptions of every variable,
* an explicit tag for each column: DYNAMIC input, STATIC attribute or TARGET,
* unit conversion helpers (cfs -> mm/day) explained in words.

Folder layout expected under ``data/camels_us`` (this is NeuralHydrology's convention):

    data/camels_us/
    ├── basin_mean_forcing/daymet/<HUC2>/<gauge>_lump_cida_forcing_leap.txt
    ├── usgs_streamflow/<HUC2>/<gauge>_streamflow_qc.txt
    └── camels_attributes_v2.0/camels_{clim,geol,hydro,name,soil,topo,vege}.txt
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nhplay.paths import BASIN_LIST_FILE, CAMELS_DIR

# ----------------------------------------------------------------------------------------
# 1) What the columns mean
# ----------------------------------------------------------------------------------------
# Dynamic (time-varying) inputs: the Daymet weather, averaged over the whole basin, per day.
DYNAMIC_INPUTS = {
    "prcp(mm/day)": "Precipitation (rain + snow water equivalent), mm per day",
    "tmax(C)": "Daily maximum air temperature, deg C",
    "tmin(C)": "Daily minimum air temperature, deg C",
    "srad(W/m2)": "Incoming shortwave solar radiation (drives evaporation and snow melt)",
    "vp(Pa)": "Water vapour pressure (humidity)",
}
# Also in the file but we do NOT feed it to the model (it is a derived modelled quantity in Daymet
# and 'dayl' is a deterministic function of latitude and date): kept here so you know they exist.
EXTRA_FORCING_COLUMNS = {
    "dayl(s)": "Day length in seconds (deterministic from latitude + date)",
    "swe(mm)": "Snow water equivalent estimated by Daymet (a model output, not an observation)",
}

TARGET = {"QObs(mm/d)": "Observed daily streamflow at the USGS gauge, converted to mm/day over the basin"}

# Static attributes: one number per basin, constant in time. Names follow CAMELS v2.0.
STATIC_ATTRIBUTES = {
    "area_gages2": "Basin (drainage) area, km^2",
    "elev_mean": "Mean elevation, m above sea level",
    "slope_mean": "Mean terrain slope, m/km",
    "frac_forest": "Fraction of the basin covered by forest (0-1)",
    "lai_max": "Maximum leaf area index (how leafy the vegetation gets)",
    "soil_depth_pelletier": "Depth to bedrock, m",
    "soil_porosity": "Soil porosity (fraction of pore space)",
    "soil_conductivity": "Saturated hydraulic conductivity of the soil, cm/hr",
    "sand_frac": "Sand fraction of the soil, %",
    "clay_frac": "Clay fraction of the soil, %",
    "p_mean": "Long-term mean daily precipitation, mm/day (climate)",
    "pet_mean": "Long-term mean potential evapotranspiration, mm/day (climate)",
    "aridity": "Aridity index = PET / P (>1 means evaporation demand exceeds rainfall)",
    "frac_snow": "Fraction of precipitation falling as snow",
    "high_prec_freq": "Frequency of high-precipitation days (>= 5x mean), days/year",
    "low_prec_freq": "Frequency of dry days (< 1 mm), days/year",
    "high_prec_dur": "Average duration of high-precipitation events, days",
    "low_prec_dur": "Average duration of dry spells, days",
}

# Attribute columns that are nice to *display* but are never model inputs.
DISPLAY_ATTRIBUTES = {
    "gauge_name": "USGS gauge name (river and nearest town)",
    "gauge_lat": "Gauge latitude",
    "gauge_lon": "Gauge longitude",
    "q_mean": "Long-term mean observed streamflow, mm/day (this is derived from the TARGET, so it must never be a model input!)",
    "runoff_ratio": "q_mean / p_mean (also derived from the target)",
}


# ----------------------------------------------------------------------------------------
# 2) Loading
# ----------------------------------------------------------------------------------------
def read_basin_list(path: Path = BASIN_LIST_FILE) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip() and not line.startswith("#")]


def load_forcing(basin: str, data_dir: Path = CAMELS_DIR, forcing: str = "daymet") -> tuple[pd.DataFrame, int]:
    """Daily basin-mean weather for one basin. Returns (DataFrame indexed by date, area in m^2)."""
    from neuralhydrology.datasetzoo.camelsus import load_camels_us_forcings

    df, area_m2 = load_camels_us_forcings(data_dir=Path(data_dir), basin=basin, forcings=forcing)
    return df, int(area_m2)


def load_discharge_mm(basin: str, area_m2: int, data_dir: Path = CAMELS_DIR) -> pd.Series:
    """Observed streamflow in mm/day (NeuralHydrology's convention), NaN where USGS reported -999.

    The gauge measures volume per second (cubic feet per second). Dividing by the basin area
    turns it into a depth of water spread over the basin per day, which makes basins of very
    different sizes comparable and puts streamflow in the *same unit as rainfall*.
    """
    from neuralhydrology.datasetzoo.camelsus import load_camels_us_discharge

    q = load_camels_us_discharge(data_dir=Path(data_dir), basin=basin, area=area_m2)
    q = q.replace(-999.0, np.nan)
    q.name = "QObs(mm/d)"
    return q


def load_discharge_cfs(basin: str, data_dir: Path = CAMELS_DIR) -> pd.Series:
    """The raw USGS values (cubic feet per second) for people who want to see the original unit."""
    files = list((Path(data_dir) / "usgs_streamflow").glob(f"**/{basin}_streamflow_qc.txt"))
    if not files:
        raise FileNotFoundError(f"No streamflow file for basin {basin} under {data_dir}")
    cols = ["basin", "Year", "Mnth", "Day", "QObs(cfs)", "flag"]
    df = pd.read_csv(files[0], sep=r"\s+", header=None, names=cols)
    df.index = pd.to_datetime(df[["Year", "Mnth", "Day"]].rename(columns={"Mnth": "Month"}))
    s = df["QObs(cfs)"].replace(-999.0, np.nan)
    s.index.name = "date"
    return s


def load_attributes(basins: list[str] | None = None, data_dir: Path = CAMELS_DIR) -> pd.DataFrame:
    """All CAMELS v2.0 attributes (one row per basin). Index = gauge_id string."""
    from neuralhydrology.datasetzoo.camelsus import load_camels_us_attributes

    df = load_camels_us_attributes(data_dir=Path(data_dir), basins=basins or [])
    return df


@dataclass
class BasinData:
    basin: str
    area_m2: int
    timeseries: pd.DataFrame          # dynamic inputs + extra forcing + target, daily
    attributes: pd.Series             # all static attributes for this basin

    @property
    def area_km2(self) -> float:
        return self.area_m2 / 1e6

    def dynamic_inputs(self) -> pd.DataFrame:
        return self.timeseries[list(DYNAMIC_INPUTS)]

    def target(self) -> pd.Series:
        return self.timeseries["QObs(mm/d)"]

    def static_inputs(self) -> pd.Series:
        return self.attributes[list(STATIC_ATTRIBUTES)]

    def variable_table(self) -> pd.DataFrame:
        """A table that says, for every column, whether it is an input, a static or the target."""
        rows = []
        for name, desc in DYNAMIC_INPUTS.items():
            rows.append(("DYNAMIC input", name, desc))
        for name, desc in EXTRA_FORCING_COLUMNS.items():
            rows.append(("in file, not used", name, desc))
        for name, desc in STATIC_ATTRIBUTES.items():
            rows.append(("STATIC attribute", name, desc))
        for name, desc in TARGET.items():
            rows.append(("TARGET", name, desc))
        return pd.DataFrame(rows, columns=["role", "variable", "meaning"])


def load_basin(basin: str, data_dir: Path = CAMELS_DIR) -> BasinData:
    """Everything about one basin in one object."""
    forcing, area_m2 = load_forcing(basin, data_dir)
    q_mm = load_discharge_mm(basin, area_m2, data_dir)
    ts = forcing.join(q_mm, how="left")
    ts.index.name = "date"
    attrs = load_attributes([basin], data_dir).loc[basin]
    return BasinData(basin=basin, area_m2=area_m2, timeseries=ts, attributes=attrs)


def basin_summary(bd: BasinData) -> pd.Series:
    """Headline numbers for a basin, shown in the notebook and the app."""
    a = bd.attributes
    ts = bd.timeseries
    return pd.Series(
        {
            "gauge_id": bd.basin,
            "name": a.get("gauge_name", ""),
            "area_km2": round(bd.area_km2, 1),
            "mean_elevation_m": round(float(a["elev_mean"]), 0),
            "mean_slope_m_per_km": round(float(a["slope_mean"]), 1),
            "forest_fraction": round(float(a["frac_forest"]), 2),
            "aridity_PET_over_P": round(float(a["aridity"]), 2),
            "snow_fraction": round(float(a["frac_snow"]), 2),
            "first_day": ts.index.min().date(),
            "last_day": ts.index.max().date(),
            "n_days": int(len(ts)),
            "n_days_with_flow_obs": int(ts["QObs(mm/d)"].notna().sum()),
            "mean_rain_mm_per_day": round(float(ts["prcp(mm/day)"].mean()), 2),
            "mean_flow_mm_per_day": round(float(ts["QObs(mm/d)"].mean()), 2),
            "max_flow_mm_per_day": round(float(ts["QObs(mm/d)"].max()), 1),
            "date_of_max_flow": ts["QObs(mm/d)"].idxmax().date(),
        }
    )
