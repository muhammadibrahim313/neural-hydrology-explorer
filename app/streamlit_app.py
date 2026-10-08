"""Neural Hydrology Explorer - a small Streamlit interface over the project's data and model outputs.

Run from the project root:   streamlit run app/streamlit_app.py

Everything shown here is produced by the scripts (download_data.py, train.py, evaluate.py, compare.py);
the app only reads data/ and outputs/. It never trains anything.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nhplay.data.camels import (DYNAMIC_INPUTS, STATIC_ATTRIBUTES, TARGET, basin_summary,  # noqa: E402
                                load_basin, read_basin_list)
from nhplay.evaluation import metrics as M  # noqa: E402
from nhplay.paths import CAMELS_DIR, METRIC_DIR, PREDICTION_DIR  # noqa: E402

COLORS = {"observed": "#2b2b2b", "lstm": "#2a78d6", "transformer": "#eb6834", "rain": "#1baf7a",
          "tmax": "#eda100", "tmin": "#2a78d6", "threshold": "#d03b3b"}
LABELS = {"lstm": "LSTM", "transformer": "Transformer"}
TEST_START, TEST_END = "2004-10-01", "2014-09-30"

st.set_page_config(page_title="Neural Hydrology Explorer", page_icon="🌊", layout="wide")


# ------------------------------------------------------------------------------------------ data
@st.cache_data(show_spinner=False)
def get_basins() -> list[str]:
    return read_basin_list()


@st.cache_data(show_spinner="Loading basin data ...")
def get_basin(basin: str):
    bd = load_basin(basin)
    return bd.timeseries, bd.attributes, bd.area_m2, basin_summary(bd)


@st.cache_data(show_spinner=False)
def get_predictions(model: str, period: str = "test") -> pd.DataFrame | None:
    f = PREDICTION_DIR / f"{model}_{period}.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, dtype={"basin": str}, parse_dates=["date"])
    return df


@st.cache_data(show_spinner=False)
def get_metric_table() -> pd.DataFrame | None:
    f = METRIC_DIR / "comparison_test.csv"
    if f.exists():
        return pd.read_csv(f, dtype={"basin": str})
    frames = []
    for m in ("lstm", "transformer"):
        g = METRIC_DIR / f"{m}_test_metrics.csv"
        if g.exists():
            frames.append(pd.read_csv(g, dtype={"basin": str}))
    return pd.concat(frames, ignore_index=True) if frames else None


def water_year_bounds(wy: int) -> tuple[str, str]:
    return f"{wy - 1}-10-01", f"{wy}-09-30"


# ------------------------------------------------------------------------------------------ plots
def rain_flow_figure(ts: pd.DataFrame, sims: dict[str, pd.Series] | None, title: str, threshold: float | None = None,
                     show_temp: bool = False) -> go.Figure:
    rows = 3 if show_temp else 2
    heights = [0.22, 0.2, 0.58] if show_temp else [0.25, 0.75]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=heights)
    fig.add_trace(go.Bar(x=ts.index, y=ts["prcp(mm/day)"], name="Rain (mm/day)", marker_color=COLORS["rain"],
                         hovertemplate="%{x|%d %b %Y}<br>rain %{y:.1f} mm<extra></extra>"), row=1, col=1)
    fig.update_yaxes(title_text="rain mm/day", autorange="reversed", row=1, col=1)
    r = 2
    if show_temp:
        fig.add_trace(go.Scatter(x=ts.index, y=ts["tmax(C)"], name="Tmax (°C)", line=dict(color=COLORS["tmax"], width=1.2),
                                 hovertemplate="%{x|%d %b %Y}<br>Tmax %{y:.1f} °C<extra></extra>"), row=2, col=1)
        fig.add_trace(go.Scatter(x=ts.index, y=ts["tmin(C)"], name="Tmin (°C)", line=dict(color=COLORS["tmin"], width=1.2),
                                 hovertemplate="%{x|%d %b %Y}<br>Tmin %{y:.1f} °C<extra></extra>"), row=2, col=1)
        fig.update_yaxes(title_text="°C", row=2, col=1)
        r = 3
    fig.add_trace(go.Scatter(x=ts.index, y=ts["QObs(mm/d)"], name="Observed flow", line=dict(color=COLORS["observed"], width=1.8),
                             hovertemplate="%{x|%d %b %Y}<br>observed %{y:.2f} mm/day<extra></extra>"), row=r, col=1)
    if sims:
        for key, s in sims.items():
            s = s.reindex(ts.index)
            fig.add_trace(go.Scatter(x=s.index, y=s.values, name=f"{LABELS.get(key, key)} prediction",
                                     line=dict(color=COLORS.get(key, None), width=1.4),
                                     hovertemplate="%{x|%d %b %Y}<br>" + LABELS.get(key, key) + " %{y:.2f} mm/day<extra></extra>"),
                          row=r, col=1)
    if threshold is not None:
        fig.add_hline(y=threshold, line=dict(color=COLORS["threshold"], dash="dash", width=1.2),
                      annotation_text=f"demo threshold {threshold:.1f} mm/day", annotation_position="top left", row=r, col=1)
    fig.update_yaxes(title_text="streamflow mm/day", row=r, col=1)
    fig.update_layout(title=title, height=560 if show_temp else 480, hovermode="x unified", bargap=0,
                      legend=dict(orientation="h", y=1.08, x=0), margin=dict(l=10, r=10, t=70, b=10))
    return fig


# ------------------------------------------------------------------------------------------ UI
st.title("🌊 Neural Hydrology Explorer")
st.caption("weather (rain, temperature, radiation, humidity) + static basin attributes → LSTM / Transformer → "
           "daily streamflow → observed vs predicted hydrograph → NSE / KGE → (educational) flood threshold")

if not CAMELS_DIR.exists() or not (CAMELS_DIR / "camels_attributes_v2.0").exists():
    st.error("No data found. Run `python scripts/download_data.py` first.")
    st.stop()

basins = get_basins()
names = {}
for b in basins:
    try:
        names[b] = str(get_basin(b)[1].get("gauge_name", ""))
    except Exception:
        names[b] = ""

with st.sidebar:
    st.header("Select basin")
    basin = st.selectbox("USGS gauge", basins, format_func=lambda b: f"{b} · {names.get(b, '')}")
    st.header("Model")
    model_choice = st.radio("Show predictions from", ["LSTM", "Transformer", "Both"], index=2, horizontal=True)
    st.header("Time window")
    wy = st.slider("Water year to zoom into (1 Oct of previous year → 30 Sep)", 1981, 2014, 2011)
    st.caption("Model predictions exist for the test period only: water years 2005 to 2014.")
    st.header("Flood threshold (demo)")
    pct = st.slider("Percentile of the observed daily flow record", 90.0, 99.9, 99.0, 0.1)
    st.caption("Educational demonstration only. CAMELS has no official flood stages; this is NOT a real warning level.")

ts, attrs, area_m2, summary = get_basin(basin)
start, end = water_year_bounds(wy)

# ---- basin information ------------------------------------------------------------------
st.subheader(f"Basin {basin} — {names.get(basin, '')}")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Area", f"{attrs['area_gages2']:,.0f} km²")
c2.metric("Mean elevation", f"{attrs['elev_mean']:,.0f} m")
c3.metric("Mean slope", f"{attrs['slope_mean']:.1f} m/km")
c4.metric("Forest cover", f"{100 * attrs['frac_forest']:.0f} %")
c5.metric("Aridity (PET/P)", f"{attrs['aridity']:.2f}", help="> 1 means evaporation demand exceeds rainfall")
with st.expander("All static attributes used as model inputs (constant in time)"):
    tbl = pd.DataFrame({"value": [attrs[k] for k in STATIC_ATTRIBUTES], "meaning": list(STATIC_ATTRIBUTES.values())},
                       index=list(STATIC_ATTRIBUTES))
    st.dataframe(tbl, use_container_width=True)
with st.expander("Which variables are inputs and which is the target?"):
    roles = [("DYNAMIC input (daily)", k, v) for k, v in DYNAMIC_INPUTS.items()]
    roles += [("STATIC attribute", k, v) for k, v in STATIC_ATTRIBUTES.items()]
    roles += [("TARGET", k, v) for k, v in TARGET.items()]
    st.dataframe(pd.DataFrame(roles, columns=["role", "variable", "meaning"]), use_container_width=True, hide_index=True)

# ---- weather + observed flow --------------------------------------------------------------
st.subheader(f"Weather forcing and observed river flow — water year {wy}")
window = ts.loc[start:end]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total rain", f"{window['prcp(mm/day)'].sum():,.0f} mm")
k2.metric("Wettest day", f"{window['prcp(mm/day)'].max():.0f} mm", help=str(window["prcp(mm/day)"].idxmax().date()))
k3.metric("Mean flow", f"{window['QObs(mm/d)'].mean():.2f} mm/day")
k4.metric("Peak flow", f"{window['QObs(mm/d)'].max():.1f} mm/day", help=str(window["QObs(mm/d)"].idxmax().date()))
st.plotly_chart(rain_flow_figure(window, None, "Rain, temperature and observed streamflow", show_temp=True),
                use_container_width=True)
with st.expander("Show the raw daily table for this water year"):
    st.dataframe(window[list(DYNAMIC_INPUTS) + ["QObs(mm/d)"]].round(2), use_container_width=True)

# ---- predictions ---------------------------------------------------------------------------
st.subheader("Model prediction: observed vs predicted streamflow (test period 2004-10-01 → 2014-09-30)")
wanted = ["lstm", "transformer"] if model_choice == "Both" else [model_choice.lower()]
sims: dict[str, pd.Series] = {}
for m in wanted:
    p = get_predictions(m)
    if p is not None:
        sub = p[p["basin"] == basin].set_index("date")
        if not sub.empty:
            sims[m] = sub["sim"]
            obs_test = sub["obs"]
if not sims:
    st.info("No predictions found for this model yet. Run `python scripts/train.py` and `python scripts/evaluate.py`.")
else:
    in_test = pd.Timestamp(start) >= pd.Timestamp(TEST_START) and pd.Timestamp(end) <= pd.Timestamp(TEST_END)
    thr = float(np.nanpercentile(ts["QObs(mm/d)"], pct))
    if in_test:
        st.plotly_chart(rain_flow_figure(ts.loc[start:end], sims, f"Water year {wy}", threshold=thr), use_container_width=True)
    else:
        st.warning(f"Water year {wy} is outside the test period, showing the whole test decade instead.")
    st.plotly_chart(rain_flow_figure(ts.loc[TEST_START:TEST_END], sims, "Whole test period", threshold=thr),
                    use_container_width=True)

    # ---- metrics -------------------------------------------------------------------------
    st.subheader("Metrics on the test period")
    rows = []
    for m, s in sims.items():
        sc = M.scorecard(obs_test.values, s.reindex(obs_test.index).values)
        rows.append({"model": LABELS[m], "NSE": sc.nse, "KGE": sc.kge, "r (timing)": sc.r, "α (variability)": sc.alpha,
                     "β (bias)": sc.beta, "RMSE mm/day": sc.rmse, "peak-flow bias FHV %": sc.fhv})
    mt = pd.DataFrame(rows).set_index("model")
    st.dataframe(mt.style.format("{:.3f}"), use_container_width=True)
    st.markdown("**Simple interpretation**")
    for m, s in sims.items():
        sc = M.scorecard(obs_test.values, s.reindex(obs_test.index).values)
        bits = [f"**{LABELS[m]}** — {M.interpret(sc.nse, sc.kge)}"]
        if sc.alpha < 0.85:
            bits.append(f"Peaks are too flat (α = {sc.alpha:.2f}).")
        elif sc.alpha > 1.15:
            bits.append(f"Peaks are too spiky (α = {sc.alpha:.2f}).")
        if abs(sc.beta - 1) > 0.1:
            bits.append(f"Overall volume is {'over' if sc.beta > 1 else 'under'}-estimated by {abs(sc.beta - 1) * 100:.0f} %.")
        if sc.fhv < -15:
            bits.append(f"The highest 2 % of flows are underestimated by {abs(sc.fhv):.0f} % (FHV), which matters for floods.")
        st.write(" ".join(bits))
    if len(sims) == 2:
        a, b = (M.nse(obs_test.values, sims[m].reindex(obs_test.index).values) for m in ("lstm", "transformer"))
        better = "LSTM" if a > b else "Transformer"
        st.write(f"On this basin the **{better}** has the higher NSE ({max(a, b):.3f} vs {min(a, b):.3f}). "
                 "Check the other basins before drawing conclusions: one basin is one data point.")
    all_metrics = get_metric_table()
    if all_metrics is not None and "nse" in all_metrics:
        with st.expander("All basins, both models (median is the usual summary in hydrology papers)"):
            piv = all_metrics.pivot_table(index="basin", columns="model", values=["nse", "kge"]).round(3)
            st.dataframe(piv, use_container_width=True)
            st.dataframe(all_metrics.groupby("model")[["nse", "kge"]].median().round(3).rename(columns=str.upper))

    # ---- flood threshold demo --------------------------------------------------------------
    st.subheader("Educational flood-threshold demonstration")
    st.warning("Demonstration only. The threshold below is a percentile of this basin's own daily-mean flow record, "
               "not an official flood stage. Real systems use return-period floods or agency levels and hourly peaks.")
    obs_t = obs_test
    n_obs = int((obs_t > thr).sum())
    cols = st.columns(1 + len(sims))
    cols[0].metric(f"Threshold = {pct:.1f}th percentile", f"{thr:.1f} mm/day", help="computed over the full 1980-2014 observed record")
    for c, (m, s) in zip(cols[1:], sims.items()):
        s = s.reindex(obs_t.index)
        n_sim = int((s > thr).sum())
        hits = int(((s > thr) & (obs_t > thr)).sum())
        misses = n_obs - hits
        false_alarms = n_sim - hits
        c.metric(f"{LABELS[m]}: days above threshold", f"{n_sim} (observed: {n_obs})",
                 help=f"hits {hits}, missed {misses}, false alarms {false_alarms}")
    st.caption("A 'hit' is a day where both the observation and the model exceed the threshold. Daily-mean flow smooths "
               "out the hourly peak that actually causes a flood, so this understates real flood severity.")

st.divider()
st.caption("Data: CAMELS-US (Newman et al. 2015; Addor et al. 2017), NCAR, CC-BY-4.0, via Zenodo record 15529996. "
           "Models: NeuralHydrology (Kratzert et al. 2022). This is a learning project, not an operational forecast.")
