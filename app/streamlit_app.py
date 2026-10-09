"""Neural Hydrology Explorer - presentation-ready Streamlit interface over the project's data and outputs.

Run from the project root:   streamlit run app/streamlit_app.py --server.port 8502

The app only READS data/ and outputs/ (produced by scripts/download_data.py, train.py, evaluate.py,
compare.py). It never trains anything. Every number shown for the models is from the test decade
(2004-10-01 .. 2014-09-30) that neither model saw during training or model selection.
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
from nhplay.evaluation import events as E  # noqa: E402
from nhplay.evaluation import metrics as M  # noqa: E402
from nhplay.paths import CAMELS_DIR, METRIC_DIR, PREDICTION_DIR, RUNS_DIR  # noqa: E402

# ------------------------------------------------------------------------------------------ constants
COLORS = {"observed": "#2b2b2b", "lstm": "#2a78d6", "transformer": "#eb6834", "rain": "#1baf7a",
          "tmax": "#eda100", "tmin": "#2a78d6", "muted": "#9a9a9a"}
LEVEL_COLORS = {"warning-like (2-year)": "#fab219", "danger-like (5-year)": "#ec835a", "extreme-like (10-year)": "#d03b3b"}
LABELS = {"lstm": "LSTM", "transformer": "Transformer"}
TRAIN = ("1980-10-01", "1999-09-30")
VALID = ("1999-10-01", "2004-09-30")
TEST = ("2004-10-01", "2014-09-30")
TEST_WATER_YEARS = list(range(2005, 2015))
FULL = "stretch"

st.set_page_config(page_title="Neural Hydrology Explorer", page_icon="🌊", layout="wide")


# ------------------------------------------------------------------------------------------ data access
@st.cache_data(show_spinner=False)
def get_basins() -> list[str]:
    return read_basin_list()


@st.cache_data(show_spinner="Loading basin data ...")
def get_basin(basin: str):
    bd = load_basin(basin)
    return bd.timeseries, bd.attributes, bd.area_m2, basin_summary(bd)


@st.cache_data(show_spinner=False)
def get_names() -> dict[str, str]:
    out = {}
    for b in get_basins():
        try:
            out[b] = str(get_basin(b)[1].get("gauge_name", ""))
        except Exception:
            out[b] = ""
    return out


@st.cache_data(show_spinner=False)
def get_predictions(model: str) -> pd.DataFrame | None:
    f = PREDICTION_DIR / f"{model}_test.csv"
    if not f.exists():
        return None
    return pd.read_csv(f, dtype={"basin": str}, parse_dates=["date"])


@st.cache_data(show_spinner=False)
def get_csv(name: str) -> pd.DataFrame | None:
    f = METRIC_DIR / name
    if not f.exists():
        return None
    return pd.read_csv(f, dtype={"basin": str})


@st.cache_data(show_spinner=False)
def get_run_details() -> pd.DataFrame | None:
    rows = {}
    for m in ("lstm", "transformer"):
        pointer = RUNS_DIR / f"latest_{m}.txt"
        if not pointer.exists():
            continue
        run_dir = Path(pointer.read_text().strip())
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        if not run_dir.exists():
            continue
        try:
            from nhplay.models.run import describe_run

            d = describe_run(run_dir)
            d["run_dir"] = run_dir.name
            rows[LABELS[m]] = d
        except Exception:
            continue
    if not rows:
        return None
    df = pd.DataFrame(rows).T
    df.index.name = "model"
    return df


def basin_predictions(basin: str, models: list[str]) -> tuple[pd.Series | None, dict[str, pd.Series]]:
    obs, sims = None, {}
    for m in models:
        p = get_predictions(m)
        if p is None:
            continue
        sub = p[p["basin"] == basin].set_index("date").sort_index()
        if sub.empty:
            continue
        sims[m] = sub["sim"]
        obs = sub["obs"]
    return obs, sims


def wy_bounds(wy: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(f"{wy - 1}-10-01"), pd.Timestamp(f"{wy}-09-30")


# ------------------------------------------------------------------------------------------ figures
def rain_flow_figure(ts: pd.DataFrame, sims: dict[str, pd.Series] | None, title: str,
                     levels: dict[str, float] | None = None, show_temp: bool = False, height: int | None = None) -> go.Figure:
    rows = 3 if show_temp else 2
    heights = [0.22, 0.2, 0.58] if show_temp else [0.25, 0.75]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=heights)
    rain = ts["prcp(mm/day)"]
    if len(ts) > 3 * 366:  # long window: monthly totals instead of thousands of 1-day bars
        rain_m = rain.resample("MS").sum()
        fig.add_trace(go.Bar(x=rain_m.index, y=rain_m.values, name="Rain (mm/month)", marker_color=COLORS["rain"],
                             hovertemplate="%{x|%b %Y}<br>rain %{y:.0f} mm/month<extra></extra>"), row=1, col=1)
        fig.update_yaxes(title_text="rain mm/month", autorange="reversed", row=1, col=1)
    else:
        fig.add_trace(go.Bar(x=rain.index, y=rain.values, name="Rain (mm/day)", marker_color=COLORS["rain"],
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
                                     line=dict(color=COLORS.get(key), width=1.4),
                                     hovertemplate="%{x|%d %b %Y}<br>" + LABELS.get(key, key) + " %{y:.2f} mm/day<extra></extra>"),
                          row=r, col=1)
    if levels:
        for name, thr in levels.items():
            if thr is None or np.isnan(thr):
                continue
            fig.add_hline(y=thr, line=dict(color=LEVEL_COLORS.get(name, COLORS["muted"]), dash="dash", width=1.3),
                          annotation_text=f"{name}: {thr:.1f}", annotation_position="top left",
                          annotation_font_color=LEVEL_COLORS.get(name, COLORS["muted"]), row=r, col=1)
    fig.update_yaxes(title_text="streamflow mm/day", row=r, col=1)
    fig.update_layout(title=title, height=height or (560 if show_temp else 470), hovermode="x unified", bargap=0,
                      legend=dict(orientation="h", y=1.08, x=0), margin=dict(l=10, r=10, t=70, b=10))
    return fig


def metric_bar_figure(table: pd.DataFrame, metric: str, title: str) -> go.Figure:
    fig = go.Figure()
    for m in ("lstm", "transformer"):
        sub = table[table["model"] == m].sort_values("basin")
        if sub.empty:
            continue
        fig.add_trace(go.Bar(x=sub["basin"], y=sub[metric], name=LABELS[m], marker_color=COLORS[m],
                             text=sub[metric].map(lambda v: f"{v:.2f}"), textposition="outside",
                             hovertemplate="basin %{x}<br>" + LABELS[m] + " " + metric.upper() + " %{y:.3f}<extra></extra>"))
    fig.update_layout(title=title, barmode="group", height=360, yaxis_title=metric.upper(), xaxis_title="USGS gauge",
                      legend=dict(orientation="h", y=1.12, x=0), margin=dict(l=10, r=10, t=60, b=10))
    fig.add_hline(y=0, line=dict(color=COLORS["muted"], width=1))
    return fig


def scatter_figure(obs: pd.Series, sims: dict[str, pd.Series], title: str) -> go.Figure:
    fig = go.Figure()
    hi = float(max([obs.max()] + [s.max() for s in sims.values()]))
    fig.add_trace(go.Scatter(x=[0, hi], y=[0, hi], mode="lines", name="1:1 line", line=dict(color=COLORS["muted"], dash="dash")))
    for m, s in sims.items():
        s = s.reindex(obs.index)
        fig.add_trace(go.Scattergl(x=obs.values, y=s.values, mode="markers", name=LABELS[m],
                                   marker=dict(size=5, color=COLORS[m], opacity=0.45),
                                   hovertemplate="observed %{x:.2f}<br>" + LABELS[m] + " %{y:.2f}<extra></extra>"))
    fig.update_layout(title=title, height=380, xaxis_title="observed mm/day", yaxis_title="predicted mm/day",
                      legend=dict(orientation="h", y=1.12, x=0), margin=dict(l=10, r=10, t=60, b=10))
    return fig


def fdc_figure(obs: pd.Series, sims: dict[str, pd.Series], title: str) -> go.Figure:
    def fdc(s):
        v = np.sort(s.dropna().values)[::-1]
        p = 100 * np.arange(1, len(v) + 1) / (len(v) + 1)
        return p, np.maximum(v, 1e-4)

    fig = go.Figure()
    p, v = fdc(obs)
    fig.add_trace(go.Scatter(x=p, y=v, name="Observed", line=dict(color=COLORS["observed"], width=2)))
    for m, s in sims.items():
        p, v = fdc(s.reindex(obs.dropna().index))
        fig.add_trace(go.Scatter(x=p, y=v, name=LABELS[m], line=dict(color=COLORS[m], width=1.5)))
    fig.update_layout(title=title, height=380, xaxis_title="% of days this flow is exceeded",
                      yaxis_title="mm/day (log)", yaxis_type="log", legend=dict(orientation="h", y=1.12, x=0),
                      margin=dict(l=10, r=10, t=60, b=10))
    return fig


def regime_figure(regime: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for col in regime.columns:
        key = "observed" if col == "observed" else col
        fig.add_trace(go.Scatter(x=regime.index, y=regime[col], name=LABELS.get(col, "Observed"), mode="lines+markers",
                                 line=dict(color=COLORS[key], width=2 if col == "observed" else 1.5)))
    fig.update_layout(title=title, height=380, xaxis_title="month (water year order)", yaxis_title="mean flow mm/day",
                      legend=dict(orientation="h", y=1.12, x=0), margin=dict(l=10, r=10, t=60, b=10))
    return fig


# ------------------------------------------------------------------------------------------ page header
st.title("🌊 Neural Hydrology Explorer")
st.caption("weather (rain, temperature, radiation, humidity) + static basin attributes → LSTM / Transformer → "
           "daily streamflow → observed vs predicted hydrograph → NSE / KGE → educational flood thresholds")

if not (CAMELS_DIR / "camels_attributes_v2.0").exists():
    st.error("No data found. Run `python scripts/download_data.py` first.")
    st.stop()

basins = get_basins()
names = get_names()

with st.sidebar:
    st.header("Basin")
    basin = st.selectbox("USGS gauge", basins, format_func=lambda b: f"{b} · {names.get(b, '')}")
    st.header("Model")
    model_choice = st.radio("Show", ["Both", "LSTM", "Transformer"], index=0, horizontal=True)
    st.divider()
    st.markdown(
        "**Fixed experiment setup**\n\n"
        f"- Data: CAMELS-US, 6 basins, Daymet + USGS, 1980-2014\n"
        f"- Train: {TRAIN[0]} → {TRAIN[1]}\n- Validation: {VALID[0]} → {VALID[1]}\n- **Test: {TEST[0]} → {TEST[1]}**\n"
        "- Inputs: 5 daily weather variables × 365 days + 18 static attributes\n"
        "- Target: streamflow in mm/day\n- All model scores below are on the test decade only."
    )

models = ["lstm", "transformer"] if model_choice == "Both" else [model_choice.lower()]
ts, attrs, area_m2, summary = get_basin(basin)
obs_test, sims = basin_predictions(basin, models)
have_preds = obs_test is not None and len(sims) > 0

tab_results, tab_basin, tab_pred, tab_flood, tab_methods = st.tabs(
    ["📊 Results overview", "🗺️ Basin & weather", "📈 Predictions", "🚨 Flood thresholds", "ℹ️ Methods & glossary"])

# ========================================================================================== results overview
with tab_results:
    st.subheader("LSTM vs Transformer on the unseen test decade (2004-10-01 → 2014-09-30)")
    comp = get_csv("comparison_test.csv")
    if comp is None:
        frames = [get_csv(f"{m}_test_metrics.csv") for m in ("lstm", "transformer")]
        frames = [f for f in frames if f is not None]
        comp = pd.concat(frames, ignore_index=True) if frames else None
    if comp is None:
        st.info("No metrics found yet. Run `python scripts/train.py && python scripts/evaluate.py && python scripts/compare.py`.")
    else:
        comp["name"] = comp["basin"].map(names)
        present = [m for m in ("lstm", "transformer") if m in set(comp["model"])]
        wide = comp.pivot(index="basin", columns="model", values=["nse", "kge", "r", "alpha", "beta", "fhv"])
        table = pd.DataFrame(index=wide.index)
        table["river"] = table.index.map(names)
        for metric, label in (("nse", "NSE"), ("kge", "KGE")):
            for m in present:
                table[f"{LABELS[m]} {label}"] = wide[(metric, m)]
            if len(present) == 2:
                table[f"Δ {label} (LSTM − Transformer)"] = wide[(metric, "lstm")] - wide[(metric, "transformer")]
        if len(present) == 2:
            table["better by NSE"] = np.where(wide[("nse", "lstm")] >= wide[("nse", "transformer")], "LSTM", "Transformer")
        median = table.select_dtypes("number").median().to_frame().T
        median.index = ["MEDIAN"]
        median["river"] = "median over basins"
        if len(present) == 2:
            median["better by NSE"] = "LSTM" if median["LSTM NSE"].iloc[0] >= median["Transformer NSE"].iloc[0] else "Transformer"
        show = pd.concat([table, median])
        num_cols = [c for c in show.columns if c not in ("river", "better by NSE")]
        col_cfg = {c: st.column_config.ProgressColumn(c, min_value=0.0, max_value=1.0, format="%.3f")
                   for c in num_cols if c.endswith(("NSE", "KGE")) and not c.startswith("Δ")}
        col_cfg.update({c: st.column_config.NumberColumn(c, format="%+.3f") for c in num_cols if c.startswith("Δ")})
        st.dataframe(show, width=FULL, column_config=col_cfg)
        c1, c2 = st.columns(2)
        c1.plotly_chart(metric_bar_figure(comp, "nse", "NSE per basin (1 = perfect, 0 = as good as the mean flow)"), width=FULL)
        c2.plotly_chart(metric_bar_figure(comp, "kge", "KGE per basin (1 = perfect, −0.41 = as good as the mean flow)"), width=FULL)

        med = comp.groupby("model")[["nse", "kge"]].median()
        if len(present) == 2:
            lstm_wins = int((wide[("nse", "lstm")] >= wide[("nse", "transformer")]).sum())
            st.markdown(
                f"**One-paragraph reading.** Median NSE: LSTM **{med.loc['lstm', 'nse']:.3f}** vs Transformer "
                f"**{med.loc['transformer', 'nse']:.3f}**; median KGE {med.loc['lstm', 'kge']:.3f} vs {med.loc['transformer', 'kge']:.3f}. "
                f"The LSTM has the higher NSE in {lstm_wins} of {len(wide)} basins. Both models were trained with the same data, "
                "inputs, split, epochs and batch size; only the architecture differs. The Transformer's validation loss was "
                "still decreasing at the last epoch, so this is a result *at this training budget*, not a verdict on the "
                "architecture. Both models under-predict the largest flood of the decade in every basin (see the table below)."
            )

        with st.expander("Full metric details per basin and model (r, α, β, bias, peak-flow bias FHV)", expanded=False):
            detail = comp[["basin", "name", "model", "nse", "kge", "r", "alpha", "beta", "pbias", "fhv", "rmse", "mae"]].copy()
            detail["model"] = detail["model"].map(LABELS)
            st.dataframe(detail.round(3), width=FULL, hide_index=True)
            st.caption("r = correlation (timing/shape), α = std ratio (variability), β = mean ratio (bias), "
                       "pbias = % volume bias, FHV = % bias of the top 2 % of flows (negative = floods underestimated).")

        peaks = get_csv("comparison_peaks.csv")
        if peaks is not None:
            st.markdown("**The biggest observed flood of the test decade in each basin, and what each model said that day** (mm/day)")
            pk = peaks.rename(columns={"peak_date": "date", "rain_that_day_mm": "rain that day (mm)",
                                       "rain_prev_3_days_mm": "rain previous 3 days (mm)", "observed_mm_per_day": "observed",
                                       "lstm_mm_per_day": "LSTM", "transformer_mm_per_day": "Transformer"})
            for m in ("LSTM", "Transformer"):
                if m in pk.columns:
                    pk[f"{m} % of observed"] = (100 * pk[m] / pk["observed"]).round(0)
            st.dataframe(pk, width=FULL, hide_index=True)

        details = get_run_details()
        if details is not None:
            st.markdown("**Run details** (identical data and protocol; only the model block differs)")
            st.dataframe(details.astype(str), width=FULL)

        d1, d2 = st.columns(2)
        d1.download_button("Download per-basin metrics (CSV)", comp.to_csv(index=False).encode(), "comparison_test.csv", "text/csv")
        md = METRIC_DIR / "comparison.md"
        if md.exists():
            d2.download_button("Download comparison.md", md.read_bytes(), "comparison.md", "text/markdown")

# ========================================================================================== basin & weather
with tab_basin:
    st.subheader(f"Basin {basin} — {names.get(basin, '')}")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Area", f"{attrs['area_gages2']:,.0f} km²")
    c2.metric("Mean elevation", f"{attrs['elev_mean']:,.0f} m")
    c3.metric("Mean slope", f"{attrs['slope_mean']:.1f} m/km")
    c4.metric("Forest cover", f"{100 * attrs['frac_forest']:.0f} %")
    c5.metric("Aridity PET/P", f"{attrs['aridity']:.2f}", help="> 1: evaporation demand exceeds rainfall")
    c6.metric("Snow fraction", f"{100 * attrs['frac_snow']:.0f} %", help="share of precipitation falling as snow")

    m1, m2 = st.columns([1, 1.3])
    with m1:
        pts = []
        for b in basins:
            try:
                a = get_basin(b)[1]
                pts.append({"lat": float(a["gauge_lat"]), "lon": float(a["gauge_lon"]),
                            "size": 60000 if b == basin else 25000, "color": COLORS["transformer"] if b == basin else COLORS["lstm"]})
            except Exception:
                pass
        if pts:
            st.map(pd.DataFrame(pts), latitude="lat", longitude="lon", size="size", color="color", zoom=2, height=330)
            st.caption("Orange = selected gauge, blue = the other five basins in this project.")
    with m2:
        st.markdown("**Long-term water balance (1980-2014)**")
        wb = pd.DataFrame({
            "value": [f"{summary['mean_rain_mm_per_day']:.2f} mm/day", f"{summary['mean_flow_mm_per_day']:.2f} mm/day",
                      f"{100 * summary['mean_flow_mm_per_day'] / summary['mean_rain_mm_per_day']:.0f} %",
                      f"{summary['max_flow_mm_per_day']:.1f} mm/day on {summary['date_of_max_flow']}",
                      f"{int((ts['QObs(mm/d)'] < 0.01).sum())} of {len(ts)} days",
                      f"{summary['n_days_with_flow_obs']} of {summary['n_days']} days"],
        }, index=["mean precipitation", "mean streamflow", "runoff ratio (flow / rain)", "largest daily flow on record",
                  "dry days (< 0.01 mm/day)", "days with a flow observation"])
        st.table(wb)
    with st.expander("All 18 static attributes used as model inputs (constant in time)"):
        tbl = pd.DataFrame({"value": [attrs[k] for k in STATIC_ATTRIBUTES], "meaning": list(STATIC_ATTRIBUTES.values())},
                           index=list(STATIC_ATTRIBUTES))
        st.dataframe(tbl, width=FULL)
    with st.expander("Which variables are inputs and which is the target?"):
        roles = [("DYNAMIC input (daily)", k, v) for k, v in DYNAMIC_INPUTS.items()]
        roles += [("STATIC attribute", k, v) for k, v in STATIC_ATTRIBUTES.items()]
        roles += [("TARGET", k, v) for k, v in TARGET.items()]
        st.dataframe(pd.DataFrame(roles, columns=["role", "variable", "meaning"]), width=FULL, hide_index=True)

    st.subheader("Weather forcing and observed river flow")
    flow_all = ts["QObs(mm/d)"]
    rain_by_wy = ts["prcp(mm/day)"].groupby(E.water_year(ts.index)).sum()
    rain_by_wy = rain_by_wy[(rain_by_wy.index >= 1981) & (rain_by_wy.index <= 2014)]
    wettest_wy = int(rain_by_wy.idxmax())
    peak_day_all = flow_all.idxmax()
    preset = st.radio("Time window", ["Whole record 1980-2014", f"Wettest water year ({wettest_wy})",
                                      f"Largest flood on record (±45 days around {peak_day_all.date()})", "Choose a water year"],
                      horizontal=True, key="basin_window")
    if preset.startswith("Whole"):
        start, end = ts.index.min(), ts.index.max()
    elif preset.startswith("Wettest"):
        start, end = wy_bounds(wettest_wy)
    elif preset.startswith("Largest"):
        start, end = peak_day_all - pd.Timedelta(days=45), peak_day_all + pd.Timedelta(days=45)
    else:
        wy = st.slider("Water year (1 Oct of previous year → 30 Sep)", 1981, 2014, 2010, key="basin_wy")
        start, end = wy_bounds(wy)
    window = ts.loc[start:end]
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Window", f"{window.index.min().date()} → {window.index.max().date()}")
    k2.metric("Total rain", f"{window['prcp(mm/day)'].sum():,.0f} mm")
    k3.metric("Wettest day", f"{window['prcp(mm/day)'].max():.0f} mm", help=str(window["prcp(mm/day)"].idxmax().date()))
    k4.metric("Mean flow", f"{window['QObs(mm/d)'].mean():.2f} mm/day")
    k5.metric("Peak flow", f"{window['QObs(mm/d)'].max():.1f} mm/day", help=str(window["QObs(mm/d)"].idxmax().date()))
    st.plotly_chart(rain_flow_figure(window, None, "Rain (top), temperature (middle), observed streamflow (bottom)", show_temp=True), width=FULL)
    with st.expander("Raw daily table for this window (dynamic inputs + target)"):
        st.dataframe(window[list(DYNAMIC_INPUTS) + ["QObs(mm/d)"]].round(2), width=FULL)

# ========================================================================================== predictions
with tab_pred:
    st.subheader(f"Observed vs predicted streamflow — {basin} {names.get(basin, '')}")
    if not have_preds:
        st.info("No predictions for this model yet. Run `python scripts/train.py` and `python scripts/evaluate.py`.")
    else:
        by_year = E.metrics_by_water_year(obs_test, sims)
        first = models[0] if models[0] in sims else next(iter(sims))
        peak_day = obs_test.idxmax()
        rain_test = ts.loc[TEST[0]:TEST[1], "prcp(mm/day)"]
        wettest_test_wy = int(rain_test.groupby(E.water_year(rain_test.index)).sum().idxmax())
        best_wy = int(by_year[f"{first}_nse"].idxmax())
        worst_wy = int(by_year[f"{first}_nse"].idxmin())
        options = [f"Largest flood of the test decade (±45 days around {peak_day.date()})",
                   "Whole test decade 2004-2014",
                   f"Water year with the largest flood ({int(E.water_year(pd.DatetimeIndex([peak_day]))[0])})",
                   f"Wettest test water year ({wettest_test_wy})",
                   f"Best-predicted water year by {LABELS[first]} ({best_wy})",
                   f"Worst-predicted water year by {LABELS[first]} ({worst_wy})",
                   "Choose a water year"]
        preset = st.radio("Time window", options, horizontal=True, key="pred_window")
        if preset.startswith("Largest flood"):
            start, end = peak_day - pd.Timedelta(days=45), peak_day + pd.Timedelta(days=45)
        elif preset.startswith("Whole"):
            start, end = pd.Timestamp(TEST[0]), pd.Timestamp(TEST[1])
        elif preset.startswith("Water year with"):
            start, end = wy_bounds(int(E.water_year(pd.DatetimeIndex([peak_day]))[0]))
        elif preset.startswith("Wettest"):
            start, end = wy_bounds(wettest_test_wy)
        elif preset.startswith("Best"):
            start, end = wy_bounds(best_wy)
        elif preset.startswith("Worst"):
            start, end = wy_bounds(worst_wy)
        else:
            wy = st.select_slider("Water year (test decade only)", TEST_WATER_YEARS, value=2011, key="pred_wy")
            start, end = wy_bounds(wy)
        start = max(start, pd.Timestamp(TEST[0]))
        end = min(end, pd.Timestamp(TEST[1]))
        win = ts.loc[start:end]
        st.plotly_chart(rain_flow_figure(win, sims, f"{start.date()} → {end.date()}"), width=FULL)

        # ---- window table + whole-period metrics
        w_obs = obs_test.loc[start:end]
        rows = []
        for m, s in sims.items():
            s_w = s.reindex(w_obs.index)
            s_all = s.reindex(obs_test.index)
            sc_all = M.scorecard(obs_test.values, s_all.values)
            sc_w = M.scorecard(w_obs.values, s_w.values)
            rows.append({"model": LABELS[m],
                         "NSE (test decade)": sc_all.nse, "KGE (test decade)": sc_all.kge, "r": sc_all.r, "α": sc_all.alpha,
                         "β": sc_all.beta, "RMSE mm/day": sc_all.rmse, "peak-flow bias FHV %": sc_all.fhv,
                         "NSE (this window)": sc_w.nse, "peak observed (window)": float(w_obs.max()),
                         f"peak predicted (window)": float(s_w.loc[w_obs.idxmax()])})
        mt = pd.DataFrame(rows).set_index("model")
        st.markdown("**Scores** — left block: whole test decade (the official numbers); right block: the window shown above")
        st.dataframe(mt.style.format("{:.3f}"), width=FULL)
        st.markdown("**Plain-English reading**")
        for m, s in sims.items():
            sc = M.scorecard(obs_test.values, s.reindex(obs_test.index).values)
            bits = [f"**{LABELS[m]}** — {M.interpret(sc.nse, sc.kge)}"]
            if sc.alpha < 0.85:
                bits.append(f"Peaks are too flat (α = {sc.alpha:.2f}).")
            elif sc.alpha > 1.15:
                bits.append(f"Peaks are too spiky (α = {sc.alpha:.2f}).")
            if abs(sc.beta - 1) > 0.1:
                bits.append(f"Total volume is {'over' if sc.beta > 1 else 'under'}-estimated by {abs(sc.beta - 1) * 100:.0f} %.")
            if sc.fhv < -15:
                bits.append(f"The highest 2 % of flows are {abs(sc.fhv):.0f} % too low (FHV), which matters for floods.")
            st.write(" ".join(bits))
        if len(sims) == 2:
            a, b = (M.nse(obs_test.values, sims[m].reindex(obs_test.index).values) for m in ("lstm", "transformer"))
            st.write(f"On this basin the **{'LSTM' if a > b else 'Transformer'}** has the higher NSE ({max(a, b):.3f} vs {min(a, b):.3f}). "
                     "One basin is one data point; see the Results overview tab for all six.")

        # ---- per water year table
        st.markdown("**Year by year** (NSE / KGE per water year of the test decade, and the observed vs predicted annual peak)")
        yt = by_year.copy()
        rename = {"observed_peak_mm_per_day": "observed peak", "peak_date": "peak date", "observed_mean_mm_per_day": "observed mean"}
        for m in sims:
            rename.update({f"{m}_nse": f"{LABELS[m]} NSE", f"{m}_kge": f"{LABELS[m]} KGE", f"{m}_peak_mm_per_day": f"{LABELS[m]} peak"})
        yt = yt.rename(columns=rename)
        order = ["observed mean", "observed peak", "peak date"] + [f"{LABELS[m]} peak" for m in sims] + \
                [f"{LABELS[m]} NSE" for m in sims] + [f"{LABELS[m]} KGE" for m in sims]
        yt = yt[[c for c in order if c in yt.columns]]
        cfg = {c: st.column_config.ProgressColumn(c, min_value=0.0, max_value=1.0, format="%.2f") for c in yt.columns if c.endswith(("NSE", "KGE"))}
        cfg.update({c: st.column_config.NumberColumn(c, format="%.1f") for c in yt.columns if c.endswith(("peak", "mean"))})
        st.dataframe(yt, width=FULL, column_config=cfg)
        st.download_button("Download year-by-year table (CSV)", yt.to_csv().encode(), f"{basin}_by_water_year.csv", "text/csv")

        # ---- diagnostic plots
        st.markdown("**Diagnostics** (whole test decade)")
        p1, p2, p3 = st.columns(3)
        p1.plotly_chart(scatter_figure(obs_test, sims, "Observed vs predicted (every day)"), width=FULL)
        p2.plotly_chart(fdc_figure(obs_test, sims, "Flow duration curve"), width=FULL)
        p3.plotly_chart(regime_figure(E.monthly_regime(obs_test, sims), "Seasonal regime: mean flow by month"), width=FULL)
        st.caption("Scatter: a perfect model puts every day on the 1:1 line. Flow duration curve: how often each flow level is "
                   "exceeded, log scale, shows bias in floods (left) and droughts (right). Regime: does the model get the seasons right?")

# ========================================================================================== flood thresholds
with tab_flood:
    st.subheader("Educational flood thresholds: return levels from annual maxima")
    st.warning("Demonstration only. These levels are empirical estimates from 24 annual maxima of this gauge's record "
               "before the test decade (water years 1981-2004). They are NOT official flood stages. Google Flood Hub "
               "uses the 2-, 5- and 20-year return-period floods as its 'warning', 'danger' and 'extreme' levels, or official "
               "agency levels where they exist; we use 2, 5 and 10 years because 24 years of data cannot support a 20-year estimate.")
    maxima = E.annual_maxima(ts["QObs(mm/d)"], TRAIN[0], VALID[1])
    rl = E.empirical_return_levels(maxima["peak_mm_per_day"], (2, 5, 10))
    levels = {"warning-like (2-year)": rl[2], "danger-like (5-year)": rl[5], "extreme-like (10-year)": rl[10]}
    lv = pd.DataFrame({
        "return period": ["2 years", "5 years", "10 years"],
        "meaning": ["exceeded on average once every 2 years", "once every 5 years", "once every 10 years"],
        "level mm/day": [rl[2], rl[5], rl[10]],
        "level m³/s at the gauge": [E.mm_per_day_to_m3s(v, area_m2) for v in (rl[2], rl[5], rl[10])],
    }, index=list(levels))
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown("**Threshold levels for this basin**")
        st.dataframe(lv.style.format({"level mm/day": "{:.1f}", "level m³/s at the gauge": "{:,.0f}"}), width=FULL)
        st.markdown("**Annual maximum daily flow, water years 1981-2004** (the data the levels come from)")
        st.dataframe(maxima.rename(columns={"peak_date": "date", "peak_mm_per_day": "annual max mm/day"}).round(2), width=FULL, height=240)
    with c2:
        if have_preds:
            st.markdown("**Exceedances in the test decade** (daily values; a 'hit' is a day when both observation and model are above the level)")
            ex = E.exceedance_table(obs_test, sims, levels)
            ren = {"threshold_mm_per_day": "threshold mm/day", "observed_days_above": "observed days above", "observed_episodes": "observed events"}
            for m in sims:
                ren.update({f"{m}_days_above": f"{LABELS[m]} days above", f"{m}_hits": f"{LABELS[m]} hits",
                            f"{m}_missed": f"{LABELS[m]} missed", f"{m}_false_alarms": f"{LABELS[m]} false alarms"})
            st.dataframe(ex.rename(columns=ren).set_index("level"), width=FULL)
            st.caption("'events' counts separate runs of consecutive days above the level. Daily-mean flow smooths the hourly "
                       "peak that actually overtops a bank, so real severity is understated.")
        else:
            st.info("Train and evaluate a model to see how often it crosses these levels.")
    if have_preds:
        st.plotly_chart(rain_flow_figure(ts.loc[TEST[0]:TEST[1]], sims, "Test decade with the three threshold levels", levels=levels, height=520), width=FULL)
    else:
        st.plotly_chart(rain_flow_figure(ts.loc[TEST[0]:TEST[1]], None, "Test decade with the three threshold levels", levels=levels, height=520), width=FULL)

# ========================================================================================== methods
with tab_methods:
    st.subheader("What the models see and what they predict")
    setup = pd.DataFrame({
        "item": ["Dynamic inputs", "Static inputs", "Target", "Sequence length", "Training period", "Validation period",
                 "Test period (all scores)", "Loss", "LSTM", "Transformer", "Hardware", "Software"],
        "value": ["prcp(mm/day), tmax(C), tmin(C), srad(W/m2), vp(Pa) — Daymet, basin-mean, daily",
                  "18 CAMELS attributes: area, elevation, slope, forest, LAI, soil depth/porosity/conductivity/sand/clay, "
                  "p_mean, pet_mean, aridity, frac_snow, high/low precipitation frequency and duration",
                  "QObs(mm/d): USGS daily discharge converted from cfs to mm/day over the basin",
                  "365 days of inputs → 1 prediction (today's flow)",
                  f"{TRAIN[0]} → {TRAIN[1]} (19 water years)", f"{VALID[0]} → {VALID[1]} (5 water years)",
                  f"{TEST[0]} → {TEST[1]} (10 water years, never used for training or model selection)",
                  "basin-normalised NSE loss (Kratzert et al. 2019), Adam, batch 256, 30 epochs, seed 42",
                  "NeuralHydrology 'cudalstm': 1 layer, 64 cells, forget bias 3, output dropout 0.4, 22 849 parameters",
                  "NeuralHydrology 'transformer': inputs embedded to d_model 64 (weather 5→32, attributes 18→32), sinusoidal "
                  "positional encoding, 2 causal encoder layers × 4 heads, feed-forward 128, 67 809 parameters",
                  "Apple M1 Max, PyTorch MPS backend (CPU fallback)",
                  "NeuralHydrology 1.13.0, PyTorch 2.14, Python 3.13"],
    }).set_index("item")
    st.table(setup)
    st.subheader("Glossary")
    gl = pd.DataFrame({
        "term": ["Basin / catchment", "Gauge", "Streamflow (Q)", "mm/day", "Forcing", "Static attributes", "Hydrograph",
                 "NSE", "KGE", "r, α, β", "FHV", "Flow duration curve", "Return period", "Water year"],
        "plain English": [
            "All the land that drains to one point on a river.",
            "The station at the basin outlet that measures water level and converts it to discharge.",
            "Volume of water passing the gauge per unit time.",
            "Streamflow divided by basin area: a depth of water spread over the basin, comparable with rainfall.",
            "The weather that drives the basin: rain, temperature, radiation, humidity.",
            "Numbers describing the basin that do not change in time: area, slope, soils, climate.",
            "Streamflow plotted against time; rain is drawn as bars hanging from the top.",
            "1 − (model squared error / squared error of predicting the mean). 1 = perfect, 0 = as good as the mean. Dominated by floods.",
            "1 − distance from perfect correlation, variability and bias. 1 = perfect; −0.41 = as good as the mean flow.",
            "Correlation (timing), std ratio (are peaks spiky enough?), mean ratio (is the total volume right?).",
            "Percent bias of the highest 2 % of flows. Negative = floods underestimated.",
            "How often each flow level is exceeded; reveals bias in floods and droughts.",
            "A flow exceeded on average once every T years; estimated here from annual maxima.",
            "1 October to 30 September, named after the year in which it ends.",
        ],
    }).set_index("term")
    st.table(gl)
    st.caption("Data: CAMELS-US (Newman et al. 2015; Addor et al. 2017), NCAR, CC-BY-4.0, via Zenodo record 15529996. "
               "Models: NeuralHydrology (Kratzert et al. 2022). Full explanations: LEARNING_NOTES.md; reproduction: README.md.")
