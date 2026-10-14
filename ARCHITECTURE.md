# ARCHITECTURE.md — the complete map of this project

This file is written to be pasted into an AI assistant (ChatGPT, Claude, …) or read by a new collaborator.
It explains what every folder and file does, how data flows from raw files to charts, what the numbers
mean, and what the known limitations are. It contains no code; it points at the code.

---

## 0. One-paragraph summary

**Neural Hydrology Explorer** is a small, fully reproducible research-and-learning project. It downloads six
real river basins from the CAMELS-US dataset (daily weather 1980–2014, basin attributes, observed river flow at a
USGS gauge), trains two deep-learning *rainfall-runoff* models with the NeuralHydrology library (an **LSTM**
baseline and a **Transformer** experiment) under an identical protocol, evaluates both on a held-out decade
(2004–2014) with hydrology metrics (NSE, KGE, peak-flow bias), draws hydrographs, and presents everything in a
Streamlit app with a Flood-Hub-style educational flood-threshold view. Key result: at the same training budget
the LSTM (median test NSE 0.80) clearly beats the Transformer (0.45); both under-predict the largest floods.

---

## 1. The scientific problem in five lines

1. A **basin** is the land that drains to one point on a river; a **gauge** measures the river flow there.
2. Weather (rain, temperature, radiation, humidity) goes into the basin; river flow comes out, delayed and
   smoothed by soil, groundwater and snow.
3. A rainfall-runoff model predicts the daily flow from the weather history plus basin properties.
4. Neural hydrology learns that mapping from data with sequence models instead of hand-written equations.
5. We compare two sequence models and measure how well each reproduces a decade of observed flow it never saw.

---

## 2. Data flow (end to end)

```
Zenodo record 15529996 (official CAMELS-US archive, NCAR, CC-BY-4.0)
   │   scripts/download_data.py  — opens the 3.4 GB zip remotely (HTTP range requests) and extracts
   │                                only the 2 files per basin we need + 7 attribute tables (~10 MB total)
   ▼
data/camels_us/                       (git-ignored, re-creatable)
   ├── basin_mean_forcing/daymet/<HUC2>/<gauge>_lump_cida_forcing_leap.txt   daily weather per basin
   ├── usgs_streamflow/<HUC2>/<gauge>_streamflow_qc.txt                     daily flow (cubic feet/s)
   ├── camels_attributes_v2.0/camels_{clim,geol,hydro,name,soil,topo,vege}.txt  static attributes
   └── MANIFEST.json                                                         provenance (urls, md5)
   │
   │   scripts/inspect_data.py  — Phase 1: tables that label every variable (dynamic input / static
   │                               attribute / target), example days, figures
   ▼
outputs/figures/data_*.png, outputs/metrics/basin_summary.csv, basin_static_attributes.csv
   │
   │   scripts/train.py --model lstm|transformer|both  — NeuralHydrology training from configs/*.yml
   │                               (device auto: CUDA > Apple MPS > CPU). Then evaluates the test period.
   ▼
runs/<experiment>_<stamp>/           (git-ignored: weights model_epochXXX.pt, config.yml, logs, tensorboard,
   │                                  validation/ and test/ result pickles + metrics csv)
   │   scripts/evaluate.py      — reads the result pickle -> predictions CSV, our metrics (+ NeuralHydrology's
   │                               for cross-checking), hydrograph / scatter / flow-duration figures
   ▼
outputs/predictions/<model>_test.csv  (basin, date, obs, sim in mm/day; committed, ~0.7 MB each)
outputs/metrics/<model>_test_metrics.csv
   │
   │   scripts/compare.py       — LSTM vs Transformer: per-basin table, medians, biggest-flood table,
   │                               run details, comparison figures, comparison.md
   ▼
outputs/metrics/comparison_test.csv, comparison_summary.csv, comparison_peaks.csv, run_details.csv, comparison.md
outputs/figures/compare_*.png
   │
   ▼
app/streamlit_app.py  (reads data/ and outputs/ only)      notebooks/exploration.ipynb (same story, interactive)
```

---

## 3. Repository map (what each file is for)

```
neural-hydrology-explorer/
├── README.md                     front door: purpose, results, charts, quick start
├── ARCHITECTURE.md               this file
├── LEARNING_NOTES.md             every concept (basin, gauge, forcing, NSE, KGE, LSTM, Transformer …) explained
│                                 in plain English with numbers taken from the actual data (17 sections)
├── LICENSE                       MIT (code). Data is CC-BY-4.0 (see data/README.md)
├── pyproject.toml                package metadata for the helper package `nhplay` (src layout), dependencies
├── requirements.txt              exact pinned versions that produced outputs/ (torch 2.14.1, neuralhydrology 1.13.0, …)
├── environment.yml               conda alternative to venv + requirements.txt
├── .gitignore                    excludes data, runs/weights, venv, caches, large generated files
│
├── configs/                      NeuralHydrology run configurations (one flat YAML per run)
│   ├── _shared_data_and_split.yml  documentation copy of the blocks both models share (basins, inputs,
│   │                                target, train/validation/test dates); scripts/check_configs.py enforces it
│   ├── lstm.yml                  model: cudalstm (= PyTorch nn.LSTM + linear head), hidden_size 64,
│   │                             output_dropout 0.4, forget bias 3, lr 1e-3 -> 5e-4 -> 1e-4, 30 epochs, batch 256
│   └── transformer.yml           model: transformer, dynamics_embedding 5->32, statics_embedding 18->32
│                                 (d_model 64), 4 heads, 2 layers, feed-forward 128, positional encoding "sum",
│                                 lr 5e-4 -> 2e-4 -> 5e-5, same epochs/batch/seq_length as the LSTM
│
├── data/
│   ├── README.md                 dataset description, licence, citation, file formats, unit conversion
│   ├── basins.txt                the six USGS gauge ids (one per line); edit to study other basins
│   └── camels_us/                downloaded data (git-ignored)
│
├── scripts/                      the pipeline, run in this order
│   ├── download_data.py          1. fetch the six basins + attributes (~10 MB) and write MANIFEST.json
│   ├── inspect_data.py           2. Phase 1 inspection: variable roles, static attributes, example days, figures
│   ├── check_configs.py          3. fairness guard: both configs agree on 27 shared settings (called by train.py)
│   ├── train.py                  4. train LSTM and/or Transformer, evaluate the test period, write runs/latest_<model>.txt
│   ├── evaluate.py               5. predictions CSV + metrics CSV + figures for one model and period
│   └── compare.py                6. LSTM vs Transformer tables, figures, comparison.md, run_details.csv
│
├── src/nhplay/                   thin readable helper package (NeuralHydrology does the heavy lifting)
│   ├── paths.py                  folder layout constants (PROJECT_ROOT, CAMELS_DIR, RUNS_DIR, OUTPUT_DIR …)
│   ├── data/camels.py            loaders wrapping NeuralHydrology's CAMELS readers; dictionaries that describe
│   │                             every variable (DYNAMIC_INPUTS, STATIC_ATTRIBUTES, TARGET); BasinData object;
│   │                             basin_summary(); raw-cfs reader for teaching the unit conversion
│   ├── models/run.py             pick_device() (mps/cpu/cuda), train(), evaluate(), load_results() (result
│   │                             pickle -> DataFrame obs/sim, predictions clipped at zero like NeuralHydrology),
│   │                             load_nh_metrics(), latest_run(), describe_run() (parameter count, train time)
│   ├── evaluation/metrics.py     NSE, KGE (+ r, alpha, beta), RMSE, MAE, PBIAS, FHV written out line by line with
│   │                             docstrings; scorecard(); interpret() (plain-English sentence for a score)
│   ├── evaluation/compare.py     comparison_table(), summarise() (median per model), markdown_table()
│   ├── evaluation/events.py      water_year(), annual_maxima(), empirical_return_levels() (Weibull plotting
│   │                             positions), exceedance_table() (days above / hits / misses / false alarms),
│   │                             metrics_by_water_year(), monthly_regime(), mm_per_day_to_m3s()
│   └── visualization/plots.py    matplotlib: rain+temperature+flow panels, hydrograph obs vs models (rain on
│                                 top), scatter, flow-duration curve, grouped metric bars; fixed colour roles
│
├── app/streamlit_app.py          Neural Hydrology Explorer (5 tabs, see section 7)
├── notebooks/exploration.ipynb   executed notebook: Phase 1 data story + model results + flood demo
├── tests/
│   ├── test_metrics.py           our NSE/KGE/FHV == NeuralHydrology's reference implementation
│   ├── test_events.py            water years, return levels, exceedance counting, regime ordering
│   └── test_app.py               the Streamlit app renders every tab, with basin and model switching
│
├── outputs/                      results (small files are committed so the repo is self-explaining)
│   ├── figures/                  ~70 PNGs: data_*, hydrograph_*, scatter_*, fdc_*, compare_*
│   ├── metrics/                  CSVs + comparison.md (see section 6)
│   └── predictions/              lstm_test.csv, transformer_test.csv (basin, date, obs, sim)
├── docs/images/                  app screenshot(s) used by the README
└── runs/                         NeuralHydrology run folders with weights/logs (git-ignored; re-created by train.py)
```

---

## 4. The inputs, the target, and the tensor the models see

| Role | Variables | Source | Shape per training example |
|---|---|---|---|
| Dynamic inputs | `prcp(mm/day)`, `tmax(C)`, `tmin(C)`, `srad(W/m2)`, `vp(Pa)` | Daymet, basin mean, daily | 365 days × 5 |
| Static inputs | 18 CAMELS attributes: `area_gages2`, `elev_mean`, `slope_mean`, `frac_forest`, `lai_max`, `soil_depth_pelletier`, `soil_porosity`, `soil_conductivity`, `sand_frac`, `clay_frac`, `p_mean`, `pet_mean`, `aridity`, `frac_snow`, `high_prec_freq`, `high_prec_dur`, `low_prec_freq`, `low_prec_dur` | CAMELS attributes v2.0 | 18, repeated on each of the 365 rows |
| Target | `QObs(mm/d)` = USGS discharge in cfs × 0.0283168 × 86400 / area[m²] × 1000 | USGS gauge | 1 number (flow on the last day) |

* Each example is a `[365, 23]` tensor → one scalar. Batches of 256 examples are drawn from random basins and
  random days of the training period. Six basins × 19 years ≈ 41 000 examples.
* All inputs and the target are standardised with training-period statistics (NeuralHydrology does this).
* Observed flow is never an input, and attributes derived from flow (`q_mean`, `runoff_ratio`) are excluded.
* Columns present in the files but not used: `dayl(s)` (deterministic from latitude/date) and `swe(mm)`
  (all zeros in these files).

### Time split (chronological, water years run 1 Oct → 30 Sep)

| Period | Dates | Use |
|---|---|---|
| Train | 1980-10-01 → 1999-09-30 (19 water years) | gradient descent |
| Validation | 1999-10-01 → 2004-09-30 (5) | watch over-fitting; no tuning was done |
| Test | 2004-10-01 → 2014-09-30 (10) | every reported score |

### The six basins (`data/basins.txt`)

| Gauge | River | km² | Regime |
|---|---|---:|---|
| 01022500 | Narraguagus River, Maine | 574 | humid, forested, snow + rain |
| 02177000 | Chattooga River, Georgia | 527 | wet Appalachian mountains, rain-driven |
| 07057500 | North Fork River, Missouri | 1456 | Ozark karst, groundwater-fed |
| 08171300 | Blanco River, Texas | 1067 | semi-arid flash-flood basin, dry 22 % of days |
| 11532500 | Smith River, California | 1578 | Pacific coast, extreme winter storms |
| 13235000 | South Fork Payette River, Idaho | 1163 | high-elevation snowmelt |

---

## 5. The two models

| | LSTM (`configs/lstm.yml`) | Transformer (`configs/transformer.yml`) |
|---|---|---|
| NeuralHydrology class | `cudalstm` (PyTorch `nn.LSTM`, 1 layer) | `transformer` (PyTorch `TransformerEncoder`, causal mask) |
| Input handling | raw 23 features per day | 5 weather → 32 and 18 statics → 32 via small FC embeddings = d_model 64; sinusoidal positional encoding added |
| Core | 64 memory cells, forget-gate bias 3 | 2 encoder layers × 4 heads, feed-forward 128, dropout 0.1 |
| Head | linear regression head on the last time step, output dropout 0.4 | same |
| Parameters | 22 849 | 67 809 |
| Loss / optimiser | basin-normalised NSE loss, Adam, gradient clip 1 | same |
| Learning rate | 1e-3 → 5e-4 (epoch 15) → 1e-4 (epoch 25) | 5e-4 → 2e-4 → 5e-5 |
| Epochs / batch / seq_length | 30 / 256 / 365 | 30 / 256 / 365 |
| Training time (M1 Max, MPS) | 4.9 min | 32.7 min |

Why an LSTM suits hydrology: its memory cells updated by gates behave like water stores (soil, snow,
groundwater) updated daily. Why a Transformer might help: attention can look directly at the few days that
matter (last week's storm, the warm spell that melted the snow) without carrying them through 365 steps.

---

## 6. Results and the files that hold them

Test decade 2004–2014, median over the six basins (`outputs/metrics/comparison_summary.csv`):

| Model | NSE | KGE | r | α | β | FHV % |
|---|---:|---:|---:|---:|---:|---:|
| LSTM | 0.796 | 0.730 | 0.903 | 0.843 | 1.011 | −9.7 |
| Transformer | 0.451 | 0.641 | 0.701 | 0.772 | 1.032 | −21.2 |

Per-basin numbers: `outputs/metrics/comparison_test.csv` (long format: basin, model, nse, kge, r, alpha, beta,
rmse, mae, pbias, fhv, n_days, run_dir). Biggest flood per basin and what each model predicted that day:
`outputs/metrics/comparison_peaks.csv`. Everything as Markdown: `outputs/metrics/comparison.md`.

Metric meanings (`src/nhplay/evaluation/metrics.py`):
* **NSE** = 1 − Σ(sim−obs)² / Σ(obs−mean)²; 1 perfect, 0 = as good as predicting the mean, dominated by floods.
* **KGE** = 1 − √((r−1)² + (α−1)² + (β−1)²); r = correlation (timing), α = std ratio (variability), β = mean ratio
  (volume bias); −0.41 is the "mean flow" benchmark.
* **FHV** = % bias of the highest 2 % of flows; negative = floods under-predicted.
* **PBIAS** = % volume bias; **RMSE/MAE** in mm/day.

Honest interpretation: the LSTM wins on every basin by NSE at this budget; the Transformer's validation loss was
still falling at epoch 30 and nothing was tuned, so this is a budget-limited result, consistent with the
literature (Liu et al. 2025, HESS). Both models under-predict the biggest flood of the decade in every basin by a
factor 2–3 (e.g. Smith River 2008-12-28: observed 118 mm/day, LSTM 36, Transformer 34). The LSTM over-predicts
low flows in the dry Blanco basin (PBIAS +68 %).

### Figures (`outputs/figures/`)

| Pattern | Content |
|---|---|
| `data_<basin>_wy2010.png`, `data_<basin>_full.png` | rain (inverted bars), temperature, observed flow — one water year and the full record |
| `hydrograph_<model>_<basin>_wy2011.png`, `..._test_full.png` | observed vs one model, with rain on top |
| `compare_hydrograph_<basin>_wy2011.png`, `..._test_full.png` | observed vs both models |
| `compare_peak_<basin>.png` | ±30 days around the biggest observed flood of the test decade |
| `compare_nse.png`, `compare_kge.png` | grouped bars per basin and model |
| `scatter_<model>_test.png` | observed vs predicted, all basins pooled, 1:1 line |
| `fdc_<model>_<basin>_test.png` | flow duration curves (log scale) |

---

## 7. The Streamlit app (`app/streamlit_app.py`)

Reads only `data/` and `outputs/`. Sidebar: basin, model (LSTM / Transformer / Both), and the fixed experiment
setup. Five tabs:

1. **Results overview** — side-by-side NSE/KGE table with median row and progress bars, Δ per basin, NSE and
   KGE bar charts, a one-paragraph reading, full metric details, biggest-flood table, run details, CSV downloads.
2. **Basin & weather** — key attributes, a map of the six gauges, long-term water balance, all 18 static inputs,
   rain + temperature + flow with presets (whole record, wettest year, largest flood on record, any water year).
3. **Predictions** — hydrograph with self-selecting windows (largest flood of the test decade, whole decade,
   wettest / best-predicted / worst-predicted year), scores for the decade and for the window, plain-English
   reading, year-by-year table, scatter, flow-duration curve, seasonal regime.
4. **Flood thresholds** — educational 2/5/10-year return levels from the 1981–2004 annual maxima (the Flood Hub
   warning/danger/extreme idea), in mm/day and m³/s, drawn on the hydrograph, with days above / hits / misses /
   false alarms per model. Clearly labelled as not an official flood stage.
5. **Methods & glossary** — exact inputs, target, split, loss, model sizes; plain-English definitions.

Run: `streamlit run app/streamlit_app.py --server.port 8502`.

---

## 8. Relation to Google Flood Hub / OpenHydroNet

Google's operational system uses the same model family (LSTM) at global scale: ~16 000 gauges, weather
*forecasts* (ECMWF HRES, GraphCast, IMERG, CPC, ERA5-Land) and HydroATLAS attributes, a 512-cell forecast LSTM
with a probabilistic (CMAL) head and 7-day horizon, return-period thresholds (2/5/20-year) for warning levels,
then inundation mapping and public alerts. Google open-sourced the framework in June 2026 as **OpenHydroNet**
(github.com/google-research/flood-forecasting, Apache-2.0), a fork of NeuralHydrology, so everything here
transfers. Its pretrained weights were trained on the full 1982–2023 record with no temporal hold-out; the
repository warns that evaluating them on historical dates is invalid (in-sample). This project therefore trains
from scratch and uses no pretrained weights.

---

## 9. Known limitations

* Six basins, one random seed, 30 epochs, no hyper-parameter search: a learning setup, not a benchmark.
* Simulation with observed weather, not a forecast with weather forecasts; daily means hide hourly peaks.
* Extreme floods are under-predicted by both models; dry-basin zero flows are not reproduced by the LSTM.
* Flood thresholds in the app are empirical demonstrations, not official flood stages.
* Historical data only (Daymet to 2014, attributes from 2017).

---

## 10. How to extend (each is a small change)

* More basins: add USGS ids to `data/basins.txt`, re-run download → train → evaluate → compare.
* Ungauged prediction: put different ids in `train_basin_file` and `test_basin_file` in both configs.
* Longer / tuned Transformer: edit `configs/transformer.yml` (epochs, learning_rate, transformer_nlayers, …),
  keep the test dates untouched; run several seeds.
* Probabilistic output: `head: cmal` in the configs (NeuralHydrology supports it).
* Different forcing: `forcings: [nldas]` or `[maurer]` (also in the CAMELS archive; download script needs the
  matching file pattern).
* Pakistan or any other region: build Caravan-style files (ERA5-Land forcing + local gauges), use
  `dataset: generic` or `caravan` in the configs, fine-tune from a CAMELS or OpenHydroNet model.

---

## 11. Reproduction commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python scripts/download_data.py
python scripts/inspect_data.py
python scripts/train.py --model both
python scripts/evaluate.py
python scripts/compare.py
pytest -q
streamlit run app/streamlit_app.py --server.port 8502
```

Hardware used: Apple M1 Max, 32 GB, PyTorch MPS backend; everything also runs on CPU (slower) or CUDA.
