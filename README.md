# Neural Hydrology Playground

A small, readable, fully reproducible project for learning **neural hydrology**: predicting daily river
streamflow from weather and basin attributes with an **LSTM** and a **Transformer**, on real data
(six CAMELS-US basins), with honest out-of-sample evaluation, hydrographs, NSE/KGE metrics, a Streamlit
explorer and plain-English notes. It runs on an Apple-Silicon laptop in about 30 minutes end to end.

It is a learning and research sandbox, not a reproduction of Google Flood Hub. See
[LEARNING_NOTES.md](LEARNING_NOTES.md) for the concepts and
[Relation to Google Flood Hub](#relation-to-google-flood-hub-and-openhydronet) for how the two connect.

---

## 1. What neural hydrology is, in five sentences

A river basin turns weather into river flow: rain and snowmelt go into soil, groundwater and snowpack
and come out, delayed and smoothed, at the gauge. Classical hydrological models describe those stores
with hand-written equations. Neural hydrology (Kratzert et al. 2018, 2019) instead trains a sequence
model on many years of weather (the inputs) and gauge observations (the target) and lets it learn the
storage behaviour from data. Static basin attributes (area, slope, soils, climate) let one model serve
many basins, and even basins without any gauge. The same idea, scaled to ~16 000 gauges and weather
forecasts, is what runs Google Flood Hub today.

## 2. Workflow

```
                 CAMELS-US (NCAR, CC-BY-4.0), six basins, 1980-2014
                 ├── Daymet basin-mean weather, daily        ──┐  DYNAMIC inputs (5 variables x 365 days)
                 ├── CAMELS attributes v2.0                  ──┤  STATIC inputs (18 numbers per basin)
                 └── USGS gauge streamflow (cfs -> mm/day)   ──┘  TARGET (1 number per basin per day)
                                   │
                                   ▼
                 NeuralHydrology (PyTorch, Apple MPS or CPU)
                 ┌───────────────────────────┬────────────────────────────────┐
                 │ LSTM  (configs/lstm.yml)  │ Transformer (configs/transformer.yml) │
                 │ 64 memory cells           │ 2 encoder layers, 4 heads, d_model 64  │
                 └───────────────────────────┴────────────────────────────────┘
                   identical basins, inputs, target, chronological split, epochs, batch size
                                   │
                                   ▼
                 test decade 2004-2014 (never seen): predicted vs observed streamflow
                                   │
                 ┌─────────────────┼──────────────────────┐
                 ▼                 ▼                      ▼
            hydrographs        NSE / KGE / FHV        educational flood-threshold demo
          outputs/figures     outputs/metrics          app/streamlit_app.py
```

## 3. Project structure

```
neural-hydrology-playground/
├── README.md                  <- you are here
├── LEARNING_NOTES.md          <- every concept explained with numbers from our data
├── pyproject.toml             <- package "nhplay" (src layout) + dependencies
├── requirements.txt           <- exact pins that produced outputs/
├── environment.yml            <- conda alternative
├── configs/
│   ├── _shared_data_and_split.yml   data, inputs, target, dates shared by both models (documentation)
│   ├── lstm.yml                     NeuralHydrology config: LSTM baseline
│   └── transformer.yml              NeuralHydrology config: Transformer experiment
├── data/
│   ├── README.md              <- dataset description, licence, file formats, unit conversion
│   ├── basins.txt             <- the six USGS gauge ids
│   └── camels_us/             <- downloaded by scripts/download_data.py (git-ignored, ~10 MB)
├── scripts/
│   ├── download_data.py       <- Phase 1: pull six basins from the official archive (range requests)
│   ├── inspect_data.py        <- Phase 1: label inputs/attributes/target, print examples, plot
│   ├── check_configs.py       <- fairness guard: both configs share everything but the model
│   ├── train.py               <- Phase 2/3: train LSTM and/or Transformer (MPS -> CPU fallback)
│   ├── evaluate.py            <- predictions CSV, metrics, hydrographs for one model
│   └── compare.py             <- LSTM vs Transformer table, figures, comparison.md
├── src/nhplay/
│   ├── paths.py               <- folder layout
│   ├── data/camels.py         <- loaders (wrapping NeuralHydrology's) + variable descriptions
│   ├── models/run.py          <- thin wrapper around NeuralHydrology train/evaluate + result loading
│   ├── evaluation/metrics.py  <- NSE, KGE, ... written out line by line (tested against NeuralHydrology)
│   ├── evaluation/compare.py  <- comparison tables
│   └── visualization/plots.py <- hydrograph, forcing, scatter, flow-duration, metric bar plots
├── notebooks/exploration.ipynb<- the whole story in one notebook
├── app/streamlit_app.py       <- Neural Hydrology Explorer
├── tests/                     <- metrics vs NeuralHydrology reference; app renders
├── outputs/
│   ├── figures/               <- PNGs (committed, small)
│   ├── metrics/               <- CSV + comparison.md (committed)
│   └── predictions/           <- per-day obs/sim CSVs (git-ignored, regenerated by evaluate.py)
└── runs/                      <- NeuralHydrology run folders with weights and logs (git-ignored)
```

## 4. Environment setup

Tested on macOS 26 / Apple M1 Max / Python 3.13. CUDA is not assumed anywhere; PyTorch's Apple **MPS**
backend is used when available and the code falls back to CPU otherwise (MPS was 5 to 13 times faster
than CPU in a quick benchmark of these two models).

```bash
cd neural-hydrology-playground
python3 -m venv .venv && source .venv/bin/activate      # Python >= 3.10
pip install -r requirements.txt                         # exact versions used for outputs/
pip install -e .                                        # the small nhplay helper package
python -c "import torch; print(torch.backends.mps.is_available())"
```

Conda users: `mamba env create -f environment.yml && conda activate neural-hydrology-playground && pip install -e .`

## 5. Get the data (about 10 MB, 1 to 2 minutes)

```bash
python scripts/download_data.py
```

This fetches, for the six basins in `data/basins.txt`, the Daymet forcing and USGS streamflow files from
the official CAMELS-US archive (NCAR, DOI 10.5065/D6MW2F4D, hosted on Zenodo) using HTTP range requests
on the 3.4 GB zip, so only the needed ~1.2 MB per basin is transferred, plus the seven attribute tables.
Provenance (URLs, sizes, md5) is written to `data/camels_us/MANIFEST.json`. Details: [data/README.md](data/README.md).

To study other basins, add USGS gauge ids to `data/basins.txt` (all 671 are listed in
`data/camels_us/basin_metadata/gauge_information.txt` after the first download) and re-run.

## 6. Inspect the data (Phase 1)

```bash
python scripts/inspect_data.py              # prints tables, writes outputs/figures/data_*.png
jupyter lab notebooks/exploration.ipynb     # same, interactively
```

You will see, per basin: a table that marks each variable as **DYNAMIC input**, **STATIC attribute** or
**TARGET**; the static attributes with their meanings; a few example days around the wettest day of a
water year (date, rain, temperatures, radiation, humidity, observed flow); and three-panel figures of
rain, temperature and streamflow. Example (Narraguagus River, Maine, October 2010):

```
            prcp(mm/day)  tmax(C)  tmin(C)  QObs(mm/d)
2010-10-01          8.86    20.87    14.50        0.32
2010-10-02         26.42    19.10     9.67        1.31   <- storm
2010-10-03          0.00    16.43     2.60        1.54   <- peak one day later
2010-10-04          0.00    15.13     5.15        1.09   <- recession
```

![data](outputs/figures/data_01022500_wy2010.png)

## 7. Train the LSTM baseline (Phase 2)

```bash
python scripts/train.py --model lstm          # ~5 min on M1 Max (MPS), 30 epochs
```

One LSTM (64 cells, sequence length 365 days) is trained jointly on all six basins; the 18 static
attributes tell it which basin it is looking at. Loss is the basin-normalised NSE loss of
Kratzert et al. (2019). Validation metrics are printed every epoch; the run folder
`runs/lstm_6basins_<stamp>/` holds `config.yml`, weights, logs and TensorBoard events
(`tensorboard --logdir runs`). The equivalent plain NeuralHydrology command is
`nh-run train --config-file configs/lstm.yml`.

## 8. Train the Transformer (Phase 3)

```bash
python scripts/train.py --model transformer   # ~30 min on M1 Max (MPS), 30 epochs
```

NeuralHydrology's `transformer` model: inputs are embedded to d_model = 64 (weather 5 -> 32, attributes
18 -> 32), a sinusoidal positional encoding is added, two causal self-attention encoder layers with four
heads process the 365-day window, and the same linear head as the LSTM reads the last position.
`scripts/check_configs.py` (run automatically by `train.py`) asserts that both configs agree on all 27
shared settings, so the only difference is the model block.

## 9. Evaluate and compare

```bash
python scripts/evaluate.py            # both models: predictions CSV, metrics CSV, hydrographs, scatter, FDC
python scripts/compare.py             # LSTM vs Transformer table, per-basin hydrographs, comparison.md
pytest -q                             # our metrics == NeuralHydrology's; the app renders
```

Metrics are computed on the **test decade (1 Oct 2004 to 30 Sep 2014)**, which neither model saw during
training or model selection. `outputs/metrics/<model>_test_metrics.csv` lists NSE, KGE and its three
components (r, alpha, beta), RMSE, MAE, percent bias and the peak-flow bias FHV per basin, next to
NeuralHydrology's own numbers for the same series (they agree to 1e-7).

## 10. Launch the Streamlit explorer (Phase 4)

```bash
streamlit run app/streamlit_app.py --server.port 8502
```

Presentation-ready, five tabs, nothing to tune before showing it:

| Tab | What it shows |
|---|---|
| **Results overview** | side-by-side NSE/KGE table for both models with a median row and progress bars, NSE and KGE bar charts, a one-paragraph reading, full metric details (r, α, β, bias, FHV), the biggest flood per basin with each model's prediction, run details (parameters, epochs, training time), CSV downloads |
| **Basin & weather** | area/elevation/slope/forest/aridity/snow, a map of the six gauges, long-term water balance, all 18 static attributes with meanings, rain + temperature + flow for presets (whole record, wettest year, largest flood on record, or any water year) |
| **Predictions** | observed vs LSTM/Transformer hydrograph with presets that pick themselves (largest flood of the test decade, whole decade, wettest / best-predicted / worst-predicted water year), scores for the decade and for the window shown, plain-English reading, a year-by-year NSE/KGE/peak table, scatter, flow-duration curve and seasonal regime |
| **Flood thresholds** | educational 2-, 5- and 10-year return levels estimated from the annual maxima of 1981-2004 (the Flood Hub "warning / danger / extreme" idea), in mm/day and m³/s, drawn on the test-decade hydrograph, with days-above / hits / misses / false alarms per model |
| **Methods & glossary** | the exact inputs, target, split, loss and model sizes, plus a plain-English glossary |

The app only reads `data/` and `outputs/`; it never trains. Port 8502 is used in the examples because 8501 is
Streamlit's default and is often already taken by another app.

## 11. Results of version 1

<!-- RESULTS_START -->
Six CAMELS-US basins, identical inputs / target / split / epochs for both models, test decade
2004-10-01 to 2014-09-30, one seed, Apple M1 Max (MPS). Full tables: `outputs/metrics/comparison.md`.

| Model | Median NSE | Median KGE | Median r | Median α | Median β | Median FHV % | Params | Train time | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LSTM (64 cells) | **0.796** | **0.730** | 0.903 | 0.843 | 1.011 | -9.7 | 22 849 | 4.9 min | smooth hydrographs, good timing; over-predicts low flows in the dry Texas basin |
| Transformer (2 layers, 4 heads, d=64) | 0.451 | 0.641 | 0.701 | 0.772 | 1.032 | -21.2 | 67 809 | 32.7 min | noisy day-to-day, weaker timing; volume bias small; validation loss still falling at epoch 30 |

Per basin (NSE / KGE):

| Basin | River | LSTM | Transformer |
|---|---|---:|---:|
| 01022500 | Narraguagus, ME (snow + rain) | 0.837 / 0.827 | 0.357 / 0.513 |
| 02177000 | Chattooga, GA (wet mountains) | 0.843 / 0.848 | 0.510 / 0.644 |
| 07057500 | North Fork, MO (karst) | 0.732 / 0.648 | 0.392 / 0.504 |
| 08171300 | Blanco, TX (flashy, semi-arid) | 0.659 / 0.289 | 0.340 / 0.650 |
| 11532500 | Smith, CA (huge winter storms) | 0.754 / 0.671 | 0.593 / 0.638 |
| 13235000 | SF Payette, ID (snowmelt) | 0.891 / 0.789 | 0.643 / 0.706 |

The biggest observed flood of the test decade in each basin, and what each model said that day (mm/day):

| Basin | Date | Rain that day | Rain previous 3 days | Observed | LSTM | Transformer |
|---|---|---:|---:|---:|---:|---:|
| Narraguagus | 2007-04-18 | 16.7 | 79.7 | 26.5 | 20.1 | 8.0 |
| Chattooga | 2009-09-20 | 78.3 | 53.4 | 32.3 | 10.8 | 11.5 |
| North Fork | 2011-04-26 | 64.4 | 164.2 | 68.2 | 30.9 | 14.8 |
| Blanco | 2013-10-31 | 87.2 | 5.2 | 38.3 | 16.3 | 17.4 |
| Smith | 2008-12-28 | 90.3 | 67.7 | 118.4 | 36.2 | 34.2 |
| SF Payette | 2010-06-05 | 3.6 | 75.8 | 16.2 | 5.7 | 7.9 |

![compare](outputs/figures/compare_hydrograph_01022500_wy2011.png)
![nse](outputs/figures/compare_nse.png)

**Reading the result honestly**

* Under an identical, small training budget the **LSTM is clearly better** on every basin by NSE, and on
  five of six by KGE. This matches the literature (Liu et al. 2025: LSTMs remain best for daily streamflow
  regression; Transformers only catch up with autoregressive inputs or much more data).
* It does **not** prove the Transformer cannot work here. Its validation loss was still decreasing at epoch 30
  (0.0278 vs 0.0169 for the LSTM), it has three times as many parameters to fit from ~41 000 samples, and
  nothing was tuned. A fair research comparison needs more epochs, a learning-rate search, several seeds and
  many more basins. The point of version 1 is that the pipeline for that comparison now exists and is fair.
* **Both models under-predict the largest floods** (Smith River: 118 mm/day observed, 36 and 34 predicted).
  Peak-flow bias (FHV) is negative almost everywhere. Extremes are rare in the training data, and the
  NSE loss has no special incentive for them: this is the central open problem for flood applications.
* The LSTM **over-predicts low flows in the dry Blanco basin** (PBIAS +68 %, KGE 0.29): it never produces the
  zero flows that occur on 22 % of days. The Transformer's volume there is right (PBIAS +2 %) but its timing is
  poor. Dry basins and zero-flow days are a known weakness of regression-trained networks.
* Visually, the Transformer hydrograph jitters from day to day while the LSTM is smooth: a recurrent state
  behaves like a water store, attention over 365 separate days does not.

Reproduce: `python scripts/train.py --model both && python scripts/evaluate.py && python scripts/compare.py`.
<!-- RESULTS_END -->

## 12. What to look at first

1. `LEARNING_NOTES.md`, sections 1 to 8, with `outputs/figures/data_01022500_wy2010.png` open next to it.
2. `python scripts/inspect_data.py` and read the printed tables for one basin.
3. `configs/lstm.yml`: every line is commented; this *is* the model definition.
4. `outputs/metrics/comparison.md` and the `compare_hydrograph_*_wy2011.png` figures.
5. `src/nhplay/evaluation/metrics.py` to see exactly what NSE and KGE compute.
6. Then the notebook and the app (start with the Results overview tab).

## 13. How the experiment avoids fooling itself

* **Chronological split**: train 1980-1999, validation 1999-2004, test 2004-2014. No test-period data is
  used for anything until the final evaluation.
* **No target leakage**: observed flow is never an input; attributes derived from flow (`q_mean`,
  `runoff_ratio`) are excluded; normalisation statistics come from the training period only.
* **Same protocol for both models**: enforced by `scripts/check_configs.py`.
* **Fixed seed** (`seed: 42`). Results are deterministic on the same hardware; on another GPU/CPU expect
  differences in the second decimal.
* **No pretrained weights** from Google are used (see below), so every number here is genuinely out of sample.

## 14. Limitations (please read before showing anyone a number)

* **Six basins, one seed, 30 epochs, no hyper-parameter search.** This is a learning setup. A research
  claim about LSTM vs Transformer needs hundreds of basins, several seeds and tuned models.
* **Daily, observed weather.** The models *simulate* today's flow from today's and past weather. A real
  forecast needs weather *forecasts* as inputs and a multi-day horizon.
* **Daily means hide flood peaks.** Hourly peaks that cause damage can be far above the daily mean.
* **Dry basins are hard.** On the Blanco River (TX) the LSTM never predicts the zero flows that occur on
  22 % of days; the flow-duration curve `outputs/figures/fdc_lstm_08171300_test.png` shows it clearly.
* **Extreme floods are underestimated.** Both models saturate around 35 to 40 mm/day on the Smith River.
* **Flood thresholds in the app are a demonstration**: empirical return levels from 24 annual maxima, not official flood stages.
* Static attributes are from CAMELS v2.0 (2017) and the forcing ends in 2014; the data are historical.

## 15. Relation to Google Flood Hub and OpenHydroNet

```
Our learning project                        Google Flood Hub
---------------------                       ----------------------------------------------
rainfall + basin data (6 CAMELS basins)     global forecast meteorology (ECMWF HRES, GraphCast,
         |                                  IMERG, CPC, ERA5-Land) + HydroATLAS attributes
LSTM / Transformer                                   |
         |                                  global hydrology LSTM (~16 000 gauges, 512 cells,
streamflow prediction (today)               probabilistic CMAL head, 7-day horizon)
         |                                           |
NSE / KGE on a held-out decade              river forecasts -> return-period thresholds
                                                     |
                                            inundation modelling (satellite SAR, DEMs)
                                                     |
                                            public flood maps, alerts in Search / Maps / Android,
                                            150+ countries
```

Google open-sourced its framework in June 2026 as **OpenHydroNet**
([github.com/google-research/flood-forecasting](https://github.com/google-research/flood-forecasting),
Apache-2.0). It is a fork of NeuralHydrology extended for forecast sequences, so everything in this project
(configs, dataset conventions, metrics) transfers directly. It ships one pretrained run
(`google-floodhub-settings-110-epochs`) and an 8-basin CAMELS toy dataset in Caravan format.

**Scientific caution we follow.** The pretrained OpenHydroNet weights were trained on the *full*
1982-2023 record with *no temporal hold-out*. The repository's `pretrained-models/Pretrained-Models-README.md`
states: "you cannot use these weights to evaluate temporal forecasting performance on historical datasets.
Any standard evaluation of this model on the 1982-2023 period will result in fundamentally invalid,
artificially inflated performance metrics due to in-sample evaluation (data leakage)." Legitimate uses are
fine-tuning, evaluation on *spatially* held-out basins, or inference on data after 2023. This project
therefore trains from scratch and does not use those weights. If you later add them, label the use as
demonstration / fine-tuning / spatial generalisation / post-2023 inference, never as historical skill.

Key references: Nearing et al. (2024, *Nature*) for the v1 system; Cohen et al. (2026, EGUsphere preprint
2026-2283) for v2; Gauch et al. (2025, *HESS*) for the current "masked-mean embedding" input aggregation.

## 16. Possible research extensions (not implemented, by design)

* **Transformer variants**: deeper encoders, `concatenate` positional encoding, patching, or the
  recurrence-free variants of Liu et al. (2024, 2025). Report several seeds.
* **Multi-basin scaling**: 50, 200, 531 CAMELS basins (`data/basins.txt` + a few hours on MPS).
* **Ungauged prediction (PUB)**: train on five basins, test on the sixth; then k-fold over many basins.
* **Extreme-event performance**: return-period thresholds from the training record, hit/miss/false-alarm
  statistics, FHV per basin, peak timing.
* **Probabilistic forecasting**: NeuralHydrology's `cmal` / `gmm` heads give full predictive distributions.
* **Forecast mode**: use weather forecasts as inputs (OpenHydroNet's `mean_embedding_forecast_lstm`).
* **Pakistan river basins**: Caravan-style data from GRDC/WAPDA gauges + ERA5-Land, then transfer
  learning from a CAMELS or global model and local fine-tuning.
* **Transfer learning from OpenHydroNet weights**: fine-tune on local gauges with post-2023 or spatially
  held-out evaluation only.
* **Inundation**: couple predicted flows to a DEM-based or HAND-based flood-extent model.

## 17. Troubleshooting

* *MPS errors or slowness*: `python scripts/train.py --model lstm --device cpu`.
* *`git describe` error from NeuralHydrology at start-up*: NeuralHydrology records the git commit of the
  folder it is installed in; because `.venv/` lives inside this repository that is this project's commit.
  It only fails if the repository has no commits yet; commit once (or `git init` elsewhere) and re-run.
* *pandas 3 warnings*: harmless; the loaders were checked against pandas 3.0.
* *Re-downloading*: `python scripts/download_data.py --force`.

## 18. References

**Data**
* Newman, A. J. et al. (2015). Development of a large-sample watershed-scale hydrometeorological data set
  for the contiguous USA. *HESS* 19, 209-223. doi:10.5194/hess-19-209-2015
* Addor, N., Newman, A. J., Mizukami, N., Clark, M. P. (2017). The CAMELS data set. *HESS* 21, 5293-5313.
  doi:10.5194/hess-21-5293-2017
* Newman, Sampson, Clark, Bock, Viger, Blodgett (2014). CAMELS data set, UCAR/NCAR. doi:10.5065/D6MW2F4D
* Kratzert, F. et al. (2023). Caravan - a global community dataset for large-sample hydrology.
  *Scientific Data* 10, 61.

**Methods**
* Hochreiter, S., Schmidhuber, J. (1997). Long short-term memory. *Neural Computation* 9, 1735-1780.
* Vaswani, A. et al. (2017). Attention is all you need. *NeurIPS*.
* Kratzert, F. et al. (2018). Rainfall-runoff modelling using LSTM networks. *HESS* 22, 6005-6022.
* Kratzert, F. et al. (2019). Towards learning universal, regional, and local hydrological behaviors via
  machine learning applied to large-sample datasets. *HESS* 23, 5089-5110.
* Kratzert, F. et al. (2022). NeuralHydrology - a Python library for deep learning research in hydrology.
  *JOSS* 7(71), 4050. https://neuralhydrology.readthedocs.io
* Nash, J. E., Sutcliffe, J. V. (1970). River flow forecasting through conceptual models. *J. Hydrol.* 10, 282-290.
* Gupta, H. V. et al. (2009). Decomposition of the mean squared error and NSE performance criteria. *J. Hydrol.* 377, 80-91.
* Knoben, W. J. M. et al. (2019). Inherent benchmark or not? Comparing NSE and KGE. *HESS* 23, 4323-4331.
* Liu, J. et al. (2025). A systematic comparison of Transformer architectures and LSTMs for rainfall-runoff
  modelling. *HESS* 29, 6811. doi:10.5194/hess-29-6811-2025
* Gauch, M. et al. (2025). Input aggregation for multi-source forcing in global LSTMs. *HESS* 29, 6221.
  doi:10.5194/hess-29-6221-2025

**Google Flood Hub**
* Nearing, G. et al. (2024). Global prediction of extreme floods in ungauged watersheds. *Nature* 627,
  559-563. doi:10.1038/s41586-024-07145-1
* Cohen, et al. (2026). Extending medium-range global flood forecasts: the Google Global Flood Forecasting
  Model v2. EGUsphere preprint egusphere-2026-2283.
* OpenHydroNet repository and docs: https://github.com/google-research/flood-forecasting ,
  https://openhydronet.readthedocs.io ; Flood Hub: https://sites.research.google/floods

## Licence

Code: MIT (see `LICENSE`). Data: CAMELS-US, CC-BY-4.0, cite Newman et al. (2015) and Addor et al. (2017).
