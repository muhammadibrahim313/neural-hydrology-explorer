# Learning notes: neural hydrology in plain English, with numbers from our own data

Every example below is taken from the six CAMELS-US basins in this project (`data/basins.txt`).
You can reproduce each number with `python scripts/inspect_data.py` or in `notebooks/exploration.ipynb`.

---

## 1. What is a basin (catchment)?

A **basin** is all the land from which rain and melting snow drain to one point on a river.
Draw a dot on a river; every raindrop that would eventually flow past that dot fell inside the basin
of that dot. Ridges and hilltops form the boundary (the "divide").

Our six basins:

| Gauge id | River | Area km² | Mean elevation m | Forest | Aridity PET/P | Snow fraction |
|---|---|---:|---:|---:|---:|---:|
| 01022500 | Narraguagus River, Maine | 574 | 93 | 0.92 | 0.59 | 0.25 |
| 02177000 | Chattooga River, Georgia | 527 | 706 | 0.97 | 0.52 | 0.03 |
| 07057500 | North Fork River, Missouri | 1456 | 325 | 0.58 | 0.85 | 0.06 |
| 08171300 | Blanco River, Texas | 1067 | 379 | 0.06 | 1.83 | 0.00 |
| 11532500 | Smith River, California | 1578 | 725 | 1.00 | 0.42 | 0.05 |
| 13235000 | South Fork Payette River, Idaho | 1163 | 2075 | 0.86 | 1.05 | 0.68 |

So "basin 08171300" means: the 1067 km² of Texas Hill Country that drains to the USGS gauge on the
Blanco River near Kyle. It is almost treeless (forest 0.06), evaporation demand is nearly twice the
rainfall (aridity 1.83), and it never snows. Basin 13235000 is the opposite: a high mountain basin where
68 % of the precipitation falls as snow.

## 2. What is a gauge?

A **gauge** (gauging station) is the instrument at the basin outlet. It measures the water level
("stage") every 15 minutes; a stage-discharge curve, calibrated by people wading into the river with
current meters, turns stage into **discharge** (volume per second). The daily mean is what we get.
In the file names, `01022500` is the USGS gauge number; the basin is named after its gauge.

Raw gauge file, first days of October 2010, Narraguagus River (`usgs_streamflow/01/01022500_streamflow_qc.txt`):

```
01022500 2010 10 01   77.00 A
01022500 2010 10 02  315.00 A
01022500 2010 10 03  371.00 A
01022500 2010 10 04  261.00 A
```

Values are in **cubic feet per second (cfs)**; `A` means "approved" by USGS; `-999 M` means missing.

## 3. What is streamflow / discharge, and why do we write it in mm/day?

**Streamflow** (= discharge, = runoff at the gauge, symbol Q) is the volume of water passing the gauge
per unit time. 371 cfs is about 10.5 m³ of water every second.

Hydrologists like to divide it by the basin area, turning a volume into a **depth of water spread over
the whole basin**:

```
Q[mm/day] = Q[cfs] × 0.0283168 m³/ft³ × 86 400 s/day ÷ area[m²] × 1000 mm/m

371 cfs × 0.0283168 × 86400 / 587 676 000 m² × 1000 = 1.54 mm/day
```

Now rainfall (mm/day) and streamflow (mm/day) are in the same unit, which makes the whole subject
easier to think about:

| Basin | Mean rain mm/day | Mean flow mm/day | Runoff ratio |
|---|---:|---:|---:|
| Smith River, CA | 5.67 | 5.50 | 0.97 |
| Narraguagus, ME | 3.62 | 2.07 | 0.57 |
| Chattooga, GA | 5.29 | 2.77 | 0.52 |
| Payette, ID | 2.67 | 1.74 | 0.65 |
| North Fork, MO | 3.19 | 1.28 | 0.40 |
| Blanco, TX | 2.46 | 0.36 | 0.15 |

In the wet, cool Smith River basin almost every millimetre of rain ends up in the river. In the hot
Blanco River basin 85 % evaporates or soaks away, and the river is dry (less than 0.01 mm/day) on
2 806 of the 12 784 days in the record. Same physics, very different behaviour; that is why the model
needs to know *which* basin it is looking at (section 5).

NeuralHydrology does this conversion for us; the column it creates is called `QObs(mm/d)`.

## 4. What is meteorological forcing?

**Forcing** is the weather that "forces" (drives) the basin: it is the *input* side of the water balance.
Ours comes from Daymet, a 1 km gridded product built from weather stations, averaged over each basin for
every day from 1980 to 2014. Each basin's forcing file (`basin_mean_forcing/daymet/...`) looks like this:

```
Year Mnth Day Hr dayl(s)  prcp(mm/day) srad(W/m2) swe(mm) tmax(C) tmin(C) vp(Pa)
2010 10   02  12 41472.00 26.42        129.47     0.00    19.10   9.67    1203.42
```

We feed five of these columns to the model as **dynamic inputs** (one value per basin per day):

| Column | Meaning | Why the river cares |
|---|---|---|
| `prcp(mm/day)` | precipitation | the water supply |
| `tmax(C)`, `tmin(C)` | daily max / min temperature | rain or snow? melting? how much evaporation? |
| `srad(W/m2)` | solar radiation | energy for evaporation and snow melt |
| `vp(Pa)` | vapour pressure (humidity) | dry air evaporates more |

`dayl(s)` is a deterministic function of latitude and date, and `swe(mm)` is zero in every one of our
files, so neither is used.

## 5. What are static basin attributes?

Numbers that describe the basin and **do not change in time**: area, elevation, slope, forest cover,
soil depth and texture, and long-term climate statistics. They come from `camels_attributes_v2.0/*.txt`.
We use 18 of them (listed in `configs/_shared_data_and_split.yml`). Narraguagus vs Blanco:

| Attribute | Narraguagus (ME) | Blanco (TX) | Meaning |
|---|---:|---:|---|
| `area_gages2` | 573.6 | 1067.5 | km² |
| `elev_mean` | 92.7 | 379.2 | m |
| `slope_mean` | 19.2 | 36.0 | m/km |
| `frac_forest` | 0.92 | 0.06 | forest fraction |
| `soil_depth_pelletier` | 4.2 | 1.7 | m to bedrock |
| `aridity` | 0.59 | 1.83 | PET / P |
| `frac_snow` | 0.25 | 0.00 | fraction of precipitation falling as snow |

Why give them to a neural network? In a **multi-basin model** one network is trained on all six basins
at once. Static attributes are how the network knows that 20 mm of rain on the Blanco River means
almost nothing in summer, while 20 mm on the Smith River means a flood tomorrow. (Kratzert et al. 2019
showed that one LSTM trained on 531 basins with attributes beats one LSTM per basin.)

Two attributes we deliberately **do not** use as inputs: `q_mean` and `runoff_ratio`. They are computed
from the observed streamflow, i.e. from the thing we are trying to predict, so using them would leak the
answer into the inputs.

## 6. What is a hydrograph?

A **hydrograph** is streamflow plotted against time. Hydrologists usually draw rainfall as bars hanging
from the top of the plot and streamflow as a line below, so you can see the cause and the response.
Look at `outputs/figures/data_01022500_wy2010.png`.

The same information as numbers, Narraguagus River, October 2010:

```
date        rain mm/day   observed flow mm/day
2010-10-01   8.86          0.32   <- river low after a dry September
2010-10-02  26.42          1.31   <- 26 mm storm: river starts rising the same day
2010-10-03   0.00          1.54   <- peak one day AFTER the rain (water needs time to travel)
2010-10-04   0.00          1.09   <- "recession": the river slowly drains
2010-10-05   0.00          0.82
2010-10-06   0.00          0.66
```

Three classic hydrograph features are visible: the **rising limb**, the **peak** (lagged by a day), and
the **recession**. Note also that of 26 mm of rain only about 1.2 mm extra left as river flow on the peak
day: the rest wet the soil, filled ponds and groundwater, or will trickle out over the following weeks.

## 7. What exactly is the model predicting?

For every basin and every day *t*, the model receives:

* the **dynamic inputs** for the 365 days ending at *t* (a 365 × 5 table of weather),
* the **18 static attributes** of that basin (repeated on every one of the 365 rows),

and outputs **one number: the streamflow on day *t* in mm/day**. That is it. It is a "rainfall-runoff
model": weather in, river out. Inside NeuralHydrology every input column is standardised (mean 0,
standard deviation 1 over the training period, pooled over basins) before it enters the network, and the
prediction is de-standardised on the way out.

One training example is therefore a tensor of shape `[365 days, 5 + 18 = 23 features]` with a single
target value. A batch holds 256 such examples drawn from random basins and random days of the training
period (1 Oct 1980 to 30 Sep 1999). Six basins × 19 years gives about 41 000 training examples.

Important: the model is **not** given yesterday's observed flow. It must work out the state of the basin
(how wet the soil is, how much snow is lying around) purely from the weather history. That is harder,
but it is what you need for a forecast in a place without a gauge.

## 8. Why can previous rainfall affect future river flow?

Because the basin **stores** water and releases it slowly:

* **Soil moisture.** Rain on dry soil mostly soaks in. Rain on soil that is already wet runs off. The
  Blanco River shows this dramatically: on 1998-10-17, 152 mm fell on a dry basin (flow the day before:
  0.11 mm/day) and the river jumped to 59.6 mm/day; on 2002-07-05, "only" 63 mm fell, but 216 mm had
  fallen in the previous three days, the ground was saturated, and the river reached 46 mm/day.
* **Snowpack.** In the Payette basin, December 2010 had the wettest weather of the year (6.0 mm/day of
  precipitation) yet the river stayed at 0.7 mm/day because it all fell as snow at -0.5 °C. The water
  came out in May and June (4.9 and 8.0 mm/day) when the temperature rose to 12 to 16 °C and there was
  little rain. The "cause" of the June flood was five months earlier.
* **Groundwater and karst.** The North Fork River (Ozarks) flows through limestone full of springs; its
  recessions take weeks.
* **Wet spells.** On the Smith River, four consecutive days of 52, 69, 52 and 67 mm of rain
  (29 Nov to 2 Dec 2012) produced flows of 24, 53, 52 and 79 mm/day: the fourth day of rain gave the
  biggest flow although it was not the biggest rain.

So today's flow depends on a long history of weather, with time scales from one day (travel time)
through weeks (soil, groundwater) to months (snow). This is why we show the model 365 days of inputs.

## 9. Why are sequence models useful here?

A model that only sees today's weather cannot distinguish the two Blanco River days above (152 mm on a
dry basin vs 63 mm on a saturated one), nor can it predict a June snowmelt flood on a sunny day with no
rain. The state of the basin has to be *inferred from the sequence* of past inputs. Sequence models
(recurrent networks, Transformers) are exactly the tools that read a sequence and build an internal
summary of it.

A classical hydrological model does the same thing with hand-written equations for soil buckets and
snow stores. The neural approach learns the storage behaviour from data instead.

## 10. Why might an LSTM work?

An **LSTM** (Long Short-Term Memory network) walks through the 365 days one day at a time and carries
a vector of memory cells along. Each day, gates decide how much of each cell to *forget*, how much new
information to *write*, and what to *output*. That is strikingly similar to a water-balance model:

| LSTM | bucket model |
|---|---|
| memory cell value | water stored in soil / snow / groundwater |
| input gate | how much of today's rain enters the store |
| forget gate | how fast the store drains or evaporates |
| output gate | how much of the stored water shows up as river flow |

Because the cells are updated by the same rule every day, an LSTM is good at smooth, slowly varying
quantities such as soil moisture and snowpack, and the "forget bias = 3" in our config tells it to start
by remembering (useful for slow stores). Our LSTM has 64 memory cells (`hidden_size: 64`); the real
Google Flood Hub model is the same idea with 512 cells trained on thousands of basins.

## 11. Why might a Transformer work?

A **Transformer** does not walk through the days in order. It looks at all 365 days at once and, for each
day, uses **attention** to decide which other days are relevant: "to predict today, look hard at the
three days of heavy rain last week and at the warm spell that melted the snow". Position information is
added with a "positional encoding" so the model knows which day is which, and a causal mask stops it from
peeking at days after *t*.

Potential advantages: direct access to events far in the past without having to carry them through 365
update steps; parallel computation. Potential disadvantages for daily hydrology: there is no built-in
notion of a slowly evolving storage, so the model has to re-derive "how wet is the soil" from scratch via
attention; it has more ways to over-fit on a small dataset; and it needs more compute (our Transformer
epoch took about 65 s against 10 s for the LSTM on an M1 Max). The literature so far (e.g. Liu et al.
2025, HESS, 11 Transformer variants vs LSTM on CAMELS and 3 434 global basins) finds that LSTMs remain
at least as good for day-ahead streamflow regression, so a Transformer win is not to be assumed.
We run the experiment and let the data decide (section 15).

## 12. What is NSE?

The **Nash-Sutcliffe Efficiency** compares the model's squared errors with the squared errors you would
get by predicting the long-term mean flow every day:

```
NSE = 1 − Σ (sim − obs)² / Σ (obs − mean(obs))²
```

* NSE = 1: perfect. NSE = 0: no better than the mean. NSE < 0: worse than the mean.
* Rough reading for daily streamflow: > 0.8 very good, 0.7 to 0.8 good, 0.5 to 0.7 acceptable.
* Because errors are squared, NSE is dominated by the big flows: it is mostly a **flood metric**.

Tiny worked example with three days of the Narraguagus storm above, pretending a model predicted
1.0, 1.4 and 1.2 mm/day for 2, 3 and 4 October 2010 (observed 1.31, 1.54, 1.09; mean 1.31):

```
Σ(sim−obs)² = (1.0−1.31)² + (1.4−1.54)² + (1.2−1.09)² = 0.096 + 0.020 + 0.012 = 0.128
Σ(obs−mean)² = 0 + 0.053 + 0.048 = 0.101
NSE = 1 − 0.128/0.101 = −0.27        (worse than guessing the mean of these three days!)
```

Over a full ten-year test period the mean is a very poor predictor, so a reasonable model gets positive
values; the code is in `src/nhplay/evaluation/metrics.py`. NeuralHydrology also uses a per-basin
*normalised* version of this as the **training loss** (`loss: NSE`), so that a river with big flows does
not dominate a river with small flows.

## 13. What is KGE?

The **Kling-Gupta Efficiency** splits model quality into three pieces that each mean something:

```
KGE = 1 − √( (r − 1)² + (α − 1)² + (β − 1)² )

r = correlation(obs, sim)       timing and shape: do peaks happen on the right days?
α = std(sim) / std(obs)         variability: are the simulated peaks as spiky as the real ones?
β = mean(sim) / mean(obs)       bias: does the model produce the right amount of water overall?
```

KGE = 1 is perfect. A model that predicts the mean flow every day gets KGE = 1 − √2 ≈ −0.41, so
*KGE > −0.41* is the "better than the mean" line (Knoben et al. 2019), not KGE > 0.
When a model has a good NSE but a poor KGE, look at α and β: typically peaks are too flat (α < 1) or the
total volume is biased (β ≠ 1). `outputs/metrics/*_metrics.csv` reports r, α and β next to KGE.

Other numbers you will see in the outputs: **RMSE** and **MAE** in mm/day (plain error sizes), **PBIAS**
(percent volume bias), **FHV** (percent bias of the top 2 % flows: negative = floods underestimated).

## 14. How the experiment is set up (and how we avoid cheating)

* **Same everything.** `scripts/check_configs.py` asserts that `configs/lstm.yml` and
  `configs/transformer.yml` agree on basins, inputs, target, split, epochs, batch size and sequence
  length. Only the model block differs.
* **Chronological split.** Train 1 Oct 1980 to 30 Sep 1999, validation 1 Oct 1999 to 30 Sep 2004,
  test 1 Oct 2004 to 30 Sep 2014. The test decade is strictly after everything the model has seen.
  The validation period is only used to watch for over-fitting; nothing is tuned on the test period.
* **No target leakage.** Observed flow is never an input, and attributes derived from flow (`q_mean`,
  `runoff_ratio`) are excluded. The input window of the first test days reaches back into 2004
  *weather* (which is public information), never into observed *flow*.
* **Normalisation statistics** are computed on the training period only (NeuralHydrology does this).
* **Fixed seed** (`seed: 42`) so that a re-run gives the same result on the same hardware.

Results for this run are in `outputs/metrics/comparison.md` and summarised in the README.

## 15. What happened when we ran it

Same six basins, same inputs, same decade. Median over basins on the test period 2004-2014:

| Model | NSE | KGE | r | α | β | FHV % | parameters | training time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LSTM | 0.796 | 0.730 | 0.903 | 0.843 | 1.011 | -9.7 | 22 849 | 4.9 min |
| Transformer | 0.451 | 0.641 | 0.701 | 0.772 | 1.032 | -21.2 | 67 809 | 32.7 min |

How to read the row for the LSTM in words: on a typical basin the model explains about 80 % of the day-to-day
variance of the river (NSE 0.80); the timing is good (r 0.90); the simulated peaks are about 16 % too flat
(α 0.84); the total volume is right to 1 % (β 1.01); the highest 2 % of flows are 10 % too low (FHV -9.7 %).

Concrete days, from `outputs/metrics/comparison_peaks.csv`:

```
Narraguagus River, 18 Apr 2007 (snowmelt + 80 mm of rain in the previous three days)
    observed 26.5 mm/day     LSTM 20.1     Transformer 8.0
Smith River, 28 Dec 2008 (90 mm of rain that day, 68 mm in the three days before)
    observed 118.4 mm/day    LSTM 36.2     Transformer 34.2
Blanco River, 31 Oct 2013 (87 mm of rain on a dry basin)
    observed 38.3 mm/day     LSTM 16.3     Transformer 17.4
```

Lessons that generalise beyond this run:

1. **A recurrent state is a good inductive bias for hydrology.** The LSTM's memory cells behave like water
   stores, so its hydrographs are smooth and its recessions realistic. The Transformer, which has to rebuild
   the basin state from attention over 365 separate days every time, produced jittery hydrographs and worse
   timing (r 0.70 vs 0.90). Published large-sample comparisons find the same.
2. **A fair comparison is a budget comparison.** The Transformer had three times the parameters, seven times
   the compute, and was still improving at epoch 30, so this result says "LSTM wins at this budget", not
   "Transformers cannot work". The configs are there to test longer training, learning-rate schedules and
   seeds; just keep the test decade untouched.
3. **Floods are the hard part.** Both models missed the biggest event of the decade in every basin by a factor
   of two to three. The training loss treats every day equally and the biggest floods are a handful of days
   in 19 years. Return-period-aware evaluation, probabilistic heads (CMAL) and more basins are the standard
   responses; see the README's research extensions.
4. **Dry rivers are hard too.** The LSTM never predicts the zero flows that the Blanco River shows on 22 % of
   days, which wrecks its KGE there (0.29) even though its NSE (0.66) looks acceptable. Always look at more
   than one metric and at the hydrograph itself.

## 16. From streamflow to flood risk

A river "floods" when its flow exceeds what the channel can carry. Operational systems do not use a
single number; they compare the forecast to **return-period thresholds**: Google Flood Hub, for example,
marks the flow expected once every 2 years as "warning", once in 5 years as "danger" and once in 20 years
as "extreme", or uses official agency levels where they exist.

The Streamlit app includes an *educational* version: a slider picks a percentile of the observed flow
record (default 99 %) and the app counts how many test-period days the observation and each model exceed
it. This is a demonstration of the idea, not a real flood-stage threshold. CAMELS has no official flood
stages, and a daily-mean model has no information about the hourly peak that actually overtops a bank.

## 17. Where this sits relative to Google Flood Hub

```
This project                           Google Flood Hub (operational)
----------------------------           ---------------------------------------------
6 basins, USA, 1980-2014               ~16 000 gauges worldwide, 1982-2023+
Daymet observed weather, daily         ECMWF HRES + GraphCast forecasts, IMERG, CPC, ERA5-Land
18 CAMELS attributes                   ~80 HydroATLAS attributes + climate indices
LSTM (64 cells) / Transformer          forecast LSTM (512 cells), probabilistic CMAL head
predicts today's flow                  predicts 7 days ahead, with uncertainty
NSE / KGE on a held-out decade         return-period thresholds -> inundation maps -> alerts
runs in 10-30 min on a laptop          continental data pipelines, retrained on everything
```

Google open-sourced that framework in 2026 as **OpenHydroNet** (github.com/google-research/flood-forecasting),
a fork of NeuralHydrology. Its pretrained weights were trained on **all** historical data (1982-2023) with
no temporal hold-out, and the repository warns that scoring them on historical dates is invalid
(in-sample). Legitimate uses are fine-tuning, spatially held-out basins, or truly post-2023 inference.
This project does not use those weights; it trains everything from scratch so every number is an honest
out-of-sample result. See README section "Relation to Google Flood Hub".

---

### Further reading (the three papers that define this field)

1. Kratzert, F. et al. (2018). Rainfall-runoff modelling using LSTM networks. *HESS* 22, 6005-6022.
2. Kratzert, F. et al. (2019). Towards learning universal, regional, and local hydrological behaviors via
   machine learning applied to large-sample datasets. *HESS* 23, 5089-5110.
3. Nearing, G. et al. (2024). Global prediction of extreme floods in ungauged watersheds. *Nature* 627, 559-563.
