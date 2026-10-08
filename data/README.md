# Data

Everything under `data/camels_us/` is **downloaded, never committed** (see `.gitignore`).
Run `python scripts/download_data.py` once (about 10 MB, 1–2 minutes) to recreate it.

## What dataset is this?

**CAMELS-US** (Catchment Attributes and MEteorology for Large-sample Studies), 671 small-to-medium
river basins in the contiguous USA with little human influence on streamflow.

| Part | What it contains | Source paper |
|---|---|---|
| Basin-mean forcing (Daymet) | Daily weather averaged over each basin, 1980–2014: precipitation, min/max temperature, solar radiation, vapour pressure, day length, snow-water equivalent | Newman et al. (2015), *HESS* 19, 209–223, doi:10.5194/hess-19-209-2015 |
| USGS streamflow | Daily mean discharge at the basin outlet gauge, 1980–2014, in cubic feet per second, with quality flags; `-999` = missing | same |
| Attributes v2.0 | One row per basin: topography, climate indices, soil, geology, vegetation, hydrological signatures | Addor et al. (2017), *HESS* 21, 5293–5313, doi:10.5194/hess-21-5293-2017 |

Official archive: NCAR, DOI [10.5065/D6MW2F4D](https://doi.org/10.5065/D6MW2F4D), now hosted on Zenodo as
[record 15529996](https://zenodo.org/records/15529996). Licence **CC-BY-4.0**. The data owners ask that
publications cite the two papers above.

## How the download stays small

The official time-series archive is a single 3.4 GB zip. Zenodo supports HTTP *range requests*, so
`scripts/download_data.py` opens the zip remotely with the `remotezip` package, reads only its table of
contents, and extracts just the two text files that belong to each basin we want (about 1.2 MB per basin).
The seven attribute tables (about 0.6 MB total) are downloaded as plain files.

## The six basins used in this project (`data/basins.txt`)

| USGS gauge | River / place | Area km² | Why it is interesting |
|---|---|---:|---|
| 01022500 | Narraguagus River at Cherryfield, Maine | 574 | humid, forested, winter snow then spring melt; the NeuralHydrology tutorial basin |
| 02177000 | Chattooga River near Clayton, Georgia | 527 | wet Appalachian mountains, rain-driven, usually easy to model |
| 07057500 | North Fork River near Tecumseh, Missouri | 1456 | Ozark karst: springs and groundwater give smooth recessions |
| 08171300 | Blanco River near Kyle, Texas | 1067 | semi-arid "Flash Flood Alley": long dry spells, violent floods, hard to model |
| 11532500 | Smith River near Crescent City, California | 1578 | Pacific coast, huge winter rainstorms, very high peaks |
| 13235000 | South Fork Payette River at Lowman, Idaho | 1163 | high-elevation snowmelt regime (also in Google's OpenHydroNet tutorial set) |

You can add any other CAMELS gauge id to `data/basins.txt` and re-run the download script; the list of all
671 gauges (with names, coordinates and areas) is downloaded to `data/camels_us/basin_metadata/gauge_information.txt`.

## Folder layout after download (NeuralHydrology's `camels_us` convention)

```
data/camels_us/
├── basin_mean_forcing/daymet/<HUC2>/<gauge>_lump_cida_forcing_leap.txt   # dynamic inputs
├── usgs_streamflow/<HUC2>/<gauge>_streamflow_qc.txt                       # target (raw cfs)
├── camels_attributes_v2.0/camels_{clim,geol,hydro,name,soil,topo,vege}.txt # static attributes
├── basin_metadata/gauge_information.txt
├── readme_*.txt                                                            # the dataset's own documentation
└── MANIFEST.json                                                           # provenance: URLs, sizes, md5, dates
```

`<HUC2>` is the two-digit USGS hydrologic region (e.g. `01` = New England).

## File formats (so you can open them in a text editor)

**Forcing file** – four header lines (latitude, mean elevation in m, basin area in m², column names), then one
row per day:

```
Year Mnth Day Hr dayl(s) prcp(mm/day) srad(W/m2) swe(mm) tmax(C) tmin(C) vp(Pa)
1980 01 01 12 29937.86 0.00 139.84 0.00 -6.47 -14.83 203.98
```

**Streamflow file** – whitespace separated, no header:

```
gauge_id year month day QObs(cfs) quality_flag
01022500 1980 01 01 1200.00 A
```

Quality flags: `A` approved, `A:e` approved but estimated, `M` missing (value `-999`).

**Attribute files** – semicolon separated, one row per gauge, e.g. `camels_topo.txt`:

```
gauge_id;gauge_lat;gauge_lon;elev_mean;slope_mean;area_gages2;area_geospa_fabric
```

## Unit note you will meet everywhere

The gauge reports volume per second (cubic feet per second, cfs). NeuralHydrology converts it to a water
*depth per day over the basin* so that streamflow and rainfall share the unit mm/day:

```
Q[mm/day] = Q[cfs] * 0.0283168 [m³/ft³] * 86400 [s/day] / area[m²] * 1000 [mm/m]
```

A flow of 1200 cfs over a 573.6 km² basin is 5.1 mm/day: if that day's river volume were spread evenly over
the whole basin it would be a 5.1 mm deep layer of water. Comparing that with the rainfall in mm/day tells you
immediately what fraction of the rain is leaving as river flow.
