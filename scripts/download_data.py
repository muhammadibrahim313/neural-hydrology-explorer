#!/usr/bin/env python
"""Download a handful of CAMELS-US basins straight from the official NCAR archive on Zenodo.

Why this is small: the full archive ``basin_timeseries_v1p2_metForcing_obsFlow.zip`` is 3.4 GB,
but Zenodo supports HTTP range requests, so we open the zip *remotely* (``remotezip``) and pull out
only the ~1.2 MB that belong to each basin we want (Daymet basin-mean forcing + USGS streamflow).
The seven CAMELS attribute tables (~0.6 MB in total) are plain files on the same record.

Source (official DOI 10.5065/D6MW2F4D -> Zenodo record 15529996, licence CC-BY-4.0):
    https://zenodo.org/records/15529996

Result (the folder layout NeuralHydrology's ``camels_us`` dataset class expects):
    data/camels_us/
    ├── basin_mean_forcing/daymet/<HUC2>/<gauge>_lump_cida_forcing_leap.txt
    ├── usgs_streamflow/<HUC2>/<gauge>_streamflow_qc.txt
    ├── camels_attributes_v2.0/camels_*.txt (+ readme.txt)
    ├── basin_metadata/gauge_information.txt
    └── MANIFEST.json   (what was downloaded, from where, when, md5)

Usage:
    python scripts/download_data.py                 # basins from data/basins.txt
    python scripts/download_data.py 01022500 ...    # explicit basin ids
    python scripts/download_data.py --force         # re-download even if files exist
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nhplay.data.camels import read_basin_list  # noqa: E402
from nhplay.paths import BASIN_LIST_FILE, CAMELS_DIR  # noqa: E402

ZENODO_RECORD = "15529996"
ZENODO_FILES = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files"
TIMESERIES_ZIP = f"{ZENODO_FILES}/basin_timeseries_v1p2_metForcing_obsFlow.zip/content"
ATTRIBUTE_FILES = ["camels_clim.txt", "camels_geol.txt", "camels_hydro.txt", "camels_name.txt",
                   "camels_soil.txt", "camels_topo.txt", "camels_vege.txt", "readme.txt"]
ZIP_PREFIX = "basin_dataset_public_v1p2/"
EXTRA_MEMBERS = [  # small, useful reference files inside the zip
    "basin_metadata/gauge_information.txt",
    "readme_FIRST.txt",
    "readme_basin_mean_forcing.txt",
    "readme_streamflow.txt",
]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, force: bool = False) -> bool:
    if dest.exists() and not force:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
        tmp.replace(dest)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("basins", nargs="*", help="USGS gauge ids (default: read data/basins.txt)")
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    ap.add_argument("--data-dir", type=Path, default=CAMELS_DIR)
    args = ap.parse_args()

    basins = args.basins or read_basin_list(BASIN_LIST_FILE)
    data_dir: Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "source": f"https://zenodo.org/records/{ZENODO_RECORD}",
        "doi": "10.5065/D6MW2F4D",
        "license": "CC-BY-4.0",
        "citations": [
            "Newman et al. (2015) HESS 19, 209-223, doi:10.5194/hess-19-209-2015 (time series)",
            "Addor et al. (2017) HESS 21, 5293-5313, doi:10.5194/hess-21-5293-2017 (attributes)",
        ],
        "files": {},
    }

    def record(rel: Path, url: str):
        p = data_dir / rel
        manifest["files"][str(rel)] = {"url": url, "bytes": p.stat().st_size, "md5": md5(p),
                                       "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    # 1) attribute tables ---------------------------------------------------------------------
    print("1/3  CAMELS attribute tables")
    for name in ATTRIBUTE_FILES:
        url = f"{ZENODO_FILES}/{name}/content"
        rel = Path("camels_attributes_v2.0") / name
        if download_file(url, data_dir / rel, args.force):
            record(rel, url)
            print(f"      downloaded {rel}")
        else:
            print(f"      exists     {rel}")

    # 2) per-basin time series from the remote zip --------------------------------------------
    print("2/3  per-basin forcing + streamflow (remote zip, range requests)")
    wanted: dict[str, Path] = {}
    missing_members = []
    for extra in EXTRA_MEMBERS:
        wanted[ZIP_PREFIX + extra] = Path(extra)
    need_zip = args.force or any(not (data_dir / rel).exists() for rel in wanted.values())
    basin_members: dict[str, list[str]] = {}

    from remotezip import RemoteZip

    t0 = time.time()
    with RemoteZip(TIMESERIES_ZIP) as z:
        names = z.namelist()
        print(f"      zip directory read: {len(names)} members in {time.time() - t0:.1f}s")
        for b in basins:
            forcing = [n for n in names if n.startswith(f"{ZIP_PREFIX}basin_mean_forcing/daymet/")
                       and n.endswith(f"/{b}_lump_cida_forcing_leap.txt")]
            flow = [n for n in names if n.startswith(f"{ZIP_PREFIX}usgs_streamflow/")
                    and n.endswith(f"/{b}_streamflow_qc.txt")]
            if not forcing or not flow:
                missing_members.append(b)
                continue
            basin_members[b] = [forcing[0], flow[0]]
            for m in basin_members[b]:
                wanted[m] = Path(m[len(ZIP_PREFIX):])
        for member, rel in wanted.items():
            dest = data_dir / rel
            if dest.exists() and not args.force:
                print(f"      exists     {rel}")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            t1 = time.time()
            with z.open(member) as src, open(dest, "wb") as out:
                out.write(src.read())
            record(rel, TIMESERIES_ZIP + "#" + member)
            print(f"      extracted  {rel}  ({dest.stat().st_size / 1e6:.2f} MB, {time.time() - t1:.1f}s)")

    if missing_members:
        print(f"!!    not found in archive: {missing_members}")

    # 3) sanity check + manifest ----------------------------------------------------------------
    print("3/3  sanity check")
    from nhplay.data.camels import load_basin

    manifest["basins"] = {}
    for b in basins:
        if b in missing_members:
            continue
        bd = load_basin(b, data_dir)
        q = bd.timeseries["QObs(mm/d)"]
        info = {
            "name": str(bd.attributes.get("gauge_name", "")),
            "area_km2": round(bd.area_km2, 1),
            "first_day": str(bd.timeseries.index.min().date()),
            "last_day": str(bd.timeseries.index.max().date()),
            "n_days": int(len(bd.timeseries)),
            "n_missing_flow_days": int(q.isna().sum()),
        }
        manifest["basins"][b] = info
        print(f"      {b} {info['name'][:45]:45s} {info['area_km2']:8.1f} km2  "
              f"{info['first_day']}..{info['last_day']}  missing flow days: {info['n_missing_flow_days']}")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"done. manifest -> {manifest_path}")
    return 1 if missing_members else 0


if __name__ == "__main__":
    raise SystemExit(main())
