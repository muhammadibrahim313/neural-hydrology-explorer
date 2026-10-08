"""Thin wrapper around NeuralHydrology's training / evaluation entry points.

Why a wrapper at all?  NeuralHydrology is driven by a YAML config and the ``nh-run`` CLI.
That is great for research, but for learning we want a few conveniences:

* pick the compute device automatically (Apple MPS if it works, otherwise CPU),
* know *which* run directory a training call produced,
* pull the observed / predicted series out of NeuralHydrology's result pickle into a
  plain pandas DataFrame so that the notebook and the Streamlit app can use them.

Everything model-related (LSTM, Transformer, loss, data loading, normalisation) is still
100 % NeuralHydrology — see ``configs/*.yml``.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import pandas as pd
import torch
from ruamel.yaml import YAML

from nhplay.paths import RUNS_DIR


def pick_device(prefer: str | None = None) -> str:
    """'mps' on Apple Silicon when PyTorch supports it, else 'cpu'. CUDA if someone runs this on Linux."""
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def read_config(path: Path) -> dict:
    yaml = YAML(typ="safe")
    with open(path) as f:
        return yaml.load(f)


def train(config_file: Path, device: str | None = None, overrides: dict | None = None) -> Path:
    """Train one model from a config file and return the run directory NeuralHydrology created."""
    from neuralhydrology.training.train import start_training
    from neuralhydrology.utils.config import Config

    cfg = Config(Path(config_file))
    cfg.update_config({"device": pick_device(device)})
    if overrides:
        cfg.update_config(overrides)
    run_root = Path(cfg.run_dir) if cfg.run_dir is not None else RUNS_DIR
    run_root.mkdir(parents=True, exist_ok=True)
    cfg.update_config({"run_dir": run_root})

    before = {p for p in run_root.glob(f"{cfg.experiment_name}_*") if p.is_dir()}
    t0 = time.time()
    start_training(cfg)
    minutes = (time.time() - t0) / 60
    after = {p for p in run_root.glob(f"{cfg.experiment_name}_*") if p.is_dir()}
    new = sorted(after - before, key=lambda p: p.stat().st_mtime)
    if not new:
        raise RuntimeError("Training finished but no new run directory was found.")
    run_dir = new[-1]
    (run_dir / "nhplay_train_minutes.txt").write_text(f"{minutes:.2f}\n")
    return run_dir


def evaluate(run_dir: Path, period: str = "test", epoch: int | None = None, device: str | None = None) -> Path:
    """Run NeuralHydrology's evaluation for a period and return the folder with the results pickle."""
    from neuralhydrology.evaluation.evaluate import start_evaluation
    from neuralhydrology.utils.config import Config

    run_dir = Path(run_dir)
    cfg = Config(run_dir / "config.yml")
    cfg.update_config({"device": pick_device(device)})
    start_evaluation(cfg=cfg, run_dir=run_dir, epoch=epoch, period=period)
    return results_dir(run_dir, period, epoch)


def _epoch_dir_name(run_dir: Path, epoch: int | None) -> str:
    if epoch is None:
        weights = sorted(run_dir.glob("model_epoch*.pt"))
        if not weights:
            raise FileNotFoundError(f"No model_epochXXX.pt files in {run_dir}")
        return weights[-1].stem
    return f"model_epoch{epoch:03d}"


def results_dir(run_dir: Path, period: str = "test", epoch: int | None = None) -> Path:
    run_dir = Path(run_dir)
    return run_dir / period / _epoch_dir_name(run_dir, epoch)


def load_results(run_dir: Path, period: str = "test", epoch: int | None = None,
                 target: str = "QObs(mm/d)") -> dict[str, pd.DataFrame]:
    """{basin: DataFrame[date -> obs, sim]} from NeuralHydrology's ``<period>_results.p``."""
    rdir = results_dir(run_dir, period, epoch)
    pkl = rdir / f"{period}_results.p"
    if not pkl.exists():
        raise FileNotFoundError(f"{pkl} not found. Run evaluation first.")
    with open(pkl, "rb") as f:
        results = pickle.load(f)
    out = {}
    for basin, freqs in results.items():
        freq = "1D" if "1D" in freqs else next(iter(freqs))
        xr = freqs[freq]["xr"]
        obs = xr[f"{target}_obs"].isel(time_step=-1).to_series()
        sim = xr[f"{target}_sim"].isel(time_step=-1).to_series()
        df = pd.DataFrame({"obs": obs, "sim": sim})
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        out[basin] = df
    return out


def load_nh_metrics(run_dir: Path, period: str = "test", epoch: int | None = None) -> pd.DataFrame:
    """NeuralHydrology's own metric table (``<period>_metrics.csv``), one row per basin."""
    rdir = results_dir(run_dir, period, epoch)
    csv = rdir / f"{period}_metrics.csv"
    df = pd.read_csv(csv, dtype={"basin": str})
    return df.set_index("basin")


def latest_run(experiment_name: str, run_root: Path = RUNS_DIR) -> Path | None:
    runs = sorted((p for p in Path(run_root).glob(f"{experiment_name}_*") if p.is_dir()),
                  key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def describe_run(run_dir: Path) -> dict:
    """Small summary of a run for the README / app: model type, sizes, epochs, train time."""
    cfg = read_config(Path(run_dir) / "config.yml")
    minutes_file = Path(run_dir) / "nhplay_train_minutes.txt"
    n_params = None
    try:
        from neuralhydrology.modelzoo import get_model
        from neuralhydrology.utils.config import Config

        model = get_model(Config(Path(run_dir) / "config.yml"))
        n_params = sum(p.numel() for p in model.parameters())
    except Exception:  # pragma: no cover - purely informational
        pass
    return {
        "model": cfg.get("model"),
        "hidden_size": cfg.get("hidden_size"),
        "seq_length": cfg.get("seq_length"),
        "epochs": cfg.get("epochs"),
        "batch_size": cfg.get("batch_size"),
        "learning_rate": cfg.get("learning_rate"),
        "n_parameters": n_params,
        "train_minutes": float(minutes_file.read_text()) if minutes_file.exists() else None,
        "device": cfg.get("device"),
    }
