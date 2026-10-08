#!/usr/bin/env python
"""Assert that lstm.yml and transformer.yml use IDENTICAL data, inputs, target, split and training budget.

A fair model comparison needs everything except the model itself to be the same. This script makes
that promise checkable instead of relying on people copy-pasting carefully.
"""
from __future__ import annotations

import sys
from pathlib import Path

from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
SHARED_KEYS = [
    # data
    "dataset", "data_dir", "forcings", "dynamic_inputs", "static_attributes", "target_variables",
    "clip_targets_to_zero", "train_basin_file", "validation_basin_file", "test_basin_file",
    # split
    "train_start_date", "train_end_date", "validation_start_date", "validation_end_date",
    "test_start_date", "test_end_date",
    # training budget / protocol
    "seed", "epochs", "batch_size", "seq_length", "predict_last_n", "optimizer", "loss",
    "clip_gradient_norm", "target_noise_std", "validate_every", "metrics",
]


def main() -> int:
    yaml = YAML(typ="safe")
    cfgs = {name: yaml.load(open(ROOT / "configs" / f"{name}.yml")) for name in ("lstm", "transformer")}
    shared = yaml.load(open(ROOT / "configs" / "_shared_data_and_split.yml"))
    problems = []
    for key in SHARED_KEYS:
        values = {name: cfg.get(key) for name, cfg in cfgs.items()}
        if values["lstm"] != values["transformer"]:
            problems.append(f"{key}: lstm={values['lstm']!r} transformer={values['transformer']!r}")
        if key in shared and shared[key] != values["lstm"]:
            problems.append(f"{key}: _shared_data_and_split.yml={shared[key]!r} lstm={values['lstm']!r}")
    if problems:
        print("CONFIG MISMATCH:\n  " + "\n  ".join(problems))
        return 1
    print(f"OK: lstm.yml and transformer.yml agree on all {len(SHARED_KEYS)} shared settings "
          f"({len(cfgs['lstm']['dynamic_inputs'])} dynamic inputs, {len(cfgs['lstm']['static_attributes'])} static "
          f"attributes, target {cfgs['lstm']['target_variables']}, {cfgs['lstm']['epochs']} epochs).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
