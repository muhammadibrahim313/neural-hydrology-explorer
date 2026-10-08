#!/usr/bin/env python
"""Phase 2/3 - train the LSTM baseline and/or the Transformer with NeuralHydrology.

    python scripts/train.py --model lstm
    python scripts/train.py --model transformer
    python scripts/train.py --model both                 # one after the other
    python scripts/train.py --model lstm --device cpu    # force CPU (default: MPS if available)
    python scripts/train.py --model lstm --epochs 3      # quick smoke test

Each run creates runs/<experiment_name>_<DDMM_HHMMSS>/ (NeuralHydrology's layout: config.yml,
model_epochXXX.pt, output.log, validation/..., tensorboard events) and records the folder in
runs/latest_<model>.txt so that evaluate.py / compare.py / the app find it.
After training, the test period is evaluated once with the final epoch's weights.

Equivalent NeuralHydrology CLI (from the project root, with the venv active):
    nh-run train --config-file configs/lstm.yml
    nh-run evaluate --run-dir runs/lstm_6basins_<stamp>
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nhplay.models.run import evaluate, pick_device, train  # noqa: E402
from nhplay.paths import CONFIG_DIR, RUNS_DIR  # noqa: E402

MODELS = {"lstm": CONFIG_DIR / "lstm.yml", "transformer": CONFIG_DIR / "transformer.yml"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["lstm", "transformer", "both"], default="both")
    ap.add_argument("--device", choices=["cpu", "mps", "cuda:0"], default=None, help="default: auto")
    ap.add_argument("--epochs", type=int, default=None, help="override the number of epochs in the config")
    ap.add_argument("--no-eval", action="store_true", help="skip the test-period evaluation after training")
    args = ap.parse_args()

    # fairness pre-flight: both configs must share data, split and training budget
    subprocess.run([sys.executable, str(ROOT / "scripts" / "check_configs.py")], check=True)

    device = pick_device(args.device)
    print(f"device: {device}")
    names = ["lstm", "transformer"] if args.model == "both" else [args.model]
    for name in names:
        overrides = {"epochs": args.epochs} if args.epochs else None
        print(f"\n=== training {name} from {MODELS[name].relative_to(ROOT)} ===")
        t0 = time.time()
        run_dir = train(MODELS[name], device=device, overrides=overrides)
        print(f"=== {name} done in {(time.time() - t0) / 60:.1f} min -> {run_dir}")
        (RUNS_DIR / f"latest_{name}.txt").write_text(str(run_dir) + "\n")
        if not args.no_eval:
            evaluate(run_dir, period="test", device=device)
            print(f"=== test-period results stored under {run_dir / 'test'}")
    print("\nNext: python scripts/evaluate.py  &&  python scripts/compare.py")


if __name__ == "__main__":
    main()
