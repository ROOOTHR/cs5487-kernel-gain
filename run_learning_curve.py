"""Stage 3: learning curves -- how does accuracy scale with the number of training images?

For each training-set size n, a stratified subsample of the 2000 training images is drawn
(fixed seed), the preprocessing is re-fitted on that subsample, and hyper-parameters are
re-tuned by 5-fold CV *inside* the subsample. Nothing here touches the test set except the
final single evaluation.

The largest size equals the full training set, so that column must reproduce the
corresponding cell of grid.jsonl / sweep.jsonl exactly -- a built-in consistency check.

Re-tuning at every n is deliberate: it lets us see whether the optimal regularisation
drifts with sample size, which a fixed-hyper-parameter curve would hide.

Usage: python run_learning_curve.py [--sizes 100,200,400,700,1000] [--preproc pca50]
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from common import load_digits
from exp_lib import RESULTS, append_jsonl, run_config

SIZES = [100, 200, 400, 500, 700, 1000, 1500, 2000]
MODELS = ["svm_rbf", "svm_linear", "logreg", "bdr_full_eig", "knn1"]
SEED = 0


def stratified_subsample(y_tr: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Indices of a class-balanced subsample of size n (n must be a multiple of 10)."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y_tr)
    per_class = n // len(classes)
    if per_class * len(classes) != n:
        raise ValueError(f"n={n} is not a multiple of {len(classes)}")
    keep = []
    for c in classes:
        idx = np.flatnonzero(y_tr == c)
        if len(idx) < per_class:
            raise ValueError(f"class {c} has only {len(idx)} training images, cannot draw {per_class}")
        keep.append(rng.choice(idx, size=per_class, replace=False))
    return np.sort(np.concatenate(keep))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default=",".join(str(s) for s in SIZES))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--preproc", default="pca50")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s]
    models = [m for m in args.models.split(",") if m]

    Xraw, y, trials = load_digits()
    X = Xraw / 255.0

    out = RESULTS / "learning_curve.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done.add((rec["model"], rec["preproc"], rec["n_train"], rec["trial"]))

    total = len(sizes) * len(models) * len(trials)
    i = 0
    for n in sizes:
        for model in models:
            for t, (tr, te) in enumerate(trials):
                i += 1
                key = (model, args.preproc, n, t)
                if key in done:
                    print(f"[{i}/{total}] skip {model:13s} n={n:<5d} trial{t + 1}", flush=True)
                    continue
                sub = tr[stratified_subsample(y[tr], n, seed=SEED + t)] if n < len(tr) else tr
                rec = run_config(X, y, sub, te, model, args.preproc, t)
                rec["n_train"] = int(n)
                append_jsonl(out, rec)
                if rec["test_acc"] is None:
                    print(f"[{i}/{total}] {model:13s} n={n:<5d} trial{t + 1} FAILED ({rec['error']})", flush=True)
                    continue
                print(
                    f"[{i}/{total}] {model:13s} n={n:<5d} trial{t + 1} "
                    f"test={rec['test_acc']:.4f} cv={rec['cv_acc']} "
                    f"({rec['fit_seconds']:.1f}s) {rec['best_params']}",
                    flush=True,
                )


if __name__ == "__main__":
    sys.exit(main())
