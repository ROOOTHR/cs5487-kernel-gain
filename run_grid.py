"""Stage 1: main comparison grid -- linear vs kernel classifiers x preprocessing variants.

Usage:  python run_grid.py [--models svm_rbf,logreg] [--preprocs raw,pca100]
Results are appended to results/grid.jsonl (one record per model x preproc x trial),
so the script can be interrupted and resumed.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from common import load_digits
from exp_lib import RESULTS, append_jsonl, run_config

PREPROCS = ["raw", "std", "pca20", "pca50", "pca100", "pca200",
            "pca20w", "pca50w", "pca100w", "pca200w"]
MODELS = ["knn1", "bdr_shared", "bdr_full", "bdr_full_eig", "logreg", "svm_linear", "svm_rbf"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--preprocs", default=",".join(PREPROCS))
    args = ap.parse_args()

    models = [m for m in args.models.split(",") if m]
    preprocs = [p for p in args.preprocs.split(",") if p]

    Xraw, y, trials = load_digits()
    X = Xraw / 255.0

    out = RESULTS / "grid.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            done.add((rec["model"], rec["preproc"], rec["trial"]))

    total = len(models) * len(preprocs) * len(trials)
    i = 0
    for model in models:
        for preproc in preprocs:
            for t, (tr, te) in enumerate(trials):
                i += 1
                if (model, preproc, t) in done:
                    print(f"[{i}/{total}] skip {model:11s} {preproc:8s} trial{t + 1}", flush=True)
                    continue
                rec = run_config(X, y, tr, te, model, preproc, t)
                append_jsonl(out, rec)
                if rec["test_acc"] is None:
                    print(f"[{i}/{total}] {model:13s} {preproc:8s} trial{t + 1} "
                          f"FAILED ({rec['error']})", flush=True)
                    continue
                print(
                    f"[{i}/{total}] {model:13s} {preproc:8s} trial{t + 1} "
                    f"test={rec['test_acc']:.4f} cv={rec['cv_acc']} "
                    f"({rec['fit_seconds']:.1f}s) {rec['best_params']}",
                    flush=True,
                )


if __name__ == "__main__":
    sys.exit(main())
