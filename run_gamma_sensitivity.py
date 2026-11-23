"""Stage 4: sensitivity of the RBF SVM to (C, gamma), across feature dimensions.

The heat maps are built from **5-fold cross-validation accuracy on the training set only**.
This is a deliberate constraint, not a convenience: a surface evaluated on the test set would
be a test-set tuning tool, which PA-3 forbids. Using the training-set CV score means the whole
figure can be inspected without any risk of having selected a configuration on test data.

Two things the surface is meant to answer:
  * is there a wide plateau in (C, gamma), or does the kernel only work on a knife edge?
  * does that plateau move when the feature dimension changes?

Usage: python run_gamma_sensitivity.py [--preprocs pca20,pca50,pca200,raw]
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from common import load_digits
from exp_lib import RESULTS, SEED, append_jsonl, feature_stats, preproc_steps

GAMMA_MULT = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
C_VALUES = (1.0, 10.0, 100.0, 1000.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocs", default="pca20,pca50,pca200,raw")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    preprocs = [p for p in args.preprocs.split(",") if p]
    Xraw, y, trials = load_digits()
    X = Xraw / 255.0

    out = RESULTS / "gamma_sensitivity.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done.add((rec["preproc"], rec["trial"]))

    for preproc in preprocs:
        for t, (tr, _te) in enumerate(trials):
            if (preproc, t) in done:
                print(f"skip {preproc:8s} trial{t + 1}", flush=True)
                continue
            Xtr, ytr = X[tr], y[tr]
            k, var = feature_stats(Xtr, preproc)
            g0 = 1.0 / (k * var)  # same median-heuristic scale used in the main grid
            pipe = Pipeline(preproc_steps(preproc) + [("clf", SVC(kernel="rbf", cache_size=500))])
            grid = {
                "clf__C": list(C_VALUES),
                "clf__gamma": [m * g0 for m in GAMMA_MULT],
            }
            cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
            search = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=1, refit=False)
            search.fit(Xtr, ytr)

            for params, score in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"]):
                append_jsonl(out, {
                    "preproc": preproc,
                    "n_features": int(k),
                    "trial": t,
                    "fold_count": int(args.folds),
                    "C": float(params["clf__C"]),
                    "gamma": float(params["clf__gamma"]),
                    "gamma_mult": round(float(params["clf__gamma"] / g0), 4),
                    "cv_acc": round(float(score), 4),
                })
            best = search.best_params_
            print(
                f"{preproc:8s} trial{t + 1} k={k:<4d} best CV={search.best_score_:.4f} "
                f"C={best['clf__C']:.0f} gamma={best['clf__gamma'] / g0:.2f}*g0 "
                f"(search {len(search.cv_results_['params'])} x {args.folds} folds)",
                flush=True,
            )


if __name__ == "__main__":
    sys.exit(main())
