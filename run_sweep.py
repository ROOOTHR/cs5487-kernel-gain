"""Stage 2: dimension sweep -- where does the kernel gain disappear?

For k in KS, compare linear (svm_linear, logreg, bdr_shared) against RBF SVM on the same
k-dimensional PCA space. This is the core figure of the project: accuracy vs. dimension,
one curve per classifier, plus the "kernel gain" curve (RBF - best linear).

Usage: python run_sweep.py [--ks 10,20,...] [--whiten]
"""
from __future__ import annotations

import argparse
import json
import sys

from common import load_digits
from exp_lib import RESULTS, append_jsonl, run_config

KS = [10, 20, 30, 50, 80, 120, 200, 300, 500]
MODELS = ["svm_rbf", "svm_linear", "logreg", "bdr_shared"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default=",".join(str(k) for k in KS))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--whiten", action="store_true", help="use PCA-whitened features")
    args = ap.parse_args()

    ks = [int(k) for k in args.ks.split(",") if k]
    models = [m for m in args.models.split(",") if m]
    suffix = "w" if args.whiten else ""

    Xraw, y, trials = load_digits()
    X = Xraw / 255.0

    out = RESULTS / f"sweep{suffix or ''}.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            done.add((rec["model"], rec["preproc"], rec["trial"]))

    total = len(ks) * len(models) * len(trials)
    i = 0
    for k in ks:
        preproc = f"pca{k}{suffix}"
        for model in models:
            for t, (tr, te) in enumerate(trials):
                i += 1
                if (model, preproc, t) in done:
                    print(f"[{i}/{total}] skip {model:11s} k={k:<4d} trial{t + 1}", flush=True)
                    continue
                rec = run_config(X, y, tr, te, model, preproc, t)
                append_jsonl(out, rec)
                if rec["test_acc"] is None:
                    print(f"[{i}/{total}] {model:13s} k={k:<4d} trial{t + 1} FAILED ({rec['error']})", flush=True)
                    continue
                print(
                    f"[{i}/{total}] {model:13s} k={k:<4d} trial{t + 1} "
                    f"test={rec['test_acc']:.4f} cv={rec['cv_acc']} ({rec['fit_seconds']:.1f}s)",
                    flush=True,
                )


if __name__ == "__main__":
    sys.exit(main())
