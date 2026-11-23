"""Control group: does a convolutional net beat the kernel on this data, and at what cost?

Motivation
----------
The main study shows an RBF SVM beating the best linear model by ~4-5 pp on a fixed feature
space, which says the *shape* of the boundary matters. A CNN differs from both in a way that
no feature-space argument captures: it learns a translation-equivariant representation instead
of being handed one. So the honest question is not "is the CNN better" (with 2,000 images it
may well not be) but "how much data does the extra inductive bias need to pay off".

Protocol
--------
Same two trials, same training subsetting as run_learning_curve.py, same test evaluation.
A stratified 10% of the training images is held out for early stopping; that validation split
is carved from the training set only, so the test set stays untouched until the final score.
Several seeds are run per (trial, n) because a single CNN run is not a measurement.

Install with: pip install -r requirements-cnn.txt
Usage: python cnn_control.py [--sizes 500,1000,2000] [--seeds 1,2]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from common import load_digits
from exp_lib import RESULTS

torch.set_num_threads(4)

SIZES = [500, 1000, 2000]
SEEDS = [1, 2]
EPOCHS = 40
PATIENCE = 6
BATCH = 64
VAL_FRACTION = 0.10


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.head(self.features(x))


def stratified_split(y: np.ndarray, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    val = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        val.append(rng.choice(idx, size=max(1, int(round(len(idx) * frac))), replace=False))
    val = np.sort(np.concatenate(val))
    mask = np.ones(len(y), bool)
    mask[val] = False
    return np.flatnonzero(mask), val


def stratified_subsample(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    per_class = n // len(np.unique(y))
    keep = [rng.choice(np.flatnonzero(y == c), size=per_class, replace=False) for c in np.unique(y)]
    return np.sort(np.concatenate(keep))


def train_one(Xtr, ytr, Xte, yte, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    fit_idx, val_idx = stratified_split(ytr, VAL_FRACTION, seed)
    Xf, yf = Xtr[fit_idx], ytr[fit_idx]
    Xv, yv = Xtr[val_idx], ytr[val_idx]

    model = SmallCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    Xf_t = torch.from_numpy(Xf).view(-1, 1, 28, 28)
    yf_t = torch.from_numpy(yf).long()
    Xv_t = torch.from_numpy(Xv).view(-1, 1, 28, 28)
    Xte_t = torch.from_numpy(Xte).view(-1, 1, 28, 28)

    best_val, best_state, best_epoch, waited = -1.0, None, 0, 0
    history = []
    t0 = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(Xf_t))
        tot = 0.0
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Xf_t[b]), yf_t[b])
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
        model.eval()
        with torch.no_grad():
            val_acc = float((model(Xv_t).argmax(1) == torch.from_numpy(yv).long()).float().mean())
        history.append({"epoch": epoch, "train_loss": round(tot / len(Xf_t), 4), "val_acc": round(val_acc, 4)})
        if val_acc > best_val + 1e-6:
            best_val, best_epoch, waited = val_acc, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xte_t).argmax(1).numpy()
    return {
        "val_acc": round(best_val, 4),
        "test_acc": round(float((pred == yte).mean()), 4),
        "best_epoch": best_epoch,
        "seconds": round(time.perf_counter() - t0, 1),
        "history": history,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default=",".join(str(s) for s in SIZES))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s]
    seeds = [int(s) for s in args.seeds.split(",") if s]

    Xraw, y, trials = load_digits()
    X = (Xraw / 255.0).astype(np.float32)

    out = RESULTS / "cnn_control.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["n_train"], r["trial"], r["seed"]))

    total = len(sizes) * len(trials) * len(seeds)
    i = 0
    for n in sizes:
        for t, (tr, te) in enumerate(trials):
            sub = tr[stratified_subsample(y[tr], n, seed=t + 1)] if n < len(tr) else tr
            for seed in seeds:
                i += 1
                if (n, t, seed) in done:
                    print(f"[{i}/{total}] skip n={n:<5d} trial{t + 1} seed{seed}", flush=True)
                    continue
                res = train_one(X[sub], y[sub], X[te], y[te], seed)
                rec = {"n_train": int(n), "trial": t, "seed": seed, **res}
                with out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                print(
                    f"[{i}/{total}] n={n:<5d} trial{t + 1} seed{seed} "
                    f"test={res['test_acc']:.4f} val={res['val_acc']:.4f} "
                    f"epoch={res['best_epoch']} ({res['seconds']:.0f}s)",
                    flush=True,
                )


if __name__ == "__main__":
    sys.exit(main())
