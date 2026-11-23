"""Shared data loading / preprocessing utilities for the CS5487 course project.

The provided txt files are TRANSPOSED relative to the MATLAB matrices
(see digits4000_txt/README.txt): each TEXT ROW is a MATLAB COLUMN.

MATLAB                        -> text file shape
digits_vec   784 x 4000       -> 4000 x 784   (row i = image i)
digits_labels 1 x 4000        -> 4000 x 1
trainset      2 x 2000        -> 2000 x 2     (col t = trial t indices)
testset       2 x 2000        -> 2000 x 2
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FILES = ("digits4000_digits_vec.txt", "digits4000_digits_labels.txt",
         "digits4000_trainset.txt", "digits4000_testset.txt")


def _looks_like_data_dir(p: Path) -> bool:
    return all((p / f).exists() for f in FILES)


def find_data_dir() -> Path:
    """Locate the digits4000 txt files.

    Checked in order:
      1. $DIGITS_DIR
      2. ./data/digits4000_txt or ./data (repo layout)
      3. ../course project proposal/digits4000_txt (the author's original layout)
    """
    candidates = []
    if os.environ.get("DIGITS_DIR"):
        candidates.append(Path(os.environ["DIGITS_DIR"]))
    candidates += [
        HERE / "data" / "digits4000_txt",
        HERE / "data",
        HERE.parent / "course project proposal" / "digits4000_txt",
    ]
    for c in candidates:
        if _looks_like_data_dir(c):
            return c
    raise FileNotFoundError(
        "digits4000 data not found. Put the four *.txt files in ./data/digits4000_txt/ "
        "or set DIGITS_DIR to the folder that contains them.\nLooked in:\n  "
        + "\n  ".join(str(c) for c in candidates)
    )


def load_digits(data_dir: Path | None = None):
    """Return X (4000, 784) float32 in [0, 255], y (4000,) int, and the two trials."""
    data_dir = Path(data_dir) if data_dir else find_data_dir()
    vec = np.loadtxt(data_dir / "digits4000_digits_vec.txt", dtype=np.int16, delimiter="\t")
    labels = np.loadtxt(data_dir / "digits4000_digits_labels.txt", dtype=np.int16, delimiter="\t")
    tr = np.loadtxt(data_dir / "digits4000_trainset.txt", dtype=np.int32, delimiter="\t")
    te = np.loadtxt(data_dir / "digits4000_testset.txt", dtype=np.int32, delimiter="\t")

    assert vec.shape == (4000, 784), vec.shape
    assert labels.shape == (4000,), labels.shape
    assert tr.shape == (2000, 2) and te.shape == (2000, 2), (tr.shape, te.shape)

    y = labels.astype(np.int64)

    # Each text column is one experimental trial; MATLAB indices are 1-based.
    trials = []
    for t in range(tr.shape[1]):
        tr_idx = tr[:, t].astype(np.int64) - 1
        te_idx = te[:, t].astype(np.int64) - 1
        assert len(np.intersect1d(tr_idx, te_idx)) == 0
        assert len(np.union1d(tr_idx, te_idx)) == 4000
        trials.append((tr_idx, te_idx))

    return vec.astype(np.float32), y, trials


if __name__ == "__main__":
    X, y, trials = load_digits()
    print("X", X.shape, X.dtype, "min/max", X.min(), X.max())
    print("y classes", np.unique(y, return_counts=True))
    for i, (tr, te) in enumerate(trials):
        print(f"trial {i + 1}: train {len(tr)} test {len(te)}")
        print("  train class counts", np.bincount(y[tr], minlength=10))
        print("  test  class counts", np.bincount(y[te], minlength=10))
