"""Experiment library: preprocessing variants, classifier grids, and a runner.

Design notes
------------
* The pixel matrices are divided by 255 up front (a global monotone rescale that leaves
  Euclidean geometry unchanged; it only improves SVM conditioning). "raw" below therefore
  means "pixels in [0,1], no per-pixel statistics".
* Every transform that estimates statistics (standardisation, PCA, whitening) lives inside
  a sklearn Pipeline, so it is re-fitted on each inner CV training fold.
* Hyper-parameters are selected by 5-fold stratified CV *inside* the 2000 training images,
  exactly as PA-3 requires ("do not tune the parameters to optimise the test accuracy").
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

SEED = 0
GAMMA_MULT = (0.1, 0.3, 1.0, 3.0)


# ---------------------------------------------------------------------- preprocessing
def preproc_steps(name: str) -> list[tuple[str, object]]:
    """Named preprocessing variants. Returns pipeline steps (possibly empty)."""
    if name == "raw":
        return []

    if name == "std":
        return [("scale", StandardScaler())]

    if name.startswith("pca"):
        whiten = name.endswith("w")
        k = int(name[3:].rstrip("w"))
        return [("pca", PCA(n_components=k, whiten=whiten, random_state=SEED))]

    if name.startswith("std_pca"):
        whiten = name.endswith("w")
        k = int(name[7:].rstrip("w"))
        return [
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=k, whiten=whiten, random_state=SEED)),
        ]

    raise ValueError(f"unknown preproc {name}")


def n_features_of(preproc: str, d: int) -> int:
    if preproc.startswith("pca"):
        return int(preproc[3:].rstrip("w"))
    if preproc.startswith("std_pca"):
        return int(preproc[7:].rstrip("w"))
    return d


# ----------------------------------------------------------------------------- models
def classifier(name: str, k: int, var: float) -> tuple[object, dict]:
    """Return (estimator, param_grid) for the reduced feature dimension ``k``.

    ``var`` is the mean per-feature variance of the *training* features in that space,
    used to put the RBF gamma grid on a comparable scale across feature spaces.
    """
    if name == "knn1":
        return KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=4), {}

    if name == "bdr_shared":  # BDR with class-independent Gaussian == LDA
        return LinearDiscriminantAnalysis(solver="lsqr"), {}

    if name == "bdr_shared_shr":  # same, with shrinkage on the pooled covariance
        return (
            LinearDiscriminantAnalysis(solver="eigen"),
            {"shrinkage": [None, 0.1, 0.3, 0.5, 0.7, 0.9, "auto"]},
        )

    if name == "bdr_full":  # BDR with class-dependent full covariance == QDA
        return QuadraticDiscriminantAnalysis(), {"reg_param": [0.001, 0.01, 0.1, 0.5]}

    if name == "bdr_full_eig":  # same, but the eigen solver tolerates p > n
        return (
            QuadraticDiscriminantAnalysis(solver="eigen"),
            {"reg_param": [0.001, 0.01, 0.1, 0.5], "shrinkage": [None, 0.1]},
        )

    if name == "logreg":
        return (
            LogisticRegression(max_iter=3000, solver="lbfgs", random_state=SEED),
            {"C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
        )

    if name == "svm_linear":
        return SVC(kernel="linear", cache_size=500), {
            "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        }

    if name == "svm_rbf":
        g0 = 1.0 / (k * var)
        return SVC(kernel="rbf", cache_size=500), {
            "C": [1.0, 10.0, 100.0, 1000.0],
            "gamma": [m * g0 for m in GAMMA_MULT],
        }

    raise ValueError(f"unknown model {name}")


def build(preproc: str, model: str, k: int, var: float) -> tuple[Pipeline, dict]:
    steps = preproc_steps(preproc) + [("clf", None)]
    pipe = Pipeline(steps)
    est, grid = classifier(model, k, var)
    pipe.set_params(clf=est)
    return pipe, {f"clf__{key}": val for key, val in grid.items()}


def feature_stats(X: np.ndarray, preproc: str) -> tuple[int, float]:
    """Dimension and mean feature variance of the preprocessed TRAINING set.

    Fitting the preprocessing on the training set to *centre the hyper-parameter grid*
    is legitimate (PA-3 allows parameter selection on training data only); the test set
    is never touched.
    """
    steps = preproc_steps(preproc)
    Z = X
    if steps:
        Z = Pipeline(steps).fit_transform(X)
    k = Z.shape[1]
    var = float(np.var(Z, axis=0).mean())
    return k, max(var, 1e-12)


# --------------------------------------------------------------------------- running
def run_config(
    X: np.ndarray,
    y: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
    model_name: str,
    preproc: str,
    trial: int,
    n_splits: int = 5,
) -> dict:
    Xtr, ytr = X[tr], y[tr]
    Xte, yte = X[te], y[te]

    k, var = feature_stats(Xtr, preproc)
    pipe, grid = build(preproc, model_name, k, var)

    t0 = time.perf_counter()
    try:
        if grid:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
            search = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=1, refit=True)
            search.fit(Xtr, ytr)
            best = search.best_estimator_
            best_params = {
                key.replace("clf__", ""): (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for key, v in search.best_params_.items()
            }
            cv_acc = float(search.best_score_)
        else:
            best = pipe.fit(Xtr, ytr)
            best_params = {}
            cv_acc = float("nan")
    except Exception as exc:  # a singular covariance in high dimension is a *result*, not a bug
        return {
            "model": model_name,
            "preproc": preproc,
            "trial": trial,
            "n_features": int(k),
            "cv_acc": None,
            "test_acc": None,
            "best_params": {},
            "fit_seconds": round(time.perf_counter() - t0, 2),
            "error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}",
            "y_pred": None,
        }
    fit_s = time.perf_counter() - t0

    y_pred = best.predict(Xte)
    return {
        "model": model_name,
        "preproc": preproc,
        "trial": trial,
        "n_features": int(k),
        "cv_acc": None if np.isnan(cv_acc) else round(cv_acc, 4),
        "test_acc": round(float(np.mean(y_pred == yte)), 4),
        "best_params": best_params,
        "fit_seconds": round(fit_s, 2),
        "y_pred": y_pred,
    }


def append_jsonl(path: Path, rec: dict) -> None:
    rec = {k: v for k, v in rec.items() if k != "y_pred"}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
