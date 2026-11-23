"""Figures for the project: accuracy vs dimension, model x preprocessing heatmap, kernel gain."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"
FIGS = RESULTS / "figs"
FIGS.mkdir(exist_ok=True)

MODEL_LABEL = {
    "knn1": "1-NN",
    "bdr_shared": "BDR shared-cov",
    "bdr_full": "BDR full-cov",
    "bdr_full_eig": "BDR full-cov (eigen)",
    "logreg": "Multinomial logistic",
    "svm_linear": "Linear SVM",
    "svm_rbf": "RBF SVM",
}
COLORS = {
    "svm_rbf": "#c0392b",
    "svm_linear": "#2c3e50",
    "logreg": "#2980b9",
    "bdr_shared": "#27ae60",
    "knn1": "#7f8c8d",
    "bdr_full": "#8e44ad",
    "bdr_full_eig": "#8e44ad",
}


def load(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def fig_sweep() -> None:
    df = load(RESULTS / "sweep.jsonl")
    df = df[df["test_acc"].notna()]
    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=160)
    for model in ["svm_rbf", "bdr_full_eig", "svm_linear", "logreg", "bdr_shared", "knn1"]:
        sub = df[df["model"] == model].sort_values("n_features")
        if sub.empty:
            continue
        g = sub.groupby("n_features")["test_acc"].agg(["mean", "min", "max"])
        ax.plot(g.index, g["mean"], marker="o", ms=4, lw=1.8,
                color=COLORS[model], label=MODEL_LABEL[model])
        ax.fill_between(g.index, g["min"], g["max"], color=COLORS[model], alpha=0.12, lw=0)
    ax.axhline(0.9160, ls="--", lw=1.2, color="#e67e22", label="1-NN baseline (0.9160)")
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 50, 100, 200, 500])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("PCA dimensions retained (feature dimension)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy vs. feature dimension: linear vs. kernel classifiers")
    ax.grid(alpha=0.25, ls=":")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.015, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_dimension_sweep.png")
    plt.close(fig)
    print("wrote", FIGS / "fig1_dimension_sweep.png")


def fig_heatmap() -> None:
    df = load(RESULTS / "grid.jsonl")
    df = df[df["test_acc"].notna()]
    tab = df.groupby(["model", "preproc"])["test_acc"].mean().reset_index()
    piv = tab.pivot(index="model", columns="preproc", values="test_acc")
    preprocs = [p for p in ["raw", "std", "pca20", "pca50", "pca100", "pca200",
                            "pca20w", "pca50w", "pca100w", "pca200w"] if p in piv.columns]
    models = [m for m in ["svm_rbf", "svm_linear", "logreg", "bdr_shared", "bdr_full_eig", "knn1"]
              if m in piv.index]
    piv = piv.reindex(index=models, columns=preprocs)
    piv.index = [MODEL_LABEL[m] for m in models]

    fig, ax = plt.subplots(figsize=(8.2, 3.6), dpi=160)
    data = piv.values.astype(float)
    im = ax.imshow(data, cmap="RdYlGn", vmin=np.nanmin(data) - 0.01, vmax=np.nanmax(data), aspect="auto")
    ax.set_xticks(range(len(preprocs)))
    ax.set_xticklabels(preprocs, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isnan(data[i, j]):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color="#555")
            else:
                ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=7.5,
                        color="black")
    ax.set_title("Test accuracy: classifier x preprocessing (mean of 2 trials)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_heatmap.png")
    plt.close(fig)
    print("wrote", FIGS / "fig2_heatmap.png")


def fig_gain() -> None:
    df = load(RESULTS / "sweep.jsonl")
    df = df[df["test_acc"].notna()]
    piv = df.groupby(["model", "n_features"])["test_acc"].mean().unstack(0)
    if "svm_rbf" not in piv.columns:
        return
    linear_cols = [c for c in ["svm_linear", "logreg", "bdr_shared"] if c in piv.columns]
    best_linear = piv[linear_cols].max(axis=1)
    gain = (piv["svm_rbf"] - best_linear)

    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=160)
    ax.plot(gain.index, gain.values * 100, marker="o", ms=5, lw=1.8, color="#c0392b")
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 50, 100, 200, 500])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("PCA dimensions retained")
    ax.set_ylabel("Accuracy gain of RBF SVM\nover best linear model (pp)")
    ax.set_title("Where does nonlinearity pay off?")
    ax.grid(alpha=0.25, ls=":")
    for x, y in zip(gain.index, gain.values * 100):
        ax.annotate(f"{y:+.1f}", (x, y), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7.5, color="#c0392b")
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_kernel_gain.png")
    plt.close(fig)
    print("wrote", FIGS / "fig3_kernel_gain.png")


if __name__ == "__main__":
    fig_sweep()
    fig_heatmap()
    fig_gain()
