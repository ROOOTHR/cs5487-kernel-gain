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


def fig_learning_curve() -> None:
    """Accuracy vs. number of training images, plus how the kernel gain evolves with n."""
    path = RESULTS / "learning_curve.jsonl"
    if not path.exists():
        return
    df = load(path)
    df = df[df["test_acc"].notna()]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0), dpi=160,
                                  gridspec_kw={"width_ratios": [1.45, 1.0]})

    order = ["svm_rbf", "bdr_full_eig", "svm_linear", "logreg", "knn1"]
    for model in order:
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        g = sub.groupby("n_train")["test_acc"].agg(["mean", "min", "max"])
        ax.plot(g.index, g["mean"], marker="o", ms=4, lw=1.8,
                color=COLORS[model], label=MODEL_LABEL[model])
        ax.fill_between(g.index, g["min"], g["max"], color=COLORS[model], alpha=0.13, lw=0)
    ax.axhline(0.9160, ls="--", lw=1.2, color="#e67e22", label="1-NN baseline (784-d)")

    # The CNN control group, at the sample sizes where it was run (bars = std over 2 seeds x 2 trials).
    cnn_path = RESULTS / "cnn_control.jsonl"
    if cnn_path.exists():
        cn = load(cnn_path)
        gc = cn.groupby("n_train")["test_acc"].agg(["mean", "std"])
        ax.errorbar(gc.index, gc["mean"], yerr=gc["std"], fmt="*", ms=12, lw=1.0,
                    color="#111111", ecolor="#111111", capsize=2.5, elinewidth=0.9,
                    label="Small CNN (2 seeds × 2 trials)", zorder=6,
                    markeredgecolor="white", markeredgewidth=0.9)
    ax.set_xscale("log")
    ax.set_xticks([100, 200, 500, 1000, 2000])
    ax.set_xlim(80, 2600)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Training images per classifier (class-balanced subsample of the 2,000)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Learning curves on 50 PCA dimensions")
    ax.grid(alpha=0.25, ls=":")
    ax.legend(fontsize=8, loc="lower right", frameon=True, framealpha=1.0,
              facecolor="white", edgecolor="#dddddd")

    piv = df.groupby(["model", "n_train"])["test_acc"].mean().unstack(0)
    linear_cols = [c for c in ["svm_linear", "logreg", "bdr_shared"] if c in piv.columns]
    if "svm_rbf" in piv.columns and linear_cols:
        gain = (piv["svm_rbf"] - piv[linear_cols].max(axis=1)) * 100
        ax2.plot(gain.index, gain.values, marker="s", ms=5, lw=1.8, color="#c0392b")
        ax2.axhline(0, color="#333", lw=0.8)
        # Only the two points that carry the story are labelled: the small-sample floor and the peak.
        peak_x = int(gain.idxmax())
        for x, yv in zip(gain.index, gain.values):
            if x not in (100, peak_x):
                continue
            ax2.annotate(f"{yv:+.1f}", (x, yv), textcoords="offset points",
                         xytext=(9, 6) if x == 100 else (0, 9),
                         ha="left" if x == 100 else "center", fontsize=7.8, color="#c0392b")
        plateau = float(gain[gain.index >= 700].mean())
        ax2.axhline(plateau, ls=":", lw=1.1, color="#7f8c8d")
        ax2.annotate(f"plateau\n{plateau:+.1f} pp (n ≥ 700)", (108, plateau),
                     textcoords="offset points", xytext=(0, 4), ha="left", va="bottom",
                     fontsize=7.2, color="#555", linespacing=1.35)
        ax2.set_xscale("log")
        ax2.set_xticks([100, 200, 500, 1000, 2000])
        ax2.set_xlim(80, 2600)
        ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax2.set_xlabel("Training images per classifier")
        ax2.set_ylabel("RBF SVM − best linear model (pp)")
        ax2.set_title("Kernel gain vs. sample size")
        ax2.grid(alpha=0.25, ls=":")
        ax2.margins(y=0.25)

    fig.tight_layout()
    fig.savefig(FIGS / "fig5_learning_curve.png")
    plt.close(fig)
    print("wrote", FIGS / "fig5_learning_curve.png")


def fig_gamma_sensitivity() -> None:
    """Training-set CV accuracy of the RBF SVM over (C, gamma), one panel per dimension."""
    path = RESULTS / "gamma_sensitivity.jsonl"
    if not path.exists():
        return
    df = load(path)
    preprocs = [p for p in ["pca20", "pca50", "pca200", "raw"] if p in set(df["preproc"])]
    if not preprocs:
        return

    panels = []
    for p in preprocs:
        sub = df[df["preproc"] == p]
        piv = sub.groupby(["gamma_mult", "C"])["cv_acc"].mean().unstack(1)
        panels.append((p, piv))

    vmin = min(float(piv.values.min()) for _, piv in panels)
    vmax = max(float(piv.values.max()) for _, piv in panels)

    ncol = 2
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.4, 3.5 * nrow), dpi=160, squeeze=False)

    for ax, (p, piv) in zip(axes.ravel(), panels):
        data = piv.values.astype(float)
        im = ax.imshow(data, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
        ax.set_xticks(range(data.shape[1]))
        ax.set_xticklabels([f"{c:g}" for c in piv.columns], fontsize=8)
        ax.set_yticks(range(data.shape[0]))
        ax.set_yticklabels([f"{g:g}" for g in piv.index], fontsize=8)
        ax.set_xlabel("C (regularisation)", fontsize=9)
        ax.set_ylabel(r"$\gamma$ as multiple of $1/(k\cdot\bar{\sigma}^2)$", fontsize=9)
        k_label = "raw 784" if p == "raw" else p.replace("pca", "PCA-")
        ax.set_title(k_label, fontsize=10)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=7, color="black")
        bi, bj = np.unravel_index(np.nanargmax(data), data.shape)
        ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, ec="#111", lw=1.8))

    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")

    fig.suptitle("RBF SVM: 5-fold CV accuracy inside the training set (box = argmax)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 0.90, 0.96))
    cax = fig.add_axes((0.915, 0.15, 0.016, 0.68))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("CV accuracy", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.savefig(FIGS / "fig6_gamma_sensitivity.png")
    plt.close(fig)
    print("wrote", FIGS / "fig6_gamma_sensitivity.png")


if __name__ == "__main__":
    fig_sweep()
    fig_heatmap()
    fig_gain()
    fig_learning_curve()
    fig_gamma_sensitivity()
