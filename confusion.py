"""Error analysis for the best configuration: confusion matrix, per-class recall,
and the most confused digit pairs (aggregated over both trials)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from common import load_digits
from exp_lib import RESULTS, build, feature_stats

FIGS = RESULTS / "figs"
FIGS.mkdir(exist_ok=True)


def best_config() -> tuple[str, str]:
    """Pick the (model, preproc) with the highest mean test accuracy in grid.jsonl."""
    rows = [json.loads(l) for l in (RESULTS / "grid.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    df = df[df["test_acc"].notna()]
    tab = df.groupby(["model", "preproc"])["test_acc"].mean().sort_values(ascending=False)
    model, preproc = tab.index[0]
    return model, preproc


def main() -> None:
    model, preproc = best_config() if len(sys.argv) < 2 else (sys.argv[1], sys.argv[2])
    print(f"analysing best config: {model} on {preproc}")

    rows = [json.loads(l) for l in (RESULTS / "grid.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    df = df[(df["model"] == model) & (df["preproc"] == preproc)].sort_values("trial")

    Xraw, y, trials = load_digits()
    X = Xraw / 255.0

    cms = []
    for t, (tr, te) in enumerate(trials):
        params = df[df["trial"] == t].iloc[0]["best_params"]
        k, var = feature_stats(X[tr], preproc)
        pipe, _ = build(preproc, model, k, var)
        pipe.set_params(**{f"clf__{key}": val for key, val in params.items()})
        pipe.fit(X[tr], y[tr])
        pred = pipe.predict(X[te])
        cm = confusion_matrix(y[te], pred, labels=range(10))
        cms.append(cm)
        acc = (pred == y[te]).mean()
        print(f"trial {t + 1}: acc={acc:.4f} params={params}")

    cm = sum(cms)
    cm_df = pd.DataFrame(cm, index=[f"true {d}" for d in range(10)], columns=[f"pred {d}" for d in range(10)])
    cm_df.to_csv(RESULTS / f"confusion_{model}_{preproc}.csv")

    # normalise rows for plotting
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.4, 4.8), dpi=170)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            if cm[i, j] == 0:
                continue
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=7,
                    color="white" if cmn[i, j] > 0.5 else "#222")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix (both trials summed) - {model} on {preproc}")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label="row-normalised rate")
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_confusion.png")
    plt.close(fig)
    print("wrote", FIGS / "fig4_confusion.png")

    # per-class recall / precision
    per = []
    for d in range(10):
        tp = cm[d, d]
        off = [(j, cm[d, j]) for j in range(10) if j != d and cm[d, j] > 0]
        off.sort(key=lambda t: -t[1])
        per.append({
            "digit": d,
            "recall": tp / cm[d].sum(),
            "precision": tp / cm[:, d].sum() if cm[:, d].sum() else np.nan,
            "n_errors": int(cm[d].sum() - tp),
            "top_confusions": ", ".join(f"->{j}:{c}" for j, c in off[:3]),
        })
    per_df = pd.DataFrame(per)
    print("\n== per-class performance (both trials) ==")
    print(per_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    per_df.to_csv(RESULTS / "per_class.csv", index=False)

    # most confused unordered pairs
    pairs = []
    for i in range(10):
        for j in range(i + 1, 10):
            pairs.append({"pair": f"{i}-{j}", "errors": int(cm[i, j] + cm[j, i])})
    pairs = pd.DataFrame(pairs).sort_values("errors", ascending=False).head(8)
    print("\n== most confused digit pairs (sum of both directions, both trials) ==")
    print(pairs.to_string(index=False))


if __name__ == "__main__":
    main()
