"""Aggregate the raw JSONL results into readable tables (markdown + CSV)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"

MODEL_LABEL = {
    "knn1": "1-NN",
    "bdr_shared": "BDR shared-cov (LDA)",
    "bdr_full": "BDR full-cov (QDA)",
    "bdr_full_eig": "BDR full-cov + reg (eigen)",
    "logreg": "Multinomial logistic",
    "svm_linear": "Linear SVM",
    "svm_rbf": "RBF SVM",
}
ORDER = list(MODEL_LABEL)


def load(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    out = (
        df.groupby(by)["test_acc"]
        .agg(mean="mean", std=lambda s: np.std(s, ddof=1) if len(s) > 1 else np.nan, n="count")
        .reset_index()
    )
    out["mean"] = out["mean"].round(4)
    out["std"] = out["std"].round(4)
    return out


def main() -> None:
    df = load(RESULTS / "grid.jsonl")
    ok = df[df["test_acc"].notna()].copy()
    failed = df[df["test_acc"].isna()]

    print(f"records: {len(df)}  ok: {len(ok)}  failed: {len(failed)}\n")

    # ---- main table: mean test accuracy over the two trials
    tab = summarise(ok, ["model", "preproc"])
    pivot = tab.pivot(index="model", columns="preproc", values="mean")
    preprocs = [p for p in ["raw", "std", "pca20", "pca50", "pca100", "pca200",
                            "pca20w", "pca50w", "pca100w", "pca200w"] if p in pivot.columns]
    pivot = pivot.reindex([m for m in ORDER if m in pivot.index])[preprocs]
    pivot.index = [MODEL_LABEL.get(i, i) for i in pivot.index]

    print("== mean test accuracy over 2 trials ==")
    print(pivot.to_string(float_format=lambda v: f"{v:.4f}"))
    pivot.to_csv(RESULTS / "table_main.csv", float_format="%.4f")

    # ---- per-model best configuration
    print("\n== best preprocessing per model ==")
    best = tab.loc[tab.groupby("model")["mean"].idxmax()].sort_values("mean", ascending=False)
    for _, r in best.iterrows():
        print(f"{MODEL_LABEL.get(r['model'], r['model']):28s} {r['mean']:.4f} +- {r['std']:.4f}"
              f"   on {r['preproc']}")

    # ---- failures
    if len(failed):
        print("\n== failed configurations (recorded, not silently dropped) ==")
        for pred in ["bdr_full", "bdr_full_eig"]:
            sub = failed[failed["model"] == pred]
            if len(sub):
                names = sorted(set(sub.apply(lambda r: f"{r['preproc']}(d={r['n_features']})", axis=1)))
                print(f"{MODEL_LABEL.get(pred, pred):28s} {len(sub)} cells failed: {', '.join(names[:12])}")
        print(f"example error: {failed.iloc[0]['error']}")

    # ---- per-trial spread of the winner
    print("\n== top 5 (model, preproc) by mean accuracy, with per-trial values ==")
    top = tab.sort_values("mean", ascending=False).head(5)
    for _, r in top.iterrows():
        per = ok[(ok["model"] == r["model"]) & (ok["preproc"] == r["preproc"])].sort_values("trial")
        vals = "  ".join(f"t{row['trial'] + 1}={row['test_acc']:.4f}" for _, row in per.iterrows())
        print(f"{MODEL_LABEL.get(r['model'], r['model']):28s} {r['preproc']:8s} "
              f"mean={r['mean']:.4f} +- {r['std']:.4f}   {vals}")

    # ---- sweep
    sw = RESULTS / "sweep.jsonl"
    if sw.exists():
        s = load(sw)
        s = s[s["test_acc"].notna()]
        agg = summarise(s, ["model", "n_features"])
        print("\n== dimension sweep: mean test accuracy vs PCA dims ==")
        piv = agg.pivot(index="n_features", columns="model", values="mean")
        cols = [c for c in ["svm_rbf", "bdr_full_eig", "svm_linear", "logreg", "bdr_shared", "knn1"]
                if c in piv.columns]
        piv = piv[cols]
        piv.columns = [MODEL_LABEL.get(c, c) for c in cols]
        print(piv.to_string(float_format=lambda v: f"{v:.4f}"))
        piv.to_csv(RESULTS / "table_sweep.csv", float_format="%.4f")

        linear_cols = [c for c in ["Linear SVM", "Multinomial logistic", "BDR shared-cov (LDA)"]
                       if c in piv.columns]
        best_linear = piv[linear_cols].max(axis=1)
        print("\n== gain over the best linear model, by dimension (pp) ==")
        print("   k    RBF-bestLin   QDA-bestLin")
        for k in piv.index:
            rbf = piv.loc[k, "RBF SVM"] - best_linear[k]
            qda = (piv.loc[k, "BDR full-cov + reg (eigen)"] - best_linear[k]
                   if "BDR full-cov + reg (eigen)" in piv.columns else float("nan"))
            print(f"{k:>4}  {rbf * 100:+11.1f}  {qda * 100:+11.1f}")

    # ---- learning curves
    lc_path = RESULTS / "learning_curve.jsonl"
    if lc_path.exists():
        l = load(lc_path)
        l = l[l["test_acc"].notna()]
        agg = summarise(l, ["model", "n_train"])
        print("\n== learning curve: mean test accuracy vs training-set size (pca50) ==")
        piv = agg.pivot(index="n_train", columns="model", values="mean")
        cols = [c for c in ["svm_rbf", "bdr_full_eig", "svm_linear", "logreg", "knn1"] if c in piv.columns]
        piv = piv[cols]
        disp = piv.copy()
        disp.columns = [MODEL_LABEL.get(c, c) for c in cols]
        print(disp.to_string(float_format=lambda v: f"{v:.4f}"))
        disp.to_csv(RESULTS / "table_learning_curve.csv", float_format="%.4f")

        lin = [c for c in ["svm_linear", "logreg", "bdr_shared"] if c in piv.columns]
        gain = (piv["svm_rbf"] - piv[lin].max(axis=1)) * 100
        print("\n== kernel gain vs sample size (pp) ==")
        for n, g in gain.items():
            print(f"n={n:<5d} gain={g:+.2f}")

        print("\n== hyper-parameters re-tuned at each n (RBF SVM); gamma as a multiple of 1/(k*var) ==")
        sub = l[l["model"] == "svm_rbf"].sort_values(["n_train", "trial"])
        for n, grp in sub.groupby("n_train"):
            cs = sorted({p.get("C") for p in grp["best_params"]})
            gm = sorted({round(p.get("gamma") / r["g0"], 2)
                         for _, r in grp.iterrows() for p in [r["best_params"]] if r.get("g0")})
            print(f"n={n:<5d} C={cs}  gamma={gm} x g0")

    # ---- gamma / C sensitivity (training-set CV only)
    gm_path = RESULTS / "gamma_sensitivity.jsonl"
    if gm_path.exists():
        g = load(gm_path)
        print("\n== RBF SVM sensitivity: best training-set CV accuracy per feature space ==")
        print("space      best CV   argmax(C, gamma/g0)   cells within 1pp   gamma range at best C")
        for p, sub in g.groupby("preproc"):
            agg2 = sub.groupby(["gamma_mult", "C"])["cv_acc"].mean().reset_index()
            best = agg2.loc[agg2["cv_acc"].idxmax()]
            near = int((agg2["cv_acc"] >= best["cv_acc"] - 0.01).sum())
            tot = len(agg2)
            row = sub[sub["C"] == best["C"]].groupby("gamma_mult")["cv_acc"].mean()
            lo = row[row >= best["cv_acc"] - 0.01].index.min()
            hi = row[row >= best["cv_acc"] - 0.01].index.max()
            print(f"{p:9s}  {best['cv_acc']:.4f}   C={best['C']:<6g} gamma={best['gamma_mult']:<5g}"
                  f"      {near:>2d}/{tot}                 {lo:g} - {hi:g} x g0")
        g.to_csv(RESULTS / "table_gamma_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
