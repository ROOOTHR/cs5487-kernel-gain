"""Sanity check: reproduce the 1-NN baseline reported in PA-3 (0.9135 / 0.9185)."""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from common import load_digits

X, y, trials = load_digits()
accs = []
for i, (tr, te) in enumerate(trials):
    clf = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=-1)
    clf.fit(X[tr], y[tr])
    acc = clf.score(X[te], y[te])
    accs.append(acc)
    print(f"trial {i + 1}: {acc:.4f}")

print(f"mean {np.mean(accs):.4f}  std(ddof=1) {np.std(accs, ddof=1):.4f}")
print("expected mean 0.9160  std 0.0035")
