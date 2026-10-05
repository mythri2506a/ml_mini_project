"""
Make all figures for the slides and write-up.

Run from the project root, AFTER src/train.py:
    python src/make_plots.py

Saves to results/figures/:
    1_pca_variance.png        how many PCA components are needed for 95% variance
    2_model_comparison.png    test ROC-AUC and F1 for every model vs baselines
    3_roc_curves.png          ROC curves of all models on the test set
    4_confusion_matrix.png    confusion matrix of the selected (best-by-CV) model
    5_feature_importance.png  which audio feature families the best model relies on
"""
import ast
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from train import RANDOM_STATE, get_models, load_data

warnings.filterwarnings("ignore")
OUT = "results/figures"


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=200)
    plt.close(fig)
    print(f"   saved {name}")


def plot_pca(X_tr):
    pca = PCA().fit(StandardScaler().fit_transform(X_tr))
    cum = np.cumsum(pca.explained_variance_ratio_)
    n95 = int(np.searchsorted(cum, 0.95) + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(cum) + 1), cum, lw=2)
    ax.axhline(0.95, color="grey", ls="--", lw=1)
    ax.axvline(n95, color="red", ls="--", lw=1)
    ax.annotate(f"{n95} components = 95%", xy=(n95, 0.95), xytext=(n95 + 30, 0.75),
                arrowprops=dict(arrowstyle="->"))
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA on 518 audio features")
    ax.grid(alpha=0.3)
    save(fig, "1_pca_variance.png")
    return n95


def plot_model_comparison(results):
    res = results.sort_values("test_roc_auc")
    y = np.arange(len(res))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(y - 0.2, res["test_roc_auc"], height=0.4, label="ROC-AUC")
    ax.barh(y + 0.2, res["test_f1"], height=0.4, label="F1 (hit class)")
    ax.axvline(0.5, color="grey", ls="--", lw=1, label="Random (AUC 0.5)")
    ax.set_yticks(y)
    ax.set_yticklabels(res["model"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score on held-out test set (unseen artists)")
    ax.set_title("Model comparison vs baselines")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    save(fig, "2_model_comparison.png")


def refit_all(results, X_tr, y_tr):
    """Refit each model once with the best hyperparameters found by train.py."""
    fitted = {}
    for name, (pipe, _) in get_models().items():
        row = results[results["model"] == name]
        if row.empty:
            continue
        params = ast.literal_eval(row["best_params"].iloc[0])
        print(f"   refitting {name} ...", flush=True)
        fitted[name] = pipe.set_params(**params).fit(X_tr, y_tr)
    return fitted


def plot_roc(fitted, X_te, y_te):
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, model in fitted.items():
        s = get_scores(model, X_te)
        fpr, tpr, _ = roc_curve(y_te, s)
        ax.plot(fpr, tpr, lw=1.5, label=f"{name} ({roc_auc_score(y_te, s):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (0.500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves on test set")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    save(fig, "3_roc_curves.png")


def plot_confusion(model, X_te, y_te):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(model, X_te, y_te, ax=ax, colorbar=False,
                                          display_labels=["Not hit", "Hit"], cmap="Blues")
    ax.set_title("Selected model: confusion matrix")
    save(fig, "4_confusion_matrix.png")


def plot_feature_importance(model, X_te, y_te, repeats=5):
    """Permutation importance per feature FAMILY (e.g. all 140 MFCC columns at once).
    Shuffling a whole family and measuring the drop in ROC-AUC tells us how much
    the model depends on that kind of audio information."""
    families = sorted({c.rsplit("_", 2)[0] for c in X_te.columns})
    base = roc_auc_score(y_te, get_scores(model, X_te))
    rng = np.random.default_rng(RANDOM_STATE)
    drops = {}
    for fam in families:
        cols = [c for c in X_te.columns if c.rsplit("_", 2)[0] == fam]
        d = []
        for _ in range(repeats):
            Xp = X_te.copy()
            Xp[cols] = Xp[cols].values[rng.permutation(len(Xp))]
            d.append(base - roc_auc_score(y_te, get_scores(model, Xp)))
        drops[fam] = (np.mean(d), np.std(d))
        print(f"   {fam:20s} AUC drop {drops[fam][0]:+.4f}", flush=True)

    imp = pd.DataFrame(drops, index=["mean", "std"]).T.sort_values("mean")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(imp.index, imp["mean"], xerr=imp["std"], color="tab:orange")
    ax.set_xlabel("Drop in test ROC-AUC when the feature family is shuffled")
    ax.set_title("Which audio features matter? (permutation importance)")
    ax.grid(axis="x", alpha=0.3)
    save(fig, "5_feature_importance.png")
    imp.to_csv("results/feature_importance.csv")


def main():
    os.makedirs(OUT, exist_ok=True)
    X, y, groups = load_data()
    results = pd.read_csv("results/results.csv")

    # Same artist-wise split as train.py (same seed), so the test set is identical.
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr, te = next(split.split(X, y, groups))
    X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]

    print("1/5 PCA variance")
    n95 = plot_pca(X_tr)
    print(f"   {n95} components explain 95% of the variance")

    print("2/5 Model comparison")
    plot_model_comparison(results)

    print("3/5 ROC curves (refits each model once; takes a few minutes)")
    plot_roc(refit_all(results, X_tr, y_tr), X_te, y_te)

    best = joblib.load("results/best_model.joblib")
    print("4/5 Confusion matrix")
    plot_confusion(best, X_te, y_te)

    print("5/5 Feature importance (a few minutes)")
    plot_feature_importance(best, X_te, y_te)

    print(f"\nDone. Figures are in {OUT}/")


if __name__ == "__main__":
    main()
