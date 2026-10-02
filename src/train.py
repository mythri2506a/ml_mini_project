"""
Train and evaluate hit-song classifiers on FMA audio features.

Run from the project root:
    python src/train.py

Input : data/dataset.csv   (made by src/prepare_data.py)
Output: results/results.csv, results/best_model.joblib
"""
import os
import warnings

import joblib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


def load_data(path="data/dataset.csv"):
    df = pd.read_csv(path, index_col=0)
    y = df.pop("hit")
    groups = df.pop("artist_id")  # used so one artist never appears in both train and test
    return df, y, groups


def evaluate(model, X, y):
    """Return test metrics. ROC-AUC uses scores, the rest use hard predictions."""
    pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X)[:, 1]
    else:
        score = model.decision_function(X)
    return {
        "test_accuracy": accuracy_score(y, pred),
        "test_precision": precision_score(y, pred, zero_division=0),
        "test_recall": recall_score(y, pred, zero_division=0),
        "test_f1": f1_score(y, pred, zero_division=0),
        "test_roc_auc": roc_auc_score(y, score),
    }


def scaled_pca(clf):
    """Scaling + PCA (95% variance) live INSIDE the pipeline, so CV has no leakage."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
        ("clf", clf),
    ])


def get_models():
    """Model name -> (pipeline, hyperparameter grid). Mirrors the paper's model list."""
    return {
        "LR (elastic net)": (
            scaled_pca(LogisticRegression(solver="saga", penalty="elasticnet",
                                          class_weight="balanced", max_iter=3000,
                                          random_state=RANDOM_STATE)),
            {"clf__C": [0.01, 0.1, 1], "clf__l1_ratio": [0.0, 0.5, 1.0]},
        ),
        "LDA (shrinkage)": (
            scaled_pca(LinearDiscriminantAnalysis(solver="lsqr")),
            {"clf__shrinkage": ["auto", 0.1, 0.5, 0.9]},
        ),
        "Linear SVM": (
            scaled_pca(LinearSVC(class_weight="balanced", max_iter=5000,
                                 random_state=RANDOM_STATE)),
            {"clf__C": [0.001, 0.01, 0.1]},
        ),
        "RBF SVM": (
            scaled_pca(SVC(kernel="rbf", class_weight="balanced",
                           random_state=RANDOM_STATE)),
            {"clf__C": [0.1, 1, 10]},
        ),
        "Poly SVM": (
            scaled_pca(SVC(kernel="poly", degree=2, class_weight="balanced",
                           random_state=RANDOM_STATE)),
            {"clf__C": [0.1, 1]},
        ),
        # Trees don't need scaling or PCA, so they get the raw 518 features.
        "Random Forest": (
            Pipeline([("clf", RandomForestClassifier(n_estimators=300,
                                                     class_weight="balanced_subsample",
                                                     n_jobs=-1,
                                                     random_state=RANDOM_STATE))]),
            {"clf__max_depth": [None, 10], "clf__min_samples_leaf": [1, 5]},
        ),
        "Gradient Boosting": (
            Pipeline([("clf", HistGradientBoostingClassifier(class_weight="balanced",
                                                             random_state=RANDOM_STATE))]),
            {"clf__learning_rate": [0.05, 0.1], "clf__max_depth": [3, None]},
        ),
        "Neural Network": (
            scaled_pca(MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True,
                                     max_iter=500, random_state=RANDOM_STATE)),
            {"clf__alpha": [1e-3, 1e-2, 1e-1]},  # alpha = L2 weight decay
        ),
    }


def main():
    X, y, groups = load_data()
    print(f"Dataset: {X.shape[0]} songs, {X.shape[1]} features, "
          f"{y.mean():.0%} hits, {groups.nunique()} artists")

    # 80/20 split by ARTIST, so the test set contains only unseen artists.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    g_tr = groups.iloc[train_idx]
    print(f"Train: {len(X_tr)} songs | Test: {len(X_te)} songs\n")

    rows = []

    # Baselines: anything we train must beat these to mean something.
    for name, strategy in [("Baseline: always 'not hit'", "most_frequent"),
                           ("Baseline: random guess", "stratified")]:
        dummy = DummyClassifier(strategy=strategy, random_state=RANDOM_STATE).fit(X_tr, y_tr)
        rows.append({"model": name, "cv_roc_auc": None, **evaluate(dummy, X_te, y_te)})

    cv = GroupKFold(n_splits=5)
    best_name, best_cv, best_model = None, -1.0, None

    for name, (pipe, grid) in get_models().items():
        print(f"Training {name} ...", flush=True)
        search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
        search.fit(X_tr, y_tr, groups=g_tr)
        metrics = evaluate(search.best_estimator_, X_te, y_te)
        rows.append({"model": name, "cv_roc_auc": search.best_score_, **metrics,
                     "best_params": search.best_params_})
        print(f"   CV ROC-AUC {search.best_score_:.3f} | test ROC-AUC "
              f"{metrics['test_roc_auc']:.3f} | test F1 {metrics['test_f1']:.3f}")

        # Pick the final model by CV score, never by test score.
        if search.best_score_ > best_cv:
            best_name, best_cv, best_model = name, search.best_score_, search.best_estimator_

    os.makedirs("results", exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv("results/results.csv", index=False)
    joblib.dump(best_model, "results/best_model.joblib")

    cols = ["model", "cv_roc_auc", "test_accuracy", "test_precision",
            "test_recall", "test_f1", "test_roc_auc"]
    print("\n" + results[cols].round(3).to_string(index=False))
    print(f"\nBest model by CV: {best_name} (saved to results/best_model.joblib)")


if __name__ == "__main__":
    main()
