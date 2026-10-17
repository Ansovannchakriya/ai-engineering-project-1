from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).parent
DATA = HERE / "data" / "mushrooms" / "mushrooms.csv"
MODELS = HERE / "models"

TARGET = "poisonous"
# The CSV stores every value as one letter. The letters are decoded into words, so the model,
# the API and its users all speak the same language ("odor": "foul", not "odor": "f").
CODES = {
    "cap_shape": {
        "b": "bell", "c": "conical", "x": "convex", "f": "flat", "k": "knobbed", "s": "sunken",
    },
    "cap_surface": {"f": "fibrous", "g": "grooves", "y": "scaly", "s": "smooth"},
    "cap_color": {
        "n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "r": "green", "p": "pink",
        "u": "purple", "e": "red", "w": "white", "y": "yellow",
    },
    "bruises": {"t": "yes", "f": "no"},
    "odor": {
        "a": "almond", "l": "anise", "c": "creosote", "y": "fishy", "f": "foul", "m": "musty",
        "n": "none", "p": "pungent", "s": "spicy",
    },
    "gill_attachment": {"a": "attached", "d": "descending", "f": "free", "n": "notched"},
    "gill_spacing": {"c": "close", "w": "crowded", "d": "distant"},
    "gill_size": {"b": "broad", "n": "narrow"},
    "gill_color": {
        "k": "black", "n": "brown", "b": "buff", "h": "chocolate", "g": "gray", "r": "green",
        "o": "orange", "p": "pink", "u": "purple", "e": "red", "w": "white", "y": "yellow",
    },
    "stalk_shape": {"e": "enlarging", "t": "tapering"},
    "stalk_root": {
        "b": "bulbous", "c": "club", "u": "cup", "e": "equal", "z": "rhizomorphs", "r": "rooted",
        "?": "missing",
    },
    "stalk_surface_above_ring": {"f": "fibrous", "y": "scaly", "k": "silky", "s": "smooth"},
    "stalk_surface_below_ring": {"f": "fibrous", "y": "scaly", "k": "silky", "s": "smooth"},
    "stalk_color_above_ring": {
        "n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "o": "orange", "p": "pink",
        "e": "red", "w": "white", "y": "yellow",
    },
    "stalk_color_below_ring": {
        "n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "o": "orange", "p": "pink",
        "e": "red", "w": "white", "y": "yellow",
    },
    "veil_type": {"p": "partial", "u": "universal"},
    "veil_color": {"n": "brown", "o": "orange", "w": "white", "y": "yellow"},
    "ring_number": {"n": "none", "o": "one", "t": "two"},
    "ring_type": {
        "c": "cobwebby", "e": "evanescent", "f": "flaring", "l": "large", "n": "none",
        "p": "pendant", "s": "sheathing", "z": "zone",
    },
    "spore_print_color": {
        "k": "black", "n": "brown", "b": "buff", "h": "chocolate", "r": "green", "o": "orange",
        "u": "purple", "w": "white", "y": "yellow",
    },
    "population": {
        "a": "abundant", "c": "clustered", "n": "numerous", "s": "scattered", "v": "several",
        "y": "solitary",
    },
    "habitat": {
        "g": "grasses", "l": "leaves", "m": "meadows", "p": "paths", "u": "urban", "w": "waste",
        "d": "woods",
    },
}  # fmt: skip
# veil_type has one value in all 8,124 rows: a column that never changes tells the model nothing.
FEATURES = [column for column in CODES if column != "veil_type"]


def load(path: Path = DATA) -> pd.DataFrame:
    """The mushrooms with readable names: `poisonous` (1 / 0) and 21 features as words."""
    raw = pd.read_csv(path)
    raw.columns = [column.replace("-", "_") for column in raw.columns]
    df = pd.DataFrame({column: raw[column].map(codes) for column, codes in CODES.items()})
    if df.isna().any().any():
        raise ValueError(f"{path.name} has letters that are not in CODES")
    df[TARGET] = (raw["class"] == "p").astype(int)
    return df


def step(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


def train(data_path: Path = DATA, models_dir: Path = MODELS, verbose: bool = True) -> dict:
    say = print if verbose else (lambda *args, **kwargs: None)
    show = step if verbose else (lambda title: None)

    show("1. Load the data and decode the letters into words")
    df = load(data_path)
    say(f"{len(df)} mushrooms, {len(FEATURES)} features (veil_type dropped: only one value)")
    say(f"poisonous: {df[TARGET].mean():.1%}  <- the classes are balanced")

    show("2. Split: train on 80%, keep 20% aside for the final test")
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    say(f"train: {len(X_train)} rows, test: {len(X_test)} rows")

    show("3. Baselines: a guess, and a one-line rule a forager could use")
    guess = np.zeros(len(y_test), dtype=int)  # "everything is edible"
    say(f"'always edible':  accuracy {accuracy_score(y_test, guess):.1%}, "
        f"poisonous ones eaten: {int(y_test.sum())}")  # fmt: skip
    smells_safe = X_test["odor"].isin(["almond", "anise", "none"])
    rule = (~smells_safe).astype(int)
    missed = int(((rule == 0) & (y_test == 1)).sum())
    say(f"'smells bad -> poisonous': accuracy {accuracy_score(y_test, rule):.1%}, "
        f"poisonous ones eaten: {missed}")  # fmt: skip
    say("-> the rule is very good, but every miss here is a poisoning: we need zero misses")

    show("4. Compare two models with 5-fold cross-validation")
    candidates = {
        "logistic regression": make_pipeline(
            OneHotEncoder(handle_unknown="ignore"), LogisticRegression(max_iter=1000)
        ),
        "random forest": make_pipeline(
            OneHotEncoder(handle_unknown="ignore"),
            RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        ),
    }
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = {}
    for name, model in candidates.items():
        loss = -cross_val_score(model, X_train, y_train, cv=folds, scoring="neg_log_loss")
        scores[name] = loss.mean()
        say(f"{name:<20} log loss {loss.mean():.4f} (+/- {loss.std():.4f})  lower is better")
    best_name = min(scores, key=scores.get)
    model = candidates[best_name]
    say(f"-> best: {best_name}")

    show("5. Choose the threshold: safety first (on training data, never on the test set)")
    # Probabilities each training row gets from a model that did not see it.
    proba_cv = cross_val_predict(model, X_train, y_train, cv=folds, method="predict_proba")[:, 1]
    # The highest threshold that still flags every poisonous mushroom, halved as a safety margin.
    threshold = float(np.floor(proba_cv[y_train == 1].min() / 2 * 100) / 100)
    threshold = max(threshold, 0.01)
    eaten = int(((proba_cv < threshold) & (y_train == 1)).sum())
    alarms = int(((proba_cv >= threshold) & (y_train == 0)).sum())
    say(f"threshold {threshold}: poisonous ones missed {eaten}, edible ones flagged {alarms}")
    say("(called 'edible' only when the probability of poison is below the threshold)")

    show("6. Train on all training data, evaluate once on the test set")
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    predicted = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
    metrics = {
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "accuracy": round(accuracy_score(y_test, predicted), 4),
        "precision": round(precision_score(y_test, predicted), 4),
        "recall": round(recall_score(y_test, predicted), 4),
        "poisonous_missed": int(fn),
        "edible_flagged": int(fp),
    }
    for name, value in metrics.items():
        say(f"{name:<17} {value}")
    say(f"poisonous caught: {tp} of {tp + fn}; edible ones flagged as poisonous: {fp} of {fp + tn}")

    show("7. Which features matter (permutation importance on the test set)")
    importance = permutation_importance(
        model, X_test, y_test, scoring="neg_log_loss", n_repeats=5, random_state=42, n_jobs=-1
    )
    ranked = sorted(zip(FEATURES, importance.importances_mean, strict=True), key=lambda x: -x[1])
    for feature, value in ranked[:5]:
        say(f"{feature:<26} {value:+.3f}")

    show("8. Save the model")
    bundle = {
        "model": model,
        "algorithm": best_name,
        "features": FEATURES,
        "categories": {feature: sorted(set(CODES[feature].values())) for feature in FEATURES},
        "threshold": threshold,
        "metrics": metrics,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / "mushroom.joblib"
    joblib.dump(bundle, path)
    say(f"saved to {path}")
    return bundle


if __name__ == "__main__":
    train()
