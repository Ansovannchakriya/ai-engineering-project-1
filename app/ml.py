"""Load the trained model and make predictions with it."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import joblib
import pandas as pd
import sklearn

# Where train_mushroom.py saved the model (override for tests).
MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path(__file__).parent.parent / "models"))

WARNING = "A model is not a mycologist: never eat a wild mushroom on the strength of a prediction."


def load(name: str) -> dict | None:
    """The saved bundle (model + metadata), or None if that model was not trained yet."""
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        return None
    bundle = joblib.load(path)  # only load model files you trust: joblib can run code
    if bundle.get("sklearn_version") != sklearn.__version__:
        warnings.warn(
            f"{path.name} was trained with scikit-learn {bundle.get('sklearn_version')} but "
            f"{sklearn.__version__} is installed: retrain it if predictions look wrong.",
            stacklevel=2,
        )
    return bundle


def unseen(bundle: dict, mushroom: dict) -> list[str]:
    """Features whose value never appeared in the training data (the model cannot use them)."""
    known = bundle["model"].named_steps["onehotencoder"].categories_
    return [
        feature
        for feature, categories in zip(bundle["features"], known, strict=True)
        if mushroom[feature] not in categories
    ]


def predict_mushroom(bundle: dict, mushrooms: list[dict]) -> list[dict]:
    frame = pd.DataFrame(mushrooms, columns=bundle["features"])
    probabilities = bundle["model"].predict_proba(frame)[:, 1]
    threshold = bundle["threshold"]
    return [
        {
            "poisonous_probability": round(float(p), 4),
            "is_poisonous": bool(p >= threshold),
            "verdict": "poisonous" if p >= threshold else "edible",
            "threshold": threshold,
            "unseen_features": unseen(bundle, mushroom),
            "warning": WARNING,
        }
        for p, mushroom in zip(probabilities, mushrooms, strict=True)
    ]
