"""The training script produces a model that is safe on held-out data, and the API accepts
exactly the words the model was trained on."""

import typing

import joblib

import train_mushroom
from app.schemas import Mushroom


def test_mushroom_model(models_dir):
    bundle = joblib.load(models_dir / "mushroom.joblib")
    metrics = bundle["metrics"]
    assert metrics["poisonous_missed"] == 0  # the one mistake we cannot afford
    assert metrics["recall"] == 1.0
    assert metrics["accuracy"] > 0.98
    assert 0 < bundle["threshold"] < 0.5
    assert len(bundle["features"]) == 21
    assert "veil_type" not in bundle["features"]


def test_data_decodes_fully():
    df = train_mushroom.load()
    assert len(df) == 8124
    assert not df.isna().any().any()
    assert set(df["poisonous"]) == {0, 1}


def test_schema_matches_training_codes():
    fields = Mushroom.model_fields
    assert list(fields) == train_mushroom.FEATURES
    for feature in train_mushroom.FEATURES:
        allowed = set(typing.get_args(fields[feature].annotation))
        assert allowed == set(train_mushroom.CODES[feature].values()), feature
