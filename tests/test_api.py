"""Every endpoint: predictions, invalid input (422) and a missing model (503)."""

from app.main import MODELS
from app.schemas import EXAMPLE_MUSHROOM

POISONOUS = EXAMPLE_MUSHROOM  # row 1 of the data: pungent smell, black spore print
EDIBLE = EXAMPLE_MUSHROOM | {  # row 2 of the data: smells of almond
    "cap_color": "yellow",
    "odor": "almond",
    "gill_size": "broad",
    "stalk_root": "club",
    "spore_print_color": "brown",
    "population": "numerous",
    "habitat": "grasses",
}


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "models": {"mushroom": True}}


def test_models(client):
    [info] = client.get("/models").json()
    assert info["name"] == "mushroom"
    assert info["loaded"] is True
    assert info["metrics"]["poisonous_missed"] == 0


def test_predict_poisonous(client):
    response = client.post("/mushroom/predict", json=POISONOUS)
    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "poisonous"
    assert body["is_poisonous"] is True
    assert body["unseen_features"] == []
    assert "never eat" in body["warning"]


def test_predict_edible(client):
    body = client.post("/mushroom/predict", json=EDIBLE).json()
    assert body["verdict"] == "edible"
    assert body["poisonous_probability"] < body["threshold"]


def test_batch(client):
    response = client.post("/mushroom/predict/batch", json=[POISONOUS, EDIBLE])
    assert response.status_code == 200
    assert [p["verdict"] for p in response.json()] == ["poisonous", "edible"]
    assert client.post("/mushroom/predict/batch", json=[]).json() == []


def test_value_never_seen_in_training(client):
    # "notched" gills are a valid answer, but no mushroom in the data has them.
    body = client.post("/mushroom/predict", json=POISONOUS | {"gill_attachment": "notched"}).json()
    assert body["unseen_features"] == ["gill_attachment"]


def test_invalid_input(client):
    assert client.post("/mushroom/predict", json=POISONOUS | {"odor": "f"}).status_code == 422
    assert client.post("/mushroom/predict", json=POISONOUS | {"veil_type": "x"}).status_code == 422
    missing = {k: v for k, v in POISONOUS.items() if k != "odor"}
    assert client.post("/mushroom/predict", json=missing).status_code == 422


def test_model_not_trained(client, monkeypatch):
    monkeypatch.setitem(MODELS, "mushroom", None)
    response = client.post("/mushroom/predict", json=POISONOUS)
    assert response.status_code == 503
    assert "python train_mushroom.py" in response.json()["detail"]
