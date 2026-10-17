from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app import ml
from app.schemas import ModelInfo, Mushroom, MushroomPrediction

MODELS: dict[str, dict | None] = {}
HOW_TO_TRAIN = {"mushroom": "python train_mushroom.py"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    for name in HOW_TO_TRAIN:
        MODELS[name] = ml.load(name)
    yield
    MODELS.clear()


app = FastAPI(
    title="Mushroom API",
    description="Is this mushroom edible or poisonous? (A teaching model: never eat on its word.)",
    version="1.0.0",
    lifespan=lifespan,
)


def model(name: str) -> dict:
    """The loaded model, or a clear error if it has not been trained yet."""
    bundle = MODELS.get(name)
    if bundle is None:
        hint = f"The {name} model is not trained yet: run `{HOW_TO_TRAIN[name]}`."
        raise HTTPException(status_code=503, detail=hint)
    return bundle


@app.get("/", include_in_schema=False)
def home() -> dict:
    return {"message": "Mushroom API", "docs": "/docs", "health": "/health"}


@app.get("/health", tags=["service"])
def health() -> dict:
    return {"status": "ok", "models": {name: bundle is not None for name, bundle in MODELS.items()}}


@app.get("/models", response_model=list[ModelInfo], tags=["service"])
def models() -> list[dict]:
    """Which models are loaded, how they were trained and how well they scored."""
    return [
        {"name": name, "loaded": False}
        if bundle is None
        else {
            "name": name,
            "loaded": True,
            "algorithm": bundle["algorithm"],
            "trained_at": bundle["trained_at"],
            "metrics": bundle["metrics"],
        }
        for name, bundle in MODELS.items()
    ]


@app.post("/mushroom/predict", response_model=MushroomPrediction, tags=["mushroom"])
def predict_mushroom(mushroom: Mushroom) -> dict:
    """Is this mushroom poisonous?"""
    return ml.predict_mushroom(model("mushroom"), [mushroom.model_dump()])[0]


@app.post("/mushroom/predict/batch", response_model=list[MushroomPrediction], tags=["mushroom"])
def predict_mushroom_batch(mushrooms: list[Mushroom]) -> list[dict]:
    """Several mushrooms in one call (one model call for the whole list)."""
    if not mushrooms:
        return []
    return ml.predict_mushroom(model("mushroom"), [m.model_dump() for m in mushrooms])
