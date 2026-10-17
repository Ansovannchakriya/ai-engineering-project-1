"""What the API accepts and returns. FastAPI validates requests against these and documents
them at /docs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# The words the model was trained on (`CODES` in train_mushroom.py; a test checks they match).
Surface = Literal["fibrous", "scaly", "silky", "smooth"]
StalkColor = Literal[
    "brown", "buff", "cinnamon", "gray", "orange", "pink", "red", "white", "yellow"
]  # fmt: skip

EXAMPLE_MUSHROOM = {
    "cap_shape": "convex",
    "cap_surface": "smooth",
    "cap_color": "brown",
    "bruises": "yes",
    "odor": "pungent",
    "gill_attachment": "free",
    "gill_spacing": "close",
    "gill_size": "narrow",
    "gill_color": "black",
    "stalk_shape": "enlarging",
    "stalk_root": "equal",
    "stalk_surface_above_ring": "smooth",
    "stalk_surface_below_ring": "smooth",
    "stalk_color_above_ring": "white",
    "stalk_color_below_ring": "white",
    "veil_color": "white",
    "ring_number": "one",
    "ring_type": "pendant",
    "spore_print_color": "black",
    "population": "scattered",
    "habitat": "urban",
}


class Mushroom(BaseModel):
    """What a mushroom looks (and smells) like: the 21 features the model was trained on."""

    model_config = {"extra": "forbid", "json_schema_extra": {"examples": [EXAMPLE_MUSHROOM]}}

    cap_shape: Literal["bell", "conical", "convex", "flat", "knobbed", "sunken"]
    cap_surface: Literal["fibrous", "grooves", "scaly", "smooth"]
    cap_color: Literal[
        "brown", "buff", "cinnamon", "gray", "green", "pink", "purple", "red", "white", "yellow"
    ]
    bruises: Literal["yes", "no"] = Field(description="Does it bruise when handled?")
    odor: Literal[
        "almond", "anise", "creosote", "fishy", "foul", "musty", "none", "pungent", "spicy"
    ]
    gill_attachment: Literal["attached", "descending", "free", "notched"]
    gill_spacing: Literal["close", "crowded", "distant"]
    gill_size: Literal["broad", "narrow"]
    gill_color: Literal[
        "black", "brown", "buff", "chocolate", "gray", "green", "orange", "pink", "purple", "red",
        "white", "yellow",
    ]  # fmt: skip
    stalk_shape: Literal["enlarging", "tapering"]
    stalk_root: Literal["bulbous", "club", "cup", "equal", "rhizomorphs", "rooted", "missing"] = (
        Field(description='"missing" when the root was not recorded')
    )
    stalk_surface_above_ring: Surface
    stalk_surface_below_ring: Surface
    stalk_color_above_ring: StalkColor
    stalk_color_below_ring: StalkColor
    veil_color: Literal["brown", "orange", "white", "yellow"]
    ring_number: Literal["none", "one", "two"]
    ring_type: Literal[
        "cobwebby", "evanescent", "flaring", "large", "none", "pendant", "sheathing", "zone"
    ]
    spore_print_color: Literal[
        "black", "brown", "buff", "chocolate", "green", "orange", "purple", "white", "yellow"
    ]
    population: Literal["abundant", "clustered", "numerous", "scattered", "several", "solitary"]
    habitat: Literal["grasses", "leaves", "meadows", "paths", "urban", "waste", "woods"]


class MushroomPrediction(BaseModel):
    poisonous_probability: float = Field(description="Probability that it is poisonous (0-1)")
    is_poisonous: bool = Field(description="poisonous_probability >= threshold")
    verdict: Literal["edible", "poisonous"]
    threshold: float = Field(description="The decision threshold chosen during training")
    unseen_features: list[str] = Field(
        description="Features whose value never appeared in training: treat the answer with care"
    )
    warning: str


class ModelInfo(BaseModel):
    name: str
    loaded: bool
    algorithm: str | None = None
    trained_at: str | None = None
    metrics: dict[str, float] = {}
