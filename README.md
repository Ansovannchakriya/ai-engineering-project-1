# Mushroom classification: build a model, serve it with FastAPI

Is this mushroom **edible or poisonous**? A model built with plain **scikit-learn** from the 21
things a field guide records about a mushroom (smell, cap, gills, stalk, ring, spore print,
habitat), saved to a file with **joblib** and served by a small **FastAPI** app.

Same structure and stack as the Plato churn & sentiment project: no MLflow, no database, no Docker, a few
hundred lines of Python you can read in one sitting.

> **A model is not a mycologist.** This is a classroom project on a classic dataset of 23
> species from one field guide. Never eat a wild mushroom on the strength of a prediction; the
> API says so in every answer.

```
project-1/
├── data/mushrooms/
│   ├── mushrooms.csv                 8,124 mushrooms, every value one letter
│   └── README.md                     where it comes from, the letter → word table
├── notebooks/
│   └── 1_mushroom_model.ipynb        build the model step by step, with plots
├── train_mushroom.py                 the same as the notebook, in one script
├── models/                           mushroom.joblib (created by training)
├── app/
│   ├── ml.py                         load the model, make predictions (no FastAPI here)
│   ├── schemas.py                    what the API accepts and returns (Pydantic)
│   └── main.py                       the FastAPI app: routes, loading the model at startup
├── tests/                            pytest: training quality + every endpoint
└── requests.http                     ready-made requests (VS Code REST Client, JetBrains)
```

## Setup

Python 3.11 or newer.

```bash
cd project-1
python3 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Part 1: build the model

Two ways to the same result. In class, use the notebook; to get the model quickly, run the
script.

```bash
jupyter lab notebooks             # step by step, with explanations and plots
```

```bash
python train_mushroom.py          # ~20 s, prints every step
```

Both write `models/mushroom.joblib`. It holds the fitted model **and** what the API needs to use
it: the feature order, the decision threshold, the scores, the training date and the
scikit-learn version.

| Step | What it teaches |
|---|---|
| 1. Load, decode, look | Every value is one letter (`p` is pungent, pink, pendant, paths *and* poisonous): decode them into words. `veil_type` has one value in every row, so it tells nothing: drop it. Smell nearly gives the answer away. |
| 2. Train / test split | Judge a model on data it has never seen; `stratify` keeps the share of poisonous equal. |
| 3. Baselines | "Always edible" scores 51.8 %. The one-line **nose rule** (bad smell → poisonous) scores 98.6 %, but still calls 23 poisonous mushrooms edible. Always try the simple rule first, and count the mistake that matters. |
| 4. Cross-validation | One-hot encoding inside a `Pipeline`, so the model takes words. Both models score ~100 % accuracy, so compare **log loss**: random forest 0.0007, logistic regression 0.0077. Is 100 % suspicious? |
| 5. Threshold | Mistakes are not equal: a miss is a poisoning, a false alarm a mushroom left behind. Threshold = half the lowest probability any poisonous mushroom got (cross-validated, never the test set): **0.3**. |
| 6. Test once | 783 of 783 poisonous caught, 0 false alarms out of 842 edible. |
| 7. Feature importance | Odor matters most, then gill size and spore print colour. Correlated features share the credit. |
| 8. Predict + what if | A pungent mushroom that smells of almond instead: the model is unsure (0.73), and unsure means poisonous. |
| 9. Save | `joblib.dump` a bundle, not only the model. |

Where the data comes from, and every letter code: [`data/mushrooms/README.md`](data/mushrooms/README.md).

## Part 2: serve it with FastAPI

```bash
fastapi dev app/main.py           # http://127.0.0.1:8000, reloads when the code changes
```

Open <http://127.0.0.1:8000/docs>: every endpoint, with an example request you can send from
the page ("Try it out"); the schema lists the allowed values of every feature. If port 8000 is
taken, add `--port 8010`.

| Method | Path | Does |
|---|---|---|
| GET | `/health` | Is the API up, is the model loaded |
| GET | `/models` | How the model was trained and how well it scored |
| POST | `/mushroom/predict` | One mushroom → probability of poison, verdict, unseen features |
| POST | `/mushroom/predict/batch` | A list of mushrooms in one call |

```bash
curl -s http://127.0.0.1:8000/mushroom/predict -H 'Content-Type: application/json' -d '{
  "cap_shape": "convex", "cap_surface": "smooth", "cap_color": "brown", "bruises": "yes",
  "odor": "pungent", "gill_attachment": "free", "gill_spacing": "close", "gill_size": "narrow",
  "gill_color": "black", "stalk_shape": "enlarging", "stalk_root": "equal",
  "stalk_surface_above_ring": "smooth", "stalk_surface_below_ring": "smooth",
  "stalk_color_above_ring": "white", "stalk_color_below_ring": "white", "veil_color": "white",
  "ring_number": "one", "ring_type": "pendant", "spore_print_color": "black",
  "population": "scattered", "habitat": "urban"
}'
# {"poisonous_probability":1.0,"is_poisonous":true,"verdict":"poisonous","threshold":0.3,
#  "unseen_features":[],"warning":"A model is not a mycologist: never eat a wild mushroom..."}
```

More requests, including invalid ones, are in [`requests.http`](requests.http).

> **Discuss with the class:** send `"gill_attachment": "notched"`. It is a real answer, so the
> API accepts it, but no mushroom in the data has notched gills: the one-hot encoder turns it
> into all zeros and the model simply cannot use it. The API reports it in `unseen_features`.
> What should a real service do with such a request: answer, refuse, or ask for a human?

### How the app is built

Three files, each with one job. Read them in this order:

1. [`app/ml.py`](app/ml.py): plain Python, no web code. `load()` reads a `.joblib` file,
   `predict_mushroom()` turns a list of mushrooms into a list of answers. You can use it from a
   notebook too.
2. [`app/schemas.py`](app/schemas.py): Pydantic models describe the JSON. Every feature is a
   `Literal` of the allowed words, so FastAPI **validates** requests (a letter code, a typo, a
   missing or unknown field → `422` listing the valid values, before the model ever sees it) and
   **documents** them at `/docs`. A test checks these words match the training script's `CODES`.
3. [`app/main.py`](app/main.py): the routes. Ideas worth pointing out in class:
   - **Load once, at startup.** The `lifespan` function loads the model when the app starts and
     every request reuses it. Loading per request would be slow.
   - **Thin routes.** Each route validates (schema in), calls `ml.py`, returns (schema out).
   - **Clear errors.** A model that was not trained yet answers `503` with the command to run.
   - **Batch endpoint** scores a whole list with one model call, much faster than one request
     per mushroom.

After retraining, restart the API: the model file is read at startup (`fastapi dev` reloads on
code changes, not on new model files).

For a real deployment, run without auto-reload and with several workers:

```bash
fastapi run app/main.py --workers 2
```

## Part 3: test it

```bash
pytest
```

The tests train the model into a temporary folder once (so they never touch `models/`), check
that it misses no poisonous mushroom on the test set, check that the API's allowed words match
the training data, and call every endpoint: predictions, a batch, an unseen value, invalid
input (`422`) and a missing model (`503`).

## Exercises

On the model (more at the end of the notebook):

1. Train on `odor` alone. How close does it get? Which mushrooms does it miss?
2. Remove `odor` and `spore_print_color`. Is the model still perfect? What takes over?
3. Train on 1 % of the training data. Does the safety threshold still catch every poisonous one?

On the API:

4. Make most features optional: a forager rarely knows all 21. How should missing ones be
   filled, and what does that do to the safety rule?
5. Refuse to say "edible" when `unseen_features` is not empty (answer `"verdict": "unsure"`).
6. Require an `X-API-Key` header (a FastAPI dependency) on the predict endpoints.
7. Write a `Dockerfile` that trains the model at build time and runs `fastapi run`.

## Good to know

- **Only load model files you trust.** joblib (like pickle) can run code when it loads a file.
- **Train and serve with the same scikit-learn version.** The bundle records the version and the
  API warns when it differs; models are not guaranteed to load across versions.
- **The feature order is part of the model.** The bundle stores it, and the API builds its
  DataFrame with exactly those columns.
- **Perfect on this data is not safe in a forest.** 23 species, one guide, and "edibility
  unknown" counted as poisonous.
- `models/*.joblib` are not committed: anyone can recreate them with the training script.
