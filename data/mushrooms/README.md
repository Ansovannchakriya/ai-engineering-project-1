# Mushroom data

The [UCI Mushroom dataset](https://archive.ics.uci.edu/dataset/73/mushroom) (also on Kaggle as
"Mushroom Classification"), used by `python train_mushroom.py` and
`notebooks/1_mushroom_model.ipynb`.

| File | Rows | edible / poisonous | Columns |
|---|---|---|---|
| `mushrooms.csv` | 8,124 | 4,208 / 3,916 | `class` + 22 features, every value one letter |

## Where it comes from

Descriptions of hypothetical samples of **23 species of gilled mushrooms** in the *Agaricus* and
*Lepiota* families, drawn from *The Audubon Society Field Guide to North American Mushrooms*
(1981). Each species is labelled edible or poisonous; the guide's "edibility unknown, not
recommended" was **merged into poisonous**. So the model learns that field guide, for those 23
species. It knows nothing about any other mushroom.

## Decoding

The training script decodes letters into words (`CODES` in `train_mushroom.py`), and the API
accepts the words. Column names change `-` to `_` (`cap-shape` → `cap_shape`).

| Column | Values (letter = word) |
|---|---|
| `class` | e = edible, p = poisonous → the target `poisonous` (1 / 0) |
| `cap_shape` | b = bell, c = conical, x = convex, f = flat, k = knobbed, s = sunken |
| `cap_surface` | f = fibrous, g = grooves, y = scaly, s = smooth |
| `cap_color` | n = brown, b = buff, c = cinnamon, g = gray, r = green, p = pink, u = purple, e = red, w = white, y = yellow |
| `bruises` | t = yes, f = no |
| `odor` | a = almond, l = anise, c = creosote, y = fishy, f = foul, m = musty, n = none, p = pungent, s = spicy |
| `gill_attachment` | a = attached, d = descending, f = free, n = notched |
| `gill_spacing` | c = close, w = crowded, d = distant |
| `gill_size` | b = broad, n = narrow |
| `gill_color` | k = black, n = brown, b = buff, h = chocolate, g = gray, r = green, o = orange, p = pink, u = purple, e = red, w = white, y = yellow |
| `stalk_shape` | e = enlarging, t = tapering |
| `stalk_root` | b = bulbous, c = club, u = cup, e = equal, z = rhizomorphs, r = rooted, ? = missing |
| `stalk_surface_above_ring`, `stalk_surface_below_ring` | f = fibrous, y = scaly, k = silky, s = smooth |
| `stalk_color_above_ring`, `stalk_color_below_ring` | n = brown, b = buff, c = cinnamon, g = gray, o = orange, p = pink, e = red, w = white, y = yellow |
| `veil_type` | p = partial, u = universal |
| `veil_color` | n = brown, o = orange, w = white, y = yellow |
| `ring_number` | n = none, o = one, t = two |
| `ring_type` | c = cobwebby, e = evanescent, f = flaring, l = large, n = none, p = pendant, s = sheathing, z = zone |
| `spore_print_color` | k = black, n = brown, b = buff, h = chocolate, r = green, o = orange, u = purple, w = white, y = yellow |
| `population` | a = abundant, c = clustered, n = numerous, s = scattered, v = several, y = solitary |
| `habitat` | g = grasses, l = leaves, m = meadows, p = paths, u = urban, w = waste, d = woods |

## Things to notice

- **`veil_type` never changes**: all 8,124 rows are `partial`. It is dropped, leaving 21 features.
- **`stalk_root` is missing** (`?`) in 2,480 rows (30 %). Kept as its own value, `missing`.
- **Some documented values never appear**: gill attachment `descending` / `notched`, gill
  spacing `distant`, stalk root `cup` / `rhizomorphs`, ring type `cobwebby` / `sheathing` /
  `zone`. The API accepts them (they are real answers) but reports them in `unseen_features`:
  the model has nothing to go on for those.
- **Odor nearly decides it alone**: almond and anise are always edible, every other smell always
  poisonous; only `none` is mixed (120 of 3,528 poisonous). Hence the 98.6 % "nose rule".

## Results (test set: 1,625 mushrooms, 783 poisonous)

| | Accuracy | Poisonous called edible |
|---|---|---|
| Always "edible" | 51.8 % | 783 |
| The nose rule (bad smell → poisonous) | 98.6 % | 23 |
| One-hot + random forest, safety threshold 0.3 (`train_mushroom.py`) | 100 % | 0 |
