# GooseType

GooseType is a generative font engine that turns a folder of goose photos into a living type family. The first milestone is a working interactive renderer: type some text, move style controls, and watch the engine choose which goose best fits each letter.

The long-term shape is:

```text
raw goose photos
-> segmentation and cutouts
-> geometric feature extraction
-> glyph target comparison
-> 1-2 goose glyph composition
-> browser rendering
-> optional static font export
```

This repository currently includes the photo dataset plus a lightweight prototype scaffold:

- `web/`: dependency-free browser renderer using the existing square goose photos.
- `goosetype/`: Python engine primitives for style config, features, targets, pose libraries, and glyph generation.
- `scripts/`: command-line entry points for preprocessing, feature computation, glyph target generation, and debug glyph composition.
- `data/processed/`: generated masks, cutouts, silhouettes, glyph targets, and metadata.
- `outputs/`: debug renders and future font exports.

## Try the Prototype

From the repo root:

```bash
python3 -m http.server 5173
```

Then open:

```text
http://localhost:5173/web/
```

The app has controls for readable/abstract, photo/silhouette/contour, boldness, slant, width, cursive flow, and 1-2 geese per letter. It estimates simple visual features in the browser from each photo and uses those features to pick and transform geese for each character.

## Generate Metadata

The scripts are intentionally practical and conservative. They use Pillow when available; if Pillow is missing, install it in your environment and rerun.

```bash
python3 scripts/preprocess_images.py --input goose_photos_square --output data/processed
python3 scripts/compute_features.py --metadata data/processed/metadata.json --output data/processed/features.json
python3 scripts/generate_glyphs.py --features data/processed/features.json --text GOOSE
```

`preprocess_images.py` starts with a center-weighted brightness/saturation segmentation heuristic. It is not pretending to be a perfect matting model; it creates useful first-pass masks, cutouts, silhouettes, and metadata so the rest of the engine can be exercised quickly. Manual cleanup or a stronger segmentation backend can replace it later without changing the rest of the pipeline.

## Design Notes

GooseType should feel like a type lab, not a static novelty alphabet. A goose is treated as a glyph candidate with measurable properties:

- aspect ratio and bounding box
- center of mass
- orientation / slant
- area and perimeter
- boldness and thinness
- curvature proxy
- left/right balance

Style presets work by changing matching weights. For example, an italic-ish family rewards slanted geese, while an abstract family tolerates looser glyph fit and bolder transforms.

