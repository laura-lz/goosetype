# GooseType

GooseType is a generative font engine that turns a folder of goose photos into various fonts. The same concept can be reapplied to other types of photos. 

The goal is for the following pipeline:

```text
raw goose photos
-> segmentation and cutouts
-> geometric feature extraction
-> glyph target comparison
-> 1-2 goose glyph composition
-> browser rendering
-> optional static font export
```

## Image And Font Pipeline

The analysis pipeline is split into practical scripts:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-segmentation.txt
python3 scripts/extract_geese.py --input goose_photos_square --output data/processed
python3 scripts/compute_features.py --metadata data/processed/metadata.json --output data/processed/features.json
python3 scripts/analyze_font.py --name arial --font "/System/Library/Fonts/Supplemental/Arial.ttf" --output data/processed/font_targets/arial.json
python3 scripts/select_candidates.py --features data/processed/features.json --font-targets data/processed/font_targets/arial.json --output data/processed/candidates.json
```

`extract_geese.py` identifies foreground goose regions, separates disconnected geese into individual cutouts, erases backgrounds, and keeps overlapping foreground groups together. It now prefers model-based background removal through `rembg` when available, falls back to OpenCV GrabCut, and only uses the old color-threshold heuristic as a last resort. If an overlapping scene has a fully visible front goose, a future stronger instance segmentation model or manual mask can provide that extra instance without changing the downstream metadata shape.

Useful extraction options:

```bash
python3 scripts/extract_geese.py --segmentation-backend rembg --rembg-model isnet-general-use
python3 scripts/extract_geese.py --segmentation-backend grabcut
python3 scripts/extract_geese.py --segmentation-backend heuristic
```

`analyze_font.py` is the future import-font hook: it rasterizes selected letters from a `.ttf` or `.otf`, stores masks, and calculates the same feature family used for geese.

`select_candidates.py` avoids doing expensive mask comparison for every goose-letter pair. It first ranks candidates by cheap feature similarity such as aspect ratio, curvature, boldness/fill, and slant, then applies pixel IoU only to the narrowed shortlist.

## Design Notes

GooseType treats each goose as a glyph candidate with measurable properties:

- aspect ratio and bounding box
- center of mass
- orientation / slant
- area and perimeter
- boldness and thinness
- curvature proxy
- left/right balance

Style presets work by changing matching weights. For example, an italic-ish family rewards slanted geese, while an abstract family tolerates looser glyph fit.
