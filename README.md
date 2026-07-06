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

For better extraction, install the optional free local ML stack and run a detector-guided pass:

```bash
python3 -m pip install -r requirements-ml.txt
python3 scripts/extract_geese.py \
  --input goose_photos \
  --output data/processed \
  --detector-backend owlvit \
  --segmentation-backend rembg \
  --detection-query goose \
  --detection-query geese \
  --detection-threshold 0.12 \
  --keep-rejected
```

`--detector-backend owlvit` uses the public `google/owlvit-base-patch32` model through Hugging Face Transformers. It does not require paid API tokens, but it does download model weights the first time it runs. The detector proposes goose-like boxes first, then the segmentation backend is applied inside those boxes. That keeps sticks, bushes, reeds, water texture, and other non-goose foreground from becoming candidates in the first place. If the optional detector dependencies are not installed, the script prints a fallback message and continues with the regular segmentation path.

Useful extraction options:

```bash
python3 scripts/extract_geese.py --segmentation-backend rembg --rembg-model isnet-general-use
python3 scripts/extract_geese.py --detector-backend owlvit --segmentation-backend rembg
python3 scripts/extract_geese.py --segmentation-backend grabcut
python3 scripts/extract_geese.py --segmentation-backend heuristic
python3 scripts/extract_geese.py --min-goose-score 0.65 --keep-rejected
```

After segmentation, each connected component is scored for goose-like geometry before it is saved. The filter rejects common non-goose fragments such as sticks, reeds, tiny blobs, border chunks, very sparse branch-like shapes, and dense bush-like regions using aspect ratio, fill ratio, contour thinness, size, curvature, and border contact. `--keep-rejected` writes rejected components to `data/processed/rejected/` so the threshold can be tuned visually.

`analyze_font.py` is the future import-font hook: it rasterizes selected letters from a `.ttf` or `.otf`, stores masks, and calculates the same feature family used for geese.

`select_candidates.py` avoids doing expensive mask comparison for every goose-letter pair. It first ranks candidates by cheap feature similarity such as aspect ratio, curvature, boldness/fill, and slant, then applies pixel IoU only to the narrowed shortlist.

## Public Dataset Notes

Public goose datasets can help, but they should be used as a curated reference library rather than dumped directly into the font generator. The useful sources are:

- user photos, because they match the visual style and body poses you actually want
- iNaturalist or GBIF observations with permissive licenses, because they add species and pose variety
- Wikimedia Commons images with compatible licenses, because attribution is usually explicit
- Roboflow Universe projects, if the dataset license allows reuse and the labels are actually goose-specific

The practical workflow is to keep external images in a separate folder such as `data/external_sources/`, store license/source metadata beside each image, run the same extraction pipeline into a separate processed folder, then merge only the good cutouts into the glyph candidate pool. This avoids mixing unknown-license images into exported fonts and keeps low-quality or mislabeled bird photos from dominating the letter matching.

For letter semblance, public datasets are more valuable for breadth than for accuracy by themselves. More geese gives the optimizer more silhouettes that happen to resemble `A`, `S`, `R`, italic strokes, bowl shapes, etc. The ranking should still compare against the selected font targets using features first, then mask similarity on the shortlist. In other words: use public data to increase the search space, not to replace the font-similarity scoring.

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
