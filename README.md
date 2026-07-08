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
python3 -m pip install -r requirements-ml.txt
python3 scripts/extract_instances_v2.py --input goose_photos_square --output data/instances_v2
python3 scripts/mask_crops_v2.py --crops-metadata data/crops_v2_full/metadata.json --output data/masks_v2_from_crops --status pass
python3 scripts/build_current_mask_candidates.py --metadata data/masks_v2_from_crops/metadata.json --output data/font_candidates/goosetype_candidates.json
```

`extract_instances_v2.py` is the current extraction path. It detects goose-like boxes with OWL-ViT, runs a tiled detector pass for small birds in flock photos, segments each detected box with SAM, saves one crop/mask/silhouette per accepted bird, and writes an automatic contact sheet for visual inspection. It keeps clean single-bird instances and can synthesize a small number of overlapping two-bird `composite` instances for complex letters such as `M` or `W`.

Current extraction command:

```bash
python3 scripts/extract_instances_v2.py \
  --input goose_photos \
  --output data/instances_v2 \
  --stage full \
  --segmentation-backend sam \
  --fallback-backend rembg \
  --keep-rejected
```

OWL-ViT uses the public `google/owlvit-base-patch32` model through Hugging Face Transformers. SAM uses `facebook/sam-vit-base` as a box-prompted mask model. Neither requires paid API tokens, but both need local model weights in the Hugging Face cache. If SAM is too slow on a machine, `--segmentation-backend rembg` remains a faster fallback.

To validate detector crops before running masks:

```bash
python3 scripts/extract_instances_v2.py \
  --input goose_photos \
  --output data/crops_v2 \
  --stage crop
```

Useful extraction options:

```bash
python3 scripts/extract_instances_v2.py --segmentation-backend sam --fallback-backend rembg
python3 scripts/extract_instances_v2.py --segmentation-backend rembg
python3 scripts/extract_instances_v2.py --max-composites-per-image 0
python3 scripts/extract_instances_v2.py --detection-threshold 0.12 --tile-threshold 0.09
python3 scripts/extract_instances_v2.py --min-goose-score 0.65 --keep-rejected
```

After segmentation, each mask is scored for goose-like geometry before it is saved. The filter rejects common non-goose fragments such as sticks, reeds, tiny blobs, border chunks, very sparse branch-like shapes, and dense bush-like regions using aspect ratio, fill ratio, contour thinness, size, curvature, and border contact. `--keep-rejected` writes rejected masks to the extraction output folder so thresholds can be tuned visually.

`analyze_font.py` is the future import-font hook: it rasterizes selected letters from a `.ttf` or `.otf`, stores masks, and calculates the same feature family used for geese.

`select_candidates.py` avoids doing expensive mask comparison for every goose-letter pair. It first ranks candidates by cheap feature similarity such as aspect ratio, curvature, boldness/fill, and slant, then applies pixel IoU only to the narrowed shortlist.

## Public Dataset Notes

Public goose datasets can help, but they should be used as a curated reference library rather than dumped directly into the font generator. The useful sources are:

- user photos, because they match the visual style and body poses you actually want
- iNaturalist or GBIF observations with permissive licenses, because they add species and pose variety
- Wikimedia Commons images with compatible licenses, because attribution is usually explicit
- Roboflow Universe projects, if the dataset license allows reuse and the labels are actually goose-specific

The practical workflow is to keep external images in a separate folder such as `data/external_sources/`, store license/source metadata beside each image, run the same extraction pipeline into a separate processed folder, then merge only the good cutouts into the glyph candidate pool. This avoids mixing unknown-license images into exported fonts and keeps low-quality or mislabeled bird photos from dominating the letter matching.

The first supported external downloader uses iNaturalist's public API:

```bash
python3 scripts/fetch_inaturalist_geese.py \
  --taxon "Branta canadensis" \
  --limit 80 \
  --output data/external_sources/inaturalist_canada_goose

python3 scripts/extract_instances_v2.py \
  --input data/external_sources/inaturalist_canada_goose/photos \
  --output data/instances_v2_canada_goose \
  --segmentation-backend sam \
  --fallback-backend rembg \
  --keep-rejected
```

The downloader keeps `metadata.json` beside the downloaded photos with source URLs, observation IDs, photo IDs, license codes, and attribution strings. The default license filter keeps `cc0`, `cc-by`, and `cc-by-sa` photo records.

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
