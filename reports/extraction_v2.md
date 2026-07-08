# GooseType Extraction v2

Date: 2026-07-08

The old processed datasets, candidate JSON, review annotations, and generated reports were removed because they were built from a detector-first pipeline that repeatedly selected the same visible bird in multi-bird photos and missed other birds.

## Current v2 Pipeline

Script: `scripts/extract_instances_v2.py`

1. Dedupe raw source images by filename and size.
2. Run goose-like detection on the full image.
3. Run additional tiled detection to catch small birds in flock photos.
4. Merge duplicate boxes with bbox NMS.
5. Segment each detection crop independently with SAM, falling back to rembg if needed.
6. Refine each single-instance mask to one connected component.
7. Reject tiny masks, weak goose-shaped masks, and masks poorly overlapping their detection box.
8. Suppress only strong duplicates with separate bbox and full-mask IoU thresholds.
9. Optionally synthesize a small number of two-bird `composite` instances from overlapping or near-touching accepted detections.
10. Write one cropped image, one mask, and one silhouette per accepted single/composite instance.
11. Write `metadata.json`, `features.json`, `summary.json`, and `contact_sheet.png`.

## Crop-Only Stage

Use `--stage crop` to stop after OWL-ViT detection and bbox cropping. This writes `crops/`, `metadata.json`, `summary.json`, and `contact_sheet.png`, but skips SAM/rembg masking and feature extraction. It is intended for validating whether the detector is finding one goose per crop before spending time on masks.

## Improvements Over The Old Architecture

- Detection and masking are now separate stages. The crop stage can be audited before running SAM.
- The detector runs on source photos directly and saves every accepted bbox crop, instead of collapsing the photo into one foreground blob first.
- Full-image detection and tiled detection are configurable separately; broad crop inventories can run with `--tile-size 0`, while small-bird flock mining can be a targeted second pass.
- Duplicate handling is explicit: strong duplicate boxes/masks are removed, while overlapping candidates can be kept as separate crops or later promoted into composites.
- Composite outputs are tagged with `instance_kind: composite`, so two-bird letter candidates do not pollute the clean single-bird pool.
- Generated review sheets are paginated for large runs.
- The old manual-review and processed-folder pipeline has been removed from scripts/docs so future runs do not accidentally use stale data.

## Full Crop Pass

Command shape:

```bash
python3 -u scripts/extract_instances_v2.py \
  --input data/external_sources/.../photos \
  --output data/crops_v2_full \
  --stage crop \
  --tile-size 0 \
  --max-segmentation-side 1000
```

Result:

- Unique source images after dedupe: 461
- Crop boxes written: 2,203
- Source images with at least one crop: 373
- Source images with zero crops: 88
- Largest single-photo crop count: 62
- Paginated review sheets: `data/crops_v2_full/contact_sheet.png` through `data/crops_v2_full/contact_sheet_019.png`
- Sheet manifest: `data/crops_v2_full/contact_sheets.json`

Dataset contribution by source folder:

- `inaturalist_curated_flying_geese`: 1,160
- `inaturalist_barnacle_goose`: 242
- `inaturalist_flying_snow_goose`: 172
- `inaturalist_white_fronted_goose`: 167
- `inaturalist_snow_goose`: 118
- `inaturalist_flying_canada_goose`: 105
- `inaturalist_canada_goose`: 102
- `inaturalist_flying_greylag_goose`: 60
- `inaturalist_greylag_goose`: 56
- `inaturalist_flying_barnacle_goose`: 10
- `inaturalist_flying_brant`: 8
- `inaturalist_flying_white_fronted_goose`: 3

Interpretation: the crop stage is now broad enough to collect a lot of flying/letterlike material before masking. It also intentionally includes noisy crops, tiny distant birds, and some non-goose waterfowl from mixed source sets; the next step should be crop quality scoring/filtering before running SAM on all 2,203 crops.

## Crop Quality Filter

The crop filter is currently metadata-only: it labels crops as `pass`, `review`, or `reject` but does not delete the broad crop inventory. This keeps odd but letterlike crops available while still preventing tiny/body-part/edge-clipped crops from going straight into masking.

Signals:

- crop area as a fraction of the original source photo
- minimum crop side in pixels
- detector confidence
- number of crop sides touching the original photo boundary
- crop aspect ratio
- detector-box area as a fraction of the source photo

Current thresholds:

- `tiny_crop`: crop area is less than 1% of the source image
- `small_crop`: crop area is between 1% and 1.8% of the source image
- `reject`: score below 0.42, very low detector score, or tiny crop
- `review`: score below 0.72, any image-edge contact, low detector score, or small crop
- `pass`: large enough, not edge-clipped, and detector confidence is not low

After applying this to `data/crops_v2_full`:

- `pass`: 610
- `review`: 571
- `reject`: 1,022

The user-flagged examples from the first review sheet are now all either `review` or `reject`; none remain `pass`. The filter is intentionally conservative around edge-clipped crops because some edge crops may still be useful for abstract/composite glyphs.

Output: `data/crops_v2_full/crop_quality_summary.json`

## Crop Masking Pass

Script: `scripts/mask_crops_v2.py`

This is the first masking stage that consumes the crop inventory directly instead of rerunning detection. For each crop record, it:

1. Opens the saved crop image.
2. Converts the original detector box into crop-local coordinates.
3. Builds a rough foreground support mask with the fallback backend, usually rembg.
4. Runs SAM with the crop-local detector box plus support mask.
5. Refines the result to one connected component.
6. Saves a mask, transparent cutout, silhouette, metadata/features, contact sheet, and diagnostic sheet.

The diagnostic sheets show crop / cutout / silhouette side by side, which is much more useful than reviewing silhouettes alone.

Larger bounded run:

- Output: `data/masks_v2_pass100`
- Input: first 100 `pass` crops
- Accepted: 99
- Rejected: 1 (`crop_00263`)
- Mask review statuses among accepted masks:
  - `pass`: 95
  - `review`: 4

Second bounded run:

- Output: `data/masks_v2_pass100_b`
- Input: second 100 `pass` crops, using `--skip-items 100`
- Accepted: 100
- Rejected: 0
- Mask review statuses among accepted masks:
  - `pass`: 97
  - `review`: 3

Combined progress:

- `pass` crops available from crop stage: 610
- Accepted masked pass crops so far: 199
- Remaining unmasked `pass` crops: 411
- Unmasked `review` crops: 571
- Unmasked `reject` crops: 1,022

Frontend preview candidate JSON:

- Output: `data/font_candidates/goosetype_candidates.json`
- Source mask batches: `data/masks_v2_pass100`, `data/masks_v2_pass100_b`
- Current frontend assets: 192 `mask_review_status == pass` masks
- Letters covered: uppercase and lowercase A-Z/a-z

Current read: SAM plus a rough support mask is good enough to proceed, but not perfect. It still produces some overfilled/blob silhouettes when the crop itself is blurred, low-detail, or contains multiple/partial birds. The second batch has many strong flying silhouettes and is probably the most useful material for angular/letterlike glyphs.

Known hard problem: semantic completeness. Some masks look mechanically clean but are not useful because the crop only contains body parts, misses the head/lower body, or includes a pose that is ambiguous even to a human. Simple geometry can catch edge-truncated masks with `mask_completeness_status`, but it cannot reliably prove “head and body are both present.” The best next options are:

- add a lightweight manual review UI for `pass`/`review`/`reject` labels on diagnostic triplets
- train or use a small pose/completeness classifier later, with labels such as `full_body`, `head_missing`, `body_part`, `multi_bird`, `good_composite`
- run letter matching only on `mask_review_status == pass` first, then optionally include reviewed masks for abstract/composite glyphs

## Next Required Improvement

The smoke test shows v2 can extract multiple instances from one image and preserve overlapping two-bird shapes as explicit composites, but some single masks can still include partial neighboring bodies. The next quality jump should be to add stronger partial-body/neighbor-contact scoring and then run a larger extraction pass over the full external source library.
