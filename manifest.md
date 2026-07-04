# GooseType Agent Manifest

## Goal

Build GooseType: a generative font engine that turns segmented goose images into typographic glyphs. The system should preprocess goose photos, isolate individual geese, calculate visual properties, match geese to letter targets, and render text in styles such as readable, abstract, silhouette, photo-cutout, italic, bold, and cursive-like.

The first milestone is a working interactive renderer, not perfect OpenType export.

## Core Concept

GooseType is a pipeline:

```text
raw photos
-> goose segmentation
-> individual goose extraction
-> feature extraction
-> glyph matching and optimization
-> text rendering
-> optional static font export
```

The font styles should emerge from optimization weights and filters over the goose dataset. A default family can be exported later by freezing a set of style settings and generated glyph compositions.

## Current Milestone

Prioritize a usable engine-shaped prototype:

1. Use the existing goose photos as the seed dataset.
2. Extract simple mask and shape features.
3. Render each letter with one goose by default, two geese when helpful.
4. Let users tune literal/abstract, bold/thin, slant/italic, width, and render mode.
5. Compare goose features against reference glyph features inspired by a neutral sans font such as Arial, Helvetica, Inter, or DejaVu Sans.

## Processed Data Contract

Each extracted goose should produce:

```json
{
  "id": "goose_0001",
  "source_image": "2026-05-02.png",
  "cutout_path": "data/processed/cutouts/goose_0001.png",
  "mask_path": "data/processed/masks/goose_0001.png",
  "silhouette_path": "data/processed/silhouettes/goose_0001.png",
  "bbox": [120, 182, 690, 510],
  "width": 690,
  "height": 510,
  "aspect_ratio": 1.35
}
```

Feature extraction should add:

```json
{
  "id": "goose_0001",
  "area": 80320,
  "perimeter": 2104,
  "solidity": 0.71,
  "orientation_angle": -12.4,
  "center_of_mass": [0.51, 0.48],
  "major_axis": 0.82,
  "minor_axis": 0.31,
  "curvature_score": 0.67,
  "thinness_score": 0.42,
  "boldness_score": 0.58,
  "slant_score": -0.22,
  "pose_label": "unknown"
}
```

## Style Controls

The minimum style config:

```python
StyleConfig(
    readability=0.8,
    abstraction=0.2,
    boldness=0.5,
    slant=0.0,
    width=0.5,
    cursive=0.0,
    max_geese_per_glyph=2,
    render_mode="photo",
)
```

Important early presets:

- Readable Regular
- Abstract
- Silhouette
- Photo Cutout
- Italic-ish
- Bold-ish
- Cursive-ish

These do not need to be perfect font styles. They should be visible changes in the matching and rendering behavior.

## Glyph Optimizer

Start with randomized or deterministic weighted search:

```python
generate_glyph(letter, pose_library, style, seed=None) -> GlyphComposition
```

Scoring should combine:

- target fit
- outside-target penalty
- feature similarity
- slant match
- aspect ratio match
- boldness / thickness match
- curvature match
- transform penalty
- dataset diversity bonus

The initial web prototype may approximate glyph targets with hand-authored letter feature profiles. A later pass should rasterize uppercase A-Z from a neutral sans font and compute real target masks.

