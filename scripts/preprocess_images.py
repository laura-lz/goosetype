#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from PIL import Image
except ImportError as error:
    raise SystemExit("Pillow is required: python3 -m pip install Pillow") from error

from goosetype.image_features import (
    IMAGE_SUFFIXES,
    build_foreground_mask,
    component_mask,
    connected_components,
    crop_with_padding,
    cutout_from_mask,
    measure_mask,
    silhouette_from_mask,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Identify goose foreground regions, separate non-overlapping geese into instances, "
            "crop them, erase backgrounds, and write metadata."
        )
    )
    parser.add_argument("--input", default="goose_photos_square", help="Folder of raw goose images.")
    parser.add_argument("--output", default="data/processed", help="Processed data folder.")
    parser.add_argument("--threshold", type=float, default=54.0, help="Foreground threshold.")
    parser.add_argument("--min-area", type=int, default=2200, help="Smallest connected component to keep.")
    parser.add_argument("--padding", type=int, default=28, help="Pixels of padding around each extracted goose.")
    parser.add_argument("--include-scene-mask", action="store_true", help="Also keep whole-scene masks for debugging.")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    masks_dir = output_dir / "masks"
    cutouts_dir = output_dir / "cutouts"
    silhouettes_dir = output_dir / "silhouettes"
    debug_dir = output_dir / "debug_scene_masks"
    for directory in (masks_dir, cutouts_dir, silhouettes_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if args.include_scene_mask:
        debug_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    goose_index = 1
    for image_path in sorted(path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
        image = Image.open(image_path).convert("RGBA")
        scene_mask = build_foreground_mask(image, threshold=args.threshold)
        if args.include_scene_mask:
            scene_mask.save(debug_dir / f"{image_path.stem}_mask.png")

        components = connected_components(scene_mask, min_area=args.min_area)
        if not components:
            components = []
            bbox = scene_mask.getbbox()
            if bbox:
                area = sum(1 for value in scene_mask.getdata() if value > 127)
                components.append(type("ComponentLike", (), {"bbox": bbox, "area": area})())

        for component_index, component in enumerate(components, start=1):
            goose_id = f"goose_{goose_index:04d}"
            instance_mask = component_mask(scene_mask, component.bbox)
            cropped_image, cropped_mask, padded_bbox = crop_with_padding(image, instance_mask, component.bbox, args.padding)
            cutout = cutout_from_mask(cropped_image, cropped_mask)
            silhouette = silhouette_from_mask(cropped_mask)

            mask_path = masks_dir / f"{goose_id}.png"
            cutout_path = cutouts_dir / f"{goose_id}.png"
            silhouette_path = silhouettes_dir / f"{goose_id}.png"
            cropped_mask.save(mask_path)
            cutout.save(cutout_path)
            silhouette.save(silhouette_path)

            features = measure_mask(cropped_mask)
            x0, y0, x1, y1 = padded_bbox
            metadata.append(
                {
                    "id": goose_id,
                    "source_image": str(image_path),
                    "source_component_index": component_index,
                    "extraction_kind": "single_component",
                    "overlap_note": (
                        "Disconnected foreground components are separated. Overlapping geese remain grouped "
                        "unless a stronger segmentation model or manual mask supplies separate instances."
                    ),
                    "cutout_path": str(cutout_path),
                    "mask_path": str(mask_path),
                    "silhouette_path": str(silhouette_path),
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "width": cropped_image.width,
                    "height": cropped_image.height,
                    "aspect_ratio": features["aspect_ratio"],
                    "component_area": component.area,
                }
            )
            print(f"{goose_id}: {image_path.name} component={component_index} bbox={metadata[-1]['bbox']}")
            goose_index += 1

    output_path = output_dir / "metadata.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {len(metadata)} goose instances to {output_path}")


if __name__ == "__main__":
    main()

