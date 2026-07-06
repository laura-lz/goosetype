#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from PIL import Image
except ImportError as error:
    raise SystemExit("Pillow is required: python3 -m pip install Pillow") from error

from goosetype.image_features import (
    IMAGE_SUFFIXES,
    build_detector_guided_mask,
    build_foreground_mask,
    component_mask,
    connected_components,
    crop_with_padding,
    cutout_from_mask,
    measure_mask,
    score_goose_candidate,
    silhouette_from_mask,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Identify goose foreground regions, separate non-overlapping geese into instances, "
            "crop them, erase backgrounds, and write metadata."
        )
    )
    parser.add_argument("--input", default="goose_photos", help="Folder of raw goose images.")
    parser.add_argument("--output", default="data/processed", help="Processed data folder.")
    parser.add_argument(
        "--detector-backend",
        choices=("none", "owlvit"),
        default="none",
        help="Optional semantic detector. owlvit finds goose boxes before segmentation.",
    )
    parser.add_argument(
        "--detection-query",
        action="append",
        default=None,
        help="Text prompt for detector-backed extraction. Repeat to add prompts.",
    )
    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=0.12,
        help="Minimum detector confidence for keeping a proposed goose box.",
    )
    parser.add_argument(
        "--detector-box-padding",
        type=float,
        default=0.08,
        help="Padding around detector boxes before per-box segmentation, as a ratio of box size.",
    )
    parser.add_argument(
        "--segmentation-backend",
        choices=("auto", "rembg", "grabcut", "heuristic"),
        default="auto",
        help="Foreground extraction backend. auto tries rembg, then GrabCut, then heuristic.",
    )
    parser.add_argument(
        "--rembg-model",
        default="isnet-general-use",
        help="rembg model/session name when --segmentation-backend uses rembg.",
    )
    parser.add_argument(
        "--max-segmentation-side",
        type=int,
        default=1600,
        help="Downscale longest image side for segmentation inference; 0 keeps original size.",
    )
    parser.add_argument("--threshold", type=float, default=54.0, help="Foreground threshold.")
    parser.add_argument("--min-area", type=int, default=2200, help="Smallest connected component to keep.")
    parser.add_argument("--min-goose-score", type=float, default=0.5, help="Minimum geometry score for keeping a component as a goose.")
    parser.add_argument("--keep-rejected", action="store_true", help="Save rejected non-goose components for tuning/debugging.")
    parser.add_argument("--resume", action="store_true", help="Continue from an existing metadata.json in the output folder.")
    parser.add_argument("--padding", type=int, default=28, help="Pixels of padding around each extracted goose.")
    parser.add_argument("--include-scene-mask", action="store_true", help="Also keep whole-scene masks for debugging.")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    masks_dir = output_dir / "masks"
    cutouts_dir = output_dir / "cutouts"
    silhouettes_dir = output_dir / "silhouettes"
    rejected_dir = output_dir / "rejected"
    debug_dir = output_dir / "debug_scene_masks"
    for directory in (masks_dir, cutouts_dir, silhouettes_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if args.keep_rejected:
        rejected_dir.mkdir(parents=True, exist_ok=True)
    if args.include_scene_mask:
        debug_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "metadata.json"
    metadata = []
    processed_sources = set()
    goose_index = 1
    if args.resume and output_path.exists():
        metadata = json.loads(output_path.read_text(encoding="utf-8"))
        processed_sources = {item["source_image"] for item in metadata}
        goose_index = next_goose_index(metadata)
        print(f"Resuming from {output_path}: {len(metadata)} geese, next id goose_{goose_index:04d}")

    for image_path in sorted(path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
        if str(image_path) in processed_sources:
            print(f"resume skip: {image_path.name}")
            continue
        print(f"processing: {image_path.name}")
        image = Image.open(image_path).convert("RGBA")
        detection_queries = tuple(args.detection_query or ["goose", "geese", "bird"])
        detections = []
        if args.detector_backend != "none":
            try:
                scene_mask, detections = build_detector_guided_mask(
                    image,
                    detector_backend=args.detector_backend,
                    detection_queries=detection_queries,
                    detection_threshold=args.detection_threshold,
                    segmentation_backend=args.segmentation_backend,
                    rembg_model=args.rembg_model,
                    max_segmentation_side=args.max_segmentation_side,
                    box_padding_ratio=args.detector_box_padding,
                )
                print(f"detector: {len(detections)} goose-like boxes")
            except Exception as error:
                print(f"detector fallback: {error}")
                scene_mask = build_foreground_mask(
                    image,
                    threshold=args.threshold,
                    backend=args.segmentation_backend,
                    rembg_model=args.rembg_model,
                    max_segmentation_side=args.max_segmentation_side,
                )
        else:
            scene_mask = build_foreground_mask(
                image,
                threshold=args.threshold,
                backend=args.segmentation_backend,
                rembg_model=args.rembg_model,
                max_segmentation_side=args.max_segmentation_side,
            )
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
            instance_mask = component_mask(scene_mask, component.bbox)
            cropped_image, cropped_mask, padded_bbox = crop_with_padding(image, instance_mask, component.bbox, args.padding)
            features = measure_mask(cropped_mask)
            candidate = score_goose_candidate(features, image.size, bbox=component.bbox)
            if candidate["goose_candidate_score"] < args.min_goose_score:
                if args.keep_rejected:
                    rejected_id = f"{image_path.stem}_component_{component_index:02d}"
                    cutout_from_mask(cropped_image, cropped_mask).save(rejected_dir / f"{rejected_id}.png")
                print(
                    f"skip: {image_path.name} component={component_index} "
                    f"score={candidate['goose_candidate_score']:.2f} "
                    f"reasons={','.join(candidate['goose_candidate_reasons']) or 'low_score'}"
                )
                continue

            goose_id = f"goose_{goose_index:04d}"
            cutout = cutout_from_mask(cropped_image, cropped_mask)
            silhouette = silhouette_from_mask(cropped_mask)

            mask_path = masks_dir / f"{goose_id}.png"
            cutout_path = cutouts_dir / f"{goose_id}.png"
            silhouette_path = silhouettes_dir / f"{goose_id}.png"
            cropped_mask.save(mask_path)
            cutout.save(cutout_path)
            silhouette.save(silhouette_path)

            x0, y0, x1, y1 = padded_bbox
            metadata.append(
                {
                    "id": goose_id,
                    "source_image": str(image_path),
                    "source_component_index": component_index,
                    "extraction_kind": "single_component",
                    "segmentation_backend": args.segmentation_backend,
                    "detector_backend": args.detector_backend,
                    "detection_queries": list(detection_queries) if args.detector_backend != "none" else [],
                    "detections": [
                        {"bbox": list(detection.bbox), "score": detection.score, "label": detection.label}
                        for detection in detections
                    ],
                    "rembg_model": args.rembg_model if args.segmentation_backend in {"auto", "rembg"} else None,
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
                    **candidate,
                }
            )
            print(f"{goose_id}: {image_path.name} component={component_index} bbox={metadata[-1]['bbox']}")
            goose_index += 1

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote {len(metadata)} goose instances to {output_path}")


def next_goose_index(metadata: list[dict]) -> int:
    max_index = 0
    for item in metadata:
        goose_id = str(item.get("id", ""))
        if goose_id.startswith("goose_"):
            try:
                max_index = max(max_index, int(goose_id.split("_", 1)[1]))
            except ValueError:
                pass
    return max_index + 1


if __name__ == "__main__":
    main()
