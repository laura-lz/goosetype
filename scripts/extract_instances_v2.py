#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw, ImageFont, ImageOps

from goosetype.image_features import (
    IMAGE_SUFFIXES,
    Detection,
    bbox_iou,
    build_instance_mask,
    clamp_bbox,
    connected_components,
    crop_with_padding,
    cutout_from_mask,
    detect_goose_boxes,
    mask_quality_metrics,
    measure_mask,
    pad_bbox,
    refine_instance_mask,
    score_goose_candidate,
    silhouette_from_mask,
)


DEFAULT_QUERIES = (
    "a goose",
    "a single goose",
    "a flying goose",
    "a goose bird",
    "a gosling",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "GooseType v2 extraction: dedupe source photos, find many goose-like boxes with full-image "
            "and tiled OWL-ViT passes, segment each box independently, suppress duplicate masks, and "
            "write one cropped bird instance per output record."
        )
    )
    parser.add_argument("--input", action="append", required=True, help="Raw image folder. Repeat for several datasets.")
    parser.add_argument("--output", default="data/instances_v2", help="Fresh output folder.")
    parser.add_argument("--stage", choices=("full", "crop"), default="full", help="Run full mask extraction or detection/crop only.")
    parser.add_argument("--query", action="append", default=None, help="Detection prompt. Repeat to override defaults.")
    parser.add_argument("--detection-threshold", type=float, default=0.09)
    parser.add_argument("--tile-threshold", type=float, default=0.07)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--tile-overlap", type=float, default=0.35)
    parser.add_argument("--max-detections-per-image", type=int, default=80)
    parser.add_argument("--box-nms-iou", type=float, default=0.38)
    parser.add_argument("--duplicate-mask-iou", type=float, default=0.68)
    parser.add_argument("--duplicate-box-iou", type=float, default=0.68)
    parser.add_argument("--composite-box-iou", type=float, default=0.08)
    parser.add_argument("--composite-mask-iou", type=float, default=0.02)
    parser.add_argument("--composite-gap-ratio", type=float, default=0.22)
    parser.add_argument("--max-composites-per-image", type=int, default=4)
    parser.add_argument("--box-padding", type=float, default=0.16)
    parser.add_argument("--crop-padding", type=int, default=28)
    parser.add_argument("--min-box-side", type=int, default=20)
    parser.add_argument("--min-mask-area", type=int, default=180)
    parser.add_argument("--min-mask-box-overlap", type=float, default=0.12)
    parser.add_argument("--min-goose-score", type=float, default=0.25)
    parser.add_argument("--segmentation-backend", choices=("sam", "rembg", "grabcut", "heuristic", "auto"), default="sam")
    parser.add_argument("--fallback-backend", choices=("rembg", "grabcut", "heuristic", "auto", "none"), default="rembg")
    parser.add_argument("--sam-model", default="facebook/sam-vit-base")
    parser.add_argument("--rembg-model", default="isnet-general-use")
    parser.add_argument("--max-segmentation-side", type=int, default=1400)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--keep-rejected", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    dirs = prepare_output_dirs(output, keep_rejected=args.keep_rejected)
    metadata: list[dict] = []
    source_paths = discover_unique_images([Path(path) for path in args.input])
    if args.max_images:
        source_paths = source_paths[: args.max_images]

    queries = tuple(args.query or DEFAULT_QUERIES)
    for source_index, source_path in enumerate(source_paths, start=1):
        print(f"[{source_index}/{len(source_paths)}] {source_path}")
        try:
            image = ImageOps.exif_transpose(Image.open(source_path)).convert("RGBA")
        except Exception as error:
            print(f"  skip unreadable: {error}")
            continue

        detections = propose_detections(image, queries=queries, args=args)
        print(f"  detections after box NMS: {len(detections)}")
        accepted_for_image: list[dict] = []
        for detection_index, detection in enumerate(detections, start=1):
            if args.stage == "crop":
                candidate = crop_detection(
                    image=image,
                    source_path=source_path,
                    detection=detection,
                    detection_index=detection_index,
                    args=args,
                )
                if is_duplicate_crop(candidate, accepted_for_image, args.duplicate_box_iou):
                    continue
                accepted_for_image.append(candidate)
                continue

            candidate = segment_detection(
                image=image,
                source_path=source_path,
                detection=detection,
                detection_index=detection_index,
                args=args,
            )
            if candidate is None:
                continue
            if is_duplicate_mask(candidate, accepted_for_image, args):
                maybe_save_rejected(candidate, dirs, reason="duplicate_mask", keep=args.keep_rejected)
                continue
            accepted_for_image.append(candidate)

        if args.stage == "full":
            accepted_for_image.extend(synthesize_composites(image, accepted_for_image, args))
        for candidate in accepted_for_image:
            if args.stage == "crop":
                record = save_crop_instance(candidate, dirs, len(metadata) + 1)
            else:
                record = save_instance(candidate, image, dirs, len(metadata) + 1)
            metadata.append(record)
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"  accepted: {len(accepted_for_image)}")

    if args.stage == "full":
        features = []
        for record in metadata:
            features.append({**record, **measure_mask(Image.open(record["mask_path"]).convert("L"))})
        (output / "features.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "schema": "goosetype-instances-v2-summary",
                "stage": args.stage,
                "source_images": len(source_paths),
                "instances": len(metadata),
                "queries": queries,
                "output": str(output),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(metadata)} instances to {output}")
    build_contact_sheet(output, metadata)


def prepare_output_dirs(output: Path, keep_rejected: bool) -> dict[str, Path]:
    dirs = {
        "output": output,
        "masks": output / "masks",
        "cutouts": output / "cutouts",
        "crops": output / "crops",
        "silhouettes": output / "silhouettes",
        "debug": output / "debug",
        "rejected": output / "rejected",
    }
    for key in ("masks", "cutouts", "crops", "silhouettes", "debug"):
        dirs[key].mkdir(parents=True, exist_ok=True)
    if keep_rejected:
        dirs["rejected"].mkdir(parents=True, exist_ok=True)
    return dirs


def discover_unique_images(input_dirs: list[Path]) -> list[Path]:
    seen: set[str] = set()
    paths: list[Path] = []
    for directory in input_dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            key = image_identity(path)
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)
    return paths


def image_identity(path: Path) -> str:
    stat = path.stat()
    return f"{path.name}|{stat.st_size}"


def propose_detections(image: Image.Image, queries: tuple[str, ...], args) -> list[Detection]:
    proposals: list[Detection] = []
    proposals.extend(
        detect_goose_boxes(
            image,
            backend="owlvit",
            queries=queries,
            threshold=args.detection_threshold,
            max_side=args.max_segmentation_side,
        )
    )
    for tile_box in tile_boxes(image.size, args.tile_size, args.tile_overlap):
        tile = image.crop(tile_box)
        tile_detections = detect_goose_boxes(
            tile,
            backend="owlvit",
            queries=queries,
            threshold=args.tile_threshold,
            max_side=args.max_segmentation_side,
        )
        for detection in tile_detections:
            x0, y0, x1, y1 = detection.bbox
            tx0, ty0, _, _ = tile_box
            proposals.append(
                Detection(
                    bbox=clamp_bbox((x0 + tx0, y0 + ty0, x1 + tx0, y1 + ty0), image.size),
                    score=detection.score,
                    label=f"tile:{detection.label}",
                )
            )

    filtered = []
    for detection in sorted(proposals, key=lambda item: item.score, reverse=True):
        width = detection.bbox[2] - detection.bbox[0]
        height = detection.bbox[3] - detection.bbox[1]
        if min(width, height) < args.min_box_side:
            continue
        if any(bbox_iou(detection.bbox, kept.bbox) >= args.box_nms_iou for kept in filtered):
            continue
        filtered.append(detection)
        if len(filtered) >= args.max_detections_per_image:
            break
    return filtered


def tile_boxes(size: tuple[int, int], tile_size: int, overlap: float) -> list[tuple[int, int, int, int]]:
    width, height = size
    if tile_size <= 0 or max(width, height) <= tile_size:
        return []
    stride = max(1, int(round(tile_size * (1 - overlap))))
    xs = list(range(0, max(1, width - tile_size + 1), stride))
    ys = list(range(0, max(1, height - tile_size + 1), stride))
    if not xs or xs[-1] != max(0, width - tile_size):
        xs.append(max(0, width - tile_size))
    if not ys or ys[-1] != max(0, height - tile_size):
        ys.append(max(0, height - tile_size))
    boxes = []
    for y in ys:
        for x in xs:
            boxes.append((x, y, min(width, x + tile_size), min(height, y + tile_size)))
    return boxes


def segment_detection(image: Image.Image, source_path: Path, detection: Detection, detection_index: int, args) -> dict | None:
    crop_box = pad_bbox(detection.bbox, image.size, args.box_padding)
    crop = image.crop(crop_box)
    local_box = (
        detection.bbox[0] - crop_box[0],
        detection.bbox[1] - crop_box[1],
        detection.bbox[2] - crop_box[0],
        detection.bbox[3] - crop_box[1],
    )
    mask = build_mask_with_fallback(crop, local_box, args)
    mask = refine_instance_mask(mask, min_island_area=max(24, args.min_mask_area // 12))
    if not mask.getbbox():
        return None
    area = mask_area(mask)
    if area < args.min_mask_area:
        return None

    local_bbox = mask.getbbox()
    assert local_bbox is not None
    global_bbox = (
        crop_box[0] + local_bbox[0],
        crop_box[1] + local_bbox[1],
        crop_box[0] + local_bbox[2],
        crop_box[1] + local_bbox[3],
    )
    if bbox_iou(global_bbox, detection.bbox) < args.min_mask_box_overlap:
        return None

    full_mask = Image.new("L", image.size, 0)
    full_mask.paste(mask, crop_box)
    cropped_image, cropped_mask, padded_bbox = crop_with_padding(image, full_mask, global_bbox, args.crop_padding)
    features = measure_mask(cropped_mask)
    goose_score = score_goose_candidate(features, image.size, bbox=global_bbox)
    quality = mask_quality_metrics(cropped_mask)
    if goose_score["goose_candidate_score"] < args.min_goose_score:
        return None

    return {
        "source_path": source_path,
        "detection": detection,
        "detection_index": detection_index,
        "crop_box": crop_box,
        "global_bbox": global_bbox,
        "padded_bbox": padded_bbox,
        "full_mask": full_mask,
        "cropped_image": cropped_image,
        "cropped_mask": cropped_mask,
        "features": features,
        "goose_score": goose_score,
        "mask_quality": quality,
        "mask_area": area,
        "instance_kind": "single",
        "member_detection_indices": [detection_index],
    }


def crop_detection(image: Image.Image, source_path: Path, detection: Detection, detection_index: int, args) -> dict:
    padded_bbox = pad_bbox(detection.bbox, image.size, args.box_padding)
    crop = image.crop(padded_bbox)
    quality = score_crop_quality(
        source_size=image.size,
        detection_bbox=detection.bbox,
        crop_box=padded_bbox,
        detection_score=detection.score,
    )
    return {
        "source_path": source_path,
        "source_size": image.size,
        "detection": detection,
        "detection_index": detection_index,
        "crop_box": padded_bbox,
        "padded_bbox": padded_bbox,
        "cropped_image": crop,
        "crop_quality": quality,
        "instance_kind": "crop",
        "member_detection_indices": [detection_index],
    }


def score_crop_quality(
    source_size: tuple[int, int],
    detection_bbox: tuple[int, int, int, int],
    crop_box: tuple[int, int, int, int],
    detection_score: float,
) -> dict:
    source_width, source_height = source_size
    source_area = max(1, source_width * source_height)
    dx0, dy0, dx1, dy1 = detection_bbox
    cx0, cy0, cx1, cy1 = crop_box
    detection_width = max(1, dx1 - dx0)
    detection_height = max(1, dy1 - dy0)
    crop_width = max(1, cx1 - cx0)
    crop_height = max(1, cy1 - cy0)
    detection_area_fraction = detection_width * detection_height / source_area
    crop_area_fraction = crop_width * crop_height / source_area
    min_side_fraction = min(crop_width / max(1, source_width), crop_height / max(1, source_height))
    aspect_ratio = crop_width / crop_height
    edge_sides = int(cx0 <= 0) + int(cy0 <= 0) + int(cx1 >= source_width) + int(cy1 >= source_height)

    score = 1.0
    reasons: list[str] = []
    if crop_area_fraction < 0.01:
        score -= 0.45
        reasons.append("tiny_crop")
    elif crop_area_fraction < 0.018:
        score -= 0.25
        reasons.append("small_crop")
    if min(crop_width, crop_height) < 80:
        score -= 0.25
        reasons.append("small_pixel_side")
    elif min(crop_width, crop_height) < 120:
        score -= 0.12
        reasons.append("modest_pixel_side")
    if detection_score < 0.13:
        score -= 0.4
        reasons.append("very_low_detector_score")
    elif detection_score < 0.22:
        score -= 0.22
        reasons.append("low_detector_score")
    if edge_sides >= 2:
        score -= 0.3
        reasons.append("multiple_crop_edges")
    elif edge_sides == 1:
        score -= 0.16
        reasons.append("touches_crop_edge")
    if aspect_ratio < 0.32 or aspect_ratio > 4.8:
        score -= 0.16
        reasons.append("extreme_aspect_ratio")
    if detection_area_fraction < 0.0045:
        score -= 0.18
        reasons.append("tiny_detector_box")

    score = max(0.0, min(1.0, score))
    if score < 0.42 or "very_low_detector_score" in reasons or "tiny_crop" in reasons:
        status = "reject"
    elif score < 0.72 or edge_sides > 0 or "low_detector_score" in reasons or "small_crop" in reasons:
        status = "review"
    else:
        status = "pass"

    return {
        "crop_quality_score": round(score, 4),
        "crop_quality_status": status,
        "crop_quality_reasons": reasons,
        "source_width": source_width,
        "source_height": source_height,
        "detection_area_fraction": round(detection_area_fraction, 6),
        "crop_area_fraction": round(crop_area_fraction, 6),
        "crop_min_side_fraction": round(min_side_fraction, 6),
        "crop_aspect_ratio": round(aspect_ratio, 4),
        "crop_edge_sides": edge_sides,
    }


def build_mask_with_fallback(crop: Image.Image, local_box: tuple[int, int, int, int], args) -> Image.Image:
    try:
        return build_instance_mask(
            crop,
            bbox=local_box,
            backend=args.segmentation_backend,
            sam_model=args.sam_model,
            rembg_model=args.rembg_model,
            max_segmentation_side=args.max_segmentation_side,
        )
    except Exception as error:
        if args.fallback_backend == "none":
            raise
        print(f"    segmentation fallback ({args.segmentation_backend} -> {args.fallback_backend}): {error}")
        return build_instance_mask(
            crop,
            bbox=local_box if args.fallback_backend == "sam" else None,
            backend=args.fallback_backend,
            rembg_model=args.rembg_model,
            max_segmentation_side=args.max_segmentation_side,
        )


def is_duplicate_mask(candidate: dict, accepted: list[dict], args) -> bool:
    for existing in accepted:
        if bbox_iou(candidate["global_bbox"], existing["global_bbox"]) >= args.duplicate_box_iou:
            return True
        if mask_iou(candidate["full_mask"], existing["full_mask"]) >= args.duplicate_mask_iou:
            return True
    return False


def is_duplicate_crop(candidate: dict, accepted: list[dict], threshold: float) -> bool:
    return any(bbox_iou(candidate["crop_box"], existing["crop_box"]) >= threshold for existing in accepted)


def synthesize_composites(image: Image.Image, candidates: list[dict], args) -> list[dict]:
    composites: list[dict] = []
    if len(candidates) < 2 or args.max_composites_per_image <= 0:
        return composites

    pairs: list[tuple[float, dict, dict]] = []
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            if left["source_path"] != right["source_path"]:
                continue
            score = composite_affinity(left, right, args)
            if score <= 0:
                continue
            pairs.append((score, left, right))

    used_keys: set[tuple[int, int]] = set()
    for _, left, right in sorted(pairs, key=lambda item: item[0], reverse=True):
        key = tuple(sorted((left["detection_index"], right["detection_index"])))
        if key in used_keys:
            continue
        used_keys.add(key)
        composite = build_composite_candidate(image, left, right, args)
        if composite is not None:
            composites.append(composite)
        if len(composites) >= args.max_composites_per_image:
            break
    return composites


def composite_affinity(left: dict, right: dict, args) -> float:
    box_overlap = bbox_iou(left["global_bbox"], right["global_bbox"])
    overlap = mask_iou(left["full_mask"], right["full_mask"])
    gap_ratio = bbox_gap_ratio(left["global_bbox"], right["global_bbox"])
    if box_overlap >= args.duplicate_box_iou or overlap >= args.duplicate_mask_iou:
        return 0.0
    if box_overlap < args.composite_box_iou and overlap < args.composite_mask_iou and gap_ratio > args.composite_gap_ratio:
        return 0.0
    return box_overlap * 2.0 + overlap * 3.0 + max(0.0, args.composite_gap_ratio - gap_ratio)


def bbox_gap_ratio(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    lx0, ly0, lx1, ly1 = left
    rx0, ry0, rx1, ry1 = right
    gap_x = max(0, max(lx0, rx0) - min(lx1, rx1))
    gap_y = max(0, max(ly0, ry0) - min(ly1, ry1))
    gap = (gap_x * gap_x + gap_y * gap_y) ** 0.5
    scale = max(1, max(lx1 - lx0, ly1 - ly0, rx1 - rx0, ry1 - ry0))
    return gap / scale


def build_composite_candidate(image: Image.Image, left: dict, right: dict, args) -> dict | None:
    full_mask = Image.new("L", image.size, 0)
    full_mask = Image.composite(Image.new("L", image.size, 255), full_mask, left["full_mask"])
    full_mask = Image.composite(Image.new("L", image.size, 255), full_mask, right["full_mask"])
    global_bbox = full_mask.getbbox()
    if global_bbox is None:
        return None
    area = mask_area(full_mask)
    if area < args.min_mask_area:
        return None
    cropped_image, cropped_mask, padded_bbox = crop_with_padding(image, full_mask, global_bbox, args.crop_padding)
    features = measure_mask(cropped_mask)
    goose_score = score_goose_candidate(features, image.size, bbox=global_bbox)
    quality = mask_quality_metrics(cropped_mask)
    if goose_score["goose_candidate_score"] < args.min_goose_score * 0.85:
        return None
    return {
        "source_path": left["source_path"],
        "detection": Detection(
            bbox=global_bbox,
            score=round(max(left["detection"].score, right["detection"].score), 4),
            label="composite:overlapping-geese",
        ),
        "detection_index": min(left["detection_index"], right["detection_index"]),
        "crop_box": union_bbox(left["crop_box"], right["crop_box"]),
        "global_bbox": global_bbox,
        "padded_bbox": padded_bbox,
        "full_mask": full_mask,
        "cropped_image": cropped_image,
        "cropped_mask": cropped_mask,
        "features": features,
        "goose_score": goose_score,
        "mask_quality": quality,
        "mask_area": area,
        "instance_kind": "composite",
        "member_detection_indices": sorted(set(left["member_detection_indices"] + right["member_detection_indices"])),
    }


def union_bbox(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return (
        min(left[0], right[0]),
        min(left[1], right[1]),
        max(left[2], right[2]),
        max(left[3], right[3]),
    )


def mask_iou(a: Image.Image, b: Image.Image) -> float:
    a_pixels = a.convert("L").load()
    b_pixels = b.convert("L").load()
    width, height = a.size
    intersection = union = 0
    for y in range(height):
        for x in range(width):
            av = a_pixels[x, y] > 127
            bv = b_pixels[x, y] > 127
            intersection += int(av and bv)
            union += int(av or bv)
    return intersection / union if union else 0.0


def save_instance(candidate: dict, image: Image.Image, dirs: dict[str, Path], index: int) -> dict:
    goose_id = f"goose_{index:05d}"
    mask_path = dirs["masks"] / f"{goose_id}.png"
    cutout_path = dirs["cutouts"] / f"{goose_id}.png"
    silhouette_path = dirs["silhouettes"] / f"{goose_id}.png"
    candidate["cropped_mask"].save(mask_path)
    cutout_from_mask(candidate["cropped_image"], candidate["cropped_mask"]).save(cutout_path)
    silhouette_from_mask(candidate["cropped_mask"]).save(silhouette_path)

    x0, y0, x1, y1 = candidate["padded_bbox"]
    features = dict(candidate["features"])
    mask_bbox = features.pop("bbox", None)
    record = {
        "id": goose_id,
        "source_image": str(candidate["source_path"]),
        "source_hash": source_hash(candidate["source_path"]),
        "extraction_version": "v2",
        "instance_kind": candidate["instance_kind"],
        "member_detection_indices": candidate["member_detection_indices"],
        "detection": asdict(candidate["detection"]),
        "detection_index": candidate["detection_index"],
        "crop_box": list(candidate["crop_box"]),
        "source_mask_bbox": list(candidate["global_bbox"]),
        "bbox": [x0, y0, x1 - x0, y1 - y0],
        "mask_path": str(mask_path),
        "cutout_path": str(cutout_path),
        "silhouette_path": str(silhouette_path),
        "width": candidate["cropped_image"].width,
        "height": candidate["cropped_image"].height,
        "component_area": candidate["mask_area"],
        **features,
        "mask_bbox": mask_bbox,
        **candidate["goose_score"],
        **candidate["mask_quality"],
    }
    return record


def save_crop_instance(candidate: dict, dirs: dict[str, Path], index: int) -> dict:
    crop_id = f"crop_{index:05d}"
    crop_path = dirs["crops"] / f"{crop_id}.jpg"
    candidate["cropped_image"].convert("RGB").save(crop_path, quality=92)

    x0, y0, x1, y1 = candidate["padded_bbox"]
    return {
        "id": crop_id,
        "source_image": str(candidate["source_path"]),
        "source_hash": source_hash(candidate["source_path"]),
        "extraction_version": "v2",
        "stage": "crop",
        "instance_kind": candidate["instance_kind"],
        "member_detection_indices": candidate["member_detection_indices"],
        "detection": asdict(candidate["detection"]),
        "detection_index": candidate["detection_index"],
        "crop_box": list(candidate["crop_box"]),
        "bbox": [x0, y0, x1 - x0, y1 - y0],
        "crop_path": str(crop_path),
        "width": candidate["cropped_image"].width,
        "height": candidate["cropped_image"].height,
        **candidate["crop_quality"],
    }


def maybe_save_rejected(candidate: dict, dirs: dict[str, Path], reason: str, keep: bool) -> None:
    if not keep:
        return
    path = dirs["rejected"] / f"{Path(candidate['source_path']).stem}_{candidate['detection_index']:03d}_{reason}.png"
    cutout_from_mask(candidate["cropped_image"], candidate["cropped_mask"]).save(path)


def build_contact_sheet(output: Path, metadata: list[dict]) -> None:
    if not metadata:
        return
    columns = 6
    per_sheet = 120
    sheet_paths = []
    for start in range(0, len(metadata), per_sheet):
        chunk = metadata[start : start + per_sheet]
        filename = "contact_sheet.png" if start == 0 else f"contact_sheet_{start // per_sheet + 1:03d}.png"
        sheet_paths.append(draw_contact_sheet(output, chunk, filename, start_index=start, total=len(metadata), columns=columns))
    (output / "contact_sheets.json").write_text(json.dumps(sheet_paths, indent=2), encoding="utf-8")


def draw_contact_sheet(
    output: Path,
    metadata: list[dict],
    filename: str,
    start_index: int,
    total: int,
    columns: int,
) -> str:
    cell_w = 180
    image_h = 132
    label_h = 42
    header_h = 38
    rows = (len(metadata) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, header_h + rows * (image_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    title_font, label_font = load_sheet_fonts()
    draw.text(
        (12, 9),
        f"GooseType v2 instances: {start_index + 1}-{start_index + len(metadata)} of {total}",
        fill=(20, 24, 28),
        font=title_font,
    )

    for index, item in enumerate(metadata):
        x = (index % columns) * cell_w
        y = header_h + (index // columns) * (image_h + label_h)
        draw.rectangle((x, y, x + cell_w - 1, y + image_h + label_h - 1), outline=(225, 225, 225))
        preview_path = item.get("silhouette_path") or item.get("crop_path")
        if not preview_path:
            continue
        preview = Image.open(preview_path).convert("RGBA")
        preview.thumbnail((cell_w - 18, image_h - 12), Image.Resampling.LANCZOS)
        cell = Image.new("RGBA", (cell_w, image_h), (255, 255, 255, 255))
        cell.alpha_composite(preview, ((cell_w - preview.width) // 2, (image_h - preview.height) // 2))
        sheet.paste(cell.convert("RGB"), (x, y))
        label = f"{item['id']} {item.get('instance_kind', 'single')}"
        if item.get("stage") == "crop":
            detection = item.get("detection") or {}
            status = str(item.get("crop_quality_status") or "unknown")
            score = f"{status} cq={float(item.get('crop_quality_score') or 0):.2f} det={float(detection.get('score') or 0):.2f}"
        else:
            score = f"q={float(item.get('mask_quality_score') or 0):.2f} g={float(item.get('goose_candidate_score') or 0):.2f}"
        draw.text((x + 6, y + image_h + 5), label[:25], fill=(48, 54, 60), font=label_font)
        draw.text((x + 6, y + image_h + 21), score, fill=(80, 86, 92), font=label_font)

    path = output / filename
    sheet.save(path)
    return str(path)


def load_sheet_fonts():
    try:
        return (
            ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 19),
            ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 12),
        )
    except OSError:
        return ImageFont.load_default(), ImageFont.load_default()


def mask_area(mask: Image.Image) -> int:
    return sum(1 for value in mask.convert("L").getdata() if value > 127)


def source_hash(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


if __name__ == "__main__":
    main()
