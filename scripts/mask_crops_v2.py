#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw, ImageFont

from goosetype.image_features import (
    Detection,
    build_instance_mask,
    build_foreground_mask,
    crop_with_padding,
    cutout_from_mask,
    mask_quality_metrics,
    measure_mask,
    refine_instance_mask,
    score_goose_candidate,
    silhouette_from_mask,
)
from scripts.extract_instances_v2 import build_contact_sheet, mask_area


def main() -> None:
    parser = argparse.ArgumentParser(description="Mask GooseType v2 crop inventory records without rerunning detection.")
    parser.add_argument("--crops-metadata", default="data/crops_v2_full/metadata.json")
    parser.add_argument("--output", default="data/masks_v2_from_crops")
    parser.add_argument("--status", action="append", default=None, help="Crop quality status to include. Default: pass.")
    parser.add_argument("--skip-items", type=int, default=0, help="Skip this many filtered crop records before masking.")
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--segmentation-backend", choices=("sam", "rembg", "grabcut", "heuristic", "auto"), default="sam")
    parser.add_argument("--fallback-backend", choices=("rembg", "grabcut", "heuristic", "auto", "none"), default="rembg")
    parser.add_argument("--sam-model", default="facebook/sam-vit-base")
    parser.add_argument("--rembg-model", default="isnet-general-use")
    parser.add_argument("--max-segmentation-side", type=int, default=1000)
    parser.add_argument("--min-mask-area", type=int, default=120)
    parser.add_argument("--min-goose-score", type=float, default=0.18)
    parser.add_argument("--crop-padding", type=int, default=12)
    parser.add_argument("--keep-rejected", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    dirs = prepare_output_dirs(output, keep_rejected=args.keep_rejected)
    crop_records = load_crop_records(Path(args.crops_metadata), statuses=set(args.status or ["pass"]))
    if args.skip_items:
        crop_records = crop_records[args.skip_items :]
    if args.max_items:
        crop_records = crop_records[: args.max_items]

    metadata = []
    rejected = []
    for index, crop_record in enumerate(crop_records, start=1):
        print(f"[{index}/{len(crop_records)}] {crop_record['id']} {crop_record['crop_path']}", flush=True)
        try:
            result = mask_crop_record(crop_record, args)
        except Exception as error:
            result = None
            rejected.append({"crop_id": crop_record.get("id"), "reason": f"error:{error}"})
            print(f"  rejected error: {error}", flush=True)
        if result is None:
            rejected.append({"crop_id": crop_record.get("id"), "reason": "mask_quality"})
            continue
        record = save_masked_crop(result, dirs, len(metadata) + 1)
        metadata.append(record)
        print(f"  accepted {record['id']} mask={record.get('mask_quality_score', 0):.2f}", flush=True)

    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "features.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "rejected.json").write_text(json.dumps(rejected, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "schema": "goosetype-crop-masks-v2-summary",
                "source_metadata": args.crops_metadata,
                "input_records": len(crop_records),
                "skip_items": args.skip_items,
                "accepted": len(metadata),
                "rejected": len(rejected),
                "statuses": sorted(set(args.status or ["pass"])),
                "output": str(output),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    build_contact_sheet(output, metadata)
    build_mask_diagnostic_sheet(output, metadata)
    print(f"Wrote {len(metadata)} masked crops to {output}", flush=True)


def prepare_output_dirs(output: Path, keep_rejected: bool) -> dict[str, Path]:
    dirs = {
        "output": output,
        "masks": output / "masks",
        "cutouts": output / "cutouts",
        "silhouettes": output / "silhouettes",
        "rejected": output / "rejected",
    }
    for key in ("masks", "cutouts", "silhouettes"):
        dirs[key].mkdir(parents=True, exist_ok=True)
    if keep_rejected:
        dirs["rejected"].mkdir(parents=True, exist_ok=True)
    return dirs


def load_crop_records(path: Path, statuses: set[str]) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [row for row in rows if row.get("crop_quality_status") in statuses and row.get("crop_path")]


def mask_crop_record(crop_record: dict, args) -> dict | None:
    crop = Image.open(crop_record["crop_path"]).convert("RGBA")
    prompt_box = local_detection_box(crop_record, crop.size)
    mask = build_mask_with_fallback(crop, prompt_box, args)
    mask = refine_instance_mask(mask, min_island_area=max(18, args.min_mask_area // 10))
    if not mask.getbbox() or mask_area(mask) < args.min_mask_area:
        return None

    bbox = mask.getbbox()
    assert bbox is not None
    cropped_image, cropped_mask, padded_bbox = crop_with_padding(crop, mask, bbox, args.crop_padding)
    features = measure_mask(cropped_mask)
    goose_score = score_goose_candidate(features, crop.size, bbox=bbox)
    quality = mask_quality_metrics(cropped_mask)
    if goose_score["goose_candidate_score"] < args.min_goose_score:
        return None

    return {
        "crop_record": crop_record,
        "crop_image": crop,
        "cropped_image": cropped_image,
        "cropped_mask": cropped_mask,
        "padded_bbox": padded_bbox,
        "source_mask_bbox": bbox,
        "features": features,
        "goose_score": goose_score,
        "mask_quality": quality,
        "mask_area": mask_area(cropped_mask),
    }


def local_detection_box(crop_record: dict, crop_size: tuple[int, int]) -> tuple[int, int, int, int]:
    detection = crop_record.get("detection") or {}
    source_box = detection.get("bbox") or crop_record.get("crop_box")
    crop_box = crop_record.get("crop_box")
    if not source_box or not crop_box:
        return inset_box(crop_size, 0.08)
    x0, y0, x1, y1 = source_box
    cx0, cy0, _, _ = crop_box
    local = (
        max(0, int(round(x0 - cx0))),
        max(0, int(round(y0 - cy0))),
        min(crop_size[0], int(round(x1 - cx0))),
        min(crop_size[1], int(round(y1 - cy0))),
    )
    if local[2] - local[0] < 4 or local[3] - local[1] < 4:
        return inset_box(crop_size, 0.08)
    return local


def inset_box(size: tuple[int, int], ratio: float) -> tuple[int, int, int, int]:
    width, height = size
    inset = max(1, int(round(min(width, height) * ratio)))
    return inset, inset, max(inset + 1, width - inset), max(inset + 1, height - inset)


def build_mask_with_fallback(crop: Image.Image, prompt_box: tuple[int, int, int, int], args) -> Image.Image:
    try:
        support_mask = None
        if args.segmentation_backend == "sam":
            try:
                support_mask = build_foreground_mask(
                    crop,
                    backend=args.fallback_backend if args.fallback_backend != "none" else "rembg",
                    rembg_model=args.rembg_model,
                    max_segmentation_side=args.max_segmentation_side,
                )
            except Exception:
                support_mask = None
        return build_instance_mask(
            crop,
            bbox=prompt_box,
            backend=args.segmentation_backend,
            sam_model=args.sam_model,
            rembg_model=args.rembg_model,
            max_segmentation_side=args.max_segmentation_side,
            support_mask=support_mask,
        )
    except Exception as error:
        if args.fallback_backend == "none":
            raise
        print(f"  segmentation fallback ({args.segmentation_backend} -> {args.fallback_backend}): {error}", flush=True)
        return build_instance_mask(
            crop,
            bbox=prompt_box if args.fallback_backend == "sam" else None,
            backend=args.fallback_backend,
            rembg_model=args.rembg_model,
            max_segmentation_side=args.max_segmentation_side,
        )


def save_masked_crop(result: dict, dirs: dict[str, Path], index: int) -> dict:
    crop_record = result["crop_record"]
    mask_id = f"mask_{index:05d}"
    mask_path = dirs["masks"] / f"{mask_id}.png"
    cutout_path = dirs["cutouts"] / f"{mask_id}.png"
    silhouette_path = dirs["silhouettes"] / f"{mask_id}.png"
    result["cropped_mask"].save(mask_path)
    cutout_from_mask(result["cropped_image"], result["cropped_mask"]).save(cutout_path)
    silhouette_from_mask(result["cropped_mask"]).save(silhouette_path)

    x0, y0, x1, y1 = result["padded_bbox"]
    features = dict(result["features"])
    mask_bbox = features.pop("bbox", None)
    record = {
        "id": mask_id,
        "source_crop_id": crop_record["id"],
        "source_image": crop_record["source_image"],
        "crop_path": crop_record["crop_path"],
        "extraction_version": "v2-crop-mask",
        "instance_kind": "single",
        "crop_quality_status": crop_record.get("crop_quality_status"),
        "crop_quality_score": crop_record.get("crop_quality_score"),
        "detection": crop_record.get("detection"),
        "source_crop_bbox": crop_record.get("bbox"),
        "source_mask_bbox": list(result["source_mask_bbox"]),
        "bbox": [x0, y0, x1 - x0, y1 - y0],
        "mask_path": str(mask_path),
        "cutout_path": str(cutout_path),
        "silhouette_path": str(silhouette_path),
        "width": result["cropped_image"].width,
        "height": result["cropped_image"].height,
        "source_crop_width": result["crop_image"].width,
        "source_crop_height": result["crop_image"].height,
        "component_area": result["mask_area"],
        **features,
        "mask_bbox": mask_bbox,
        **result["goose_score"],
        **result["mask_quality"],
    }
    record.update(mask_completeness_status(record))
    record.update(mask_review_status(record))
    return record


def mask_completeness_status(record: dict) -> dict:
    x0, y0, x1, y1 = record.get("source_mask_bbox") or [0, 0, 0, 0]
    width = int(record.get("source_crop_width") or 0)
    height = int(record.get("source_crop_height") or 0)
    tolerance = max(3, int(round(min(width or 1, height or 1) * 0.015)))
    edges = []
    if x0 <= tolerance:
        edges.append("left")
    if y0 <= tolerance:
        edges.append("top")
    if width and x1 >= width - tolerance:
        edges.append("right")
    if height and y1 >= height - tolerance:
        edges.append("bottom")

    reasons = []
    if edges:
        reasons.append("mask_touches_crop_edge")
    if len(edges) >= 2:
        reasons.append("mask_touches_multiple_crop_edges")
    if record.get("crop_quality_status") == "review":
        reasons.append("borderline_source_crop")

    status = "review" if reasons else "pass"
    return {
        "mask_completeness_status": status,
        "mask_completeness_reasons": reasons,
        "mask_crop_edge_sides": edges,
    }


def mask_review_status(record: dict) -> dict:
    reasons = set(record.get("mask_quality_reasons") or [])
    crop_quality = float(record.get("crop_quality_score") or 0)
    mask_quality = float(record.get("mask_quality_score") or 0)
    goose_score = float(record.get("goose_candidate_score") or 0)
    review_reasons = []

    if crop_quality < 0.72:
        review_reasons.append("borderline_source_crop")
    if record.get("mask_completeness_status") == "review":
        review_reasons.extend(record.get("mask_completeness_reasons") or [])
    if mask_quality < 0.72:
        review_reasons.append("low_mask_quality")
    if goose_score < 0.35:
        review_reasons.append("low_goose_geometry_score")
    for reason in (
        "overfilled_blob_mask",
        "giant_or_noisy_patch",
        "foreground_touches_crop_edge",
        "jagged_or_shredded_mask",
        "fragmented_components",
    ):
        if reason in reasons:
            review_reasons.append(reason)

    if "empty_mask" in reasons or mask_quality < 0.42 or goose_score < 0.18:
        status = "reject"
    elif review_reasons:
        status = "review"
    else:
        status = "pass"

    return {
        "mask_review_status": status,
        "mask_review_reasons": sorted(set(review_reasons)),
    }


def build_mask_diagnostic_sheet(output: Path, metadata: list[dict]) -> None:
    if not metadata:
        return
    columns = 3
    cell_w = 300
    image_h = 112
    label_h = 38
    header_h = 34
    per_sheet = 36
    paths = []
    for start in range(0, len(metadata), per_sheet):
        chunk = metadata[start : start + per_sheet]
        filename = "diagnostic_sheet.png" if start == 0 else f"diagnostic_sheet_{start // per_sheet + 1:03d}.png"
        path = draw_mask_diagnostic_sheet(output, filename, chunk, start, len(metadata), columns, cell_w, image_h, label_h, header_h)
        paths.append(str(path))
    (output / "diagnostic_sheets.json").write_text(json.dumps(paths, indent=2), encoding="utf-8")


def draw_mask_diagnostic_sheet(
    output: Path,
    filename: str,
    rows: list[dict],
    start_index: int,
    total: int,
    columns: int,
    cell_w: int,
    image_h: int,
    label_h: int,
    header_h: int,
) -> Path:
    sheet_rows = (len(rows) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, header_h + sheet_rows * (image_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    title_font, label_font = load_fonts()
    draw.text((10, 8), f"Crop / cutout / silhouette: {start_index + 1}-{start_index + len(rows)} of {total}", fill=(20, 24, 28), font=title_font)
    for index, item in enumerate(rows):
        x = (index % columns) * cell_w
        y = header_h + (index // columns) * (image_h + label_h)
        draw.rectangle((x, y, x + cell_w - 1, y + image_h + label_h - 1), outline=(225, 225, 225))
        previews = [
            Image.open(item["crop_path"]).convert("RGBA"),
            Image.open(item["cutout_path"]).convert("RGBA"),
            Image.open(item["silhouette_path"]).convert("RGBA"),
        ]
        slot_w = cell_w // 3
        for slot, preview in enumerate(previews):
            preview.thumbnail((slot_w - 8, image_h - 8), Image.Resampling.LANCZOS)
            tile = Image.new("RGBA", (slot_w, image_h), (255, 255, 255, 255))
            tile.alpha_composite(preview, ((slot_w - preview.width) // 2, (image_h - preview.height) // 2))
            sheet.paste(tile.convert("RGB"), (x + slot * slot_w, y))
        label = (
            f"{item['id']} {item.get('mask_review_status', 'pass')} "
            f"q={float(item.get('mask_quality_score') or 0):.2f}"
        )
        draw.text((x + 6, y + image_h + 5), label[:42], fill=(48, 54, 60), font=label_font)
        reasons = ",".join((item.get("mask_quality_reasons") or item.get("goose_candidate_reasons") or [])[:2])
        draw.text((x + 6, y + image_h + 21), reasons[:42], fill=(80, 86, 92), font=label_font)
    path = output / filename
    sheet.save(path)
    return path


def load_fonts():
    try:
        return (
            ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 17),
            ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 11),
        )
    except OSError:
        return ImageFont.load_default(), ImageFont.load_default()


if __name__ == "__main__":
    main()
