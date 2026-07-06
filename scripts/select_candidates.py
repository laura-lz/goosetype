#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from PIL import Image, ImageOps
except ImportError as error:
    raise SystemExit("Pillow is required: python3 -m pip install Pillow") from error

from goosetype.classifier import load_glyph_classifier
from goosetype.image_features import aligned_mask_similarity, mask_iou


FEATURE_KEYS = ("aspect_ratio", "curvature_score", "boldness_score", "slant_score", "fill_ratio")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select goose candidates for each letter by cheap feature similarity first, "
            "then apply slower mask IoU only to the narrowed set."
        )
    )
    parser.add_argument(
        "--features",
        action="append",
        default=None,
        help="Goose feature file. Repeat to merge local and external candidate libraries.",
    )
    parser.add_argument("--font-targets", default="data/processed/font_targets/reference.json")
    parser.add_argument("--output", default="data/processed/candidates.json")
    parser.add_argument("--top-k", type=int, default=8, help="Final candidates to keep per letter.")
    parser.add_argument("--preselect", type=int, default=36, help="Feature-ranked candidates to send to mask comparison.")
    parser.add_argument("--mask-size", type=int, default=180, help="Raster size for mask similarity.")
    parser.add_argument("--classifier-model", default=None, help="Optional EMNIST-style classifier checkpoint.")
    parser.add_argument("--classifier-device", default="cpu")
    args = parser.parse_args()

    feature_paths = args.features or ["data/processed/features.json"]
    geese = load_feature_records(feature_paths)
    targets = json.loads(Path(args.font_targets).read_text(encoding="utf-8"))["targets"]
    output = {}
    mask_cache: dict[str, Image.Image] = {}
    classifier = load_glyph_classifier(args.classifier_model, device=args.classifier_device) if args.classifier_model else None

    for target in targets:
        ranked = sorted(
            geese,
            key=lambda goose: preselect_distance(goose, target),
        )[: args.preselect]
        scored = []
        target_mask = load_mask(target["mask_path"], mask_cache)
        for goose in ranked:
            goose_mask = load_mask(goose["mask_path"], mask_cache)
            normal_score = score_mask_variant(
                goose_mask,
                target_mask,
                args.mask_size,
                flip_x=False,
                classifier=classifier,
                target_letter=target["letter"],
            )
            flipped_score = score_mask_variant(
                ImageOps.mirror(goose_mask),
                target_mask,
                args.mask_size,
                flip_x=True,
                classifier=classifier,
                target_letter=target["letter"],
            )
            mask_score = max((normal_score, flipped_score), key=lambda item: item["variant_total"])
            feature_score = 1 - min(global_feature_distance(goose, target), 1)
            structure_score = 1 - min(structure_distance(goose, target), 1)
            quality_score = candidate_quality_score(goose, goose_mask, target)
            total = final_score(
                feature_score,
                structure_score,
                mask_score,
                quality_score=quality_score,
                classifier_enabled=classifier is not None,
            )
            scored.append(
                {
                    "goose_id": goose["id"],
                    "letter": target["letter"],
                    "score": round(total, 4),
                    "feature_score": round(feature_score, 4),
                    "structure_score": round(structure_score, 4),
                    "quality_score": round(quality_score, 4),
                    **{key: value for key, value in mask_score.items() if key != "mask_total"},
                    "source_image": goose.get("source_image"),
                    "cutout_path": goose.get("cutout_path"),
                    "silhouette_path": goose.get("silhouette_path"),
                    "mask_path": goose.get("mask_path"),
                }
            )
        output[target["letter"]] = sorted(scored, key=lambda item: item["score"], reverse=True)[: args.top_k]
        print(f"{target['letter']}: kept {len(output[target['letter']])} candidates from {len(geese)} geese")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote candidate shortlist to {args.output}")


def score_mask_variant(
    goose_mask: Image.Image,
    target_mask: Image.Image,
    size: int,
    flip_x: bool,
    classifier=None,
    target_letter: str | None = None,
) -> dict:
    centered_score = mask_iou(goose_mask, target_mask, size=size)
    aligned_score = aligned_mask_similarity(goose_mask, target_mask, size=size)
    mask_total = aligned_score["aligned_iou"] * 0.58 + aligned_score["aligned_dice"] * 0.3 + centered_score["iou"] * 0.12
    classifier_score = classifier.score_mask(goose_mask, target_letter) if classifier and target_letter else None
    variant_total = mask_total if classifier_score is None else mask_total * 0.35 + classifier_score * 0.65
    return {
        "mask_total": mask_total,
        "variant_total": variant_total,
        "flip_x": flip_x,
        "classifier_score": classifier_score,
        "centered_iou": centered_score["iou"],
        "centered_identical_pixel_ratio": centered_score["identical_pixel_ratio"],
        **aligned_score,
    }


def final_score(
    feature_score: float,
    structure_score: float,
    mask_score: dict,
    quality_score: float,
    classifier_enabled: bool,
) -> float:
    if classifier_enabled:
        classifier_score = float(mask_score.get("classifier_score") or 0)
        return (
            classifier_score * 0.55
            + quality_score * 0.12
            + structure_score * 0.12
            + feature_score * 0.08
            + mask_score["aligned_iou"] * 0.08
            + mask_score["aligned_dice"] * 0.05
        )
    return (
        feature_score * 0.18
        + structure_score * 0.28
        + quality_score * 0.22
        + mask_score["aligned_iou"] * 0.19
        + mask_score["aligned_dice"] * 0.09
        + mask_score["centered_iou"] * 0.04
    )


def load_feature_records(paths: list[str]) -> list[dict]:
    records = []
    seen = set()
    for path in paths:
        for item in json.loads(Path(path).read_text(encoding="utf-8")):
            key = (item.get("id"), item.get("mask_path"))
            if key in seen:
                continue
            seen.add(key)
            records.append(item)
    return records


def load_mask(path: str, cache: dict[str, Image.Image]) -> Image.Image:
    if path not in cache:
        cache[path] = Image.open(path).convert("L")
    return cache[path]


def candidate_quality_score(goose: dict, goose_mask: Image.Image, target: dict) -> float:
    quality = float(goose.get("goose_candidate_score", 1) or 0)
    if goose.get("goose_candidate_reasons"):
        quality -= min(0.35, 0.16 * len(goose.get("goose_candidate_reasons") or []))

    detection = goose.get("detection") or {}
    detector_score = detection.get("score")
    if detector_score is not None:
        quality -= max(0.0, 0.38 - float(detector_score)) * 0.9

    coverage = goose.get("detector_mask_coverage")
    if coverage is not None:
        quality -= max(0.0, 0.24 - float(coverage)) * 1.1

    sam_support = goose.get("sam_support_coverage")
    if sam_support is not None:
        quality -= max(0.0, 0.18 - float(sam_support)) * 0.8

    quality = quality * 0.62 + mask_cleanliness_score(goose_mask, target) * 0.38
    return max(0.0, min(1.0, quality))


def mask_cleanliness_score(mask: Image.Image, target: dict, size: int = 96) -> float:
    normalized = ImageOps.contain(mask.convert("L"), (size, size))
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(normalized, ((size - normalized.width) // 2, (size - normalized.height) // 2))
    binary = canvas.point(lambda value: 255 if value > 127 else 0)
    pixels = binary.load()
    bbox = binary.getbbox()
    if not bbox:
        return 0.0

    area = 0
    perimeter = 0
    border_pixels = 0
    for y in range(size):
        for x in range(size):
            if pixels[x, y] <= 127:
                continue
            area += 1
            if x in {0, size - 1} or y in {0, size - 1}:
                border_pixels += 1
            if is_edge_pixel(pixels, x, y, size, size):
                perimeter += 1

    bbox_area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    fill_ratio = area / bbox_area
    edge_density = perimeter / max(1, area)
    border_ratio = border_pixels / max(1, perimeter)
    component_count = connected_component_count(binary, min_area=12)

    score = 1.0
    if component_count > 3:
        score -= min(0.45, (component_count - 3) * 0.08)
    if border_ratio > 0.08:
        score -= min(0.35, (border_ratio - 0.08) * 2.4)
    if edge_density > 0.42 and fill_ratio > 0.42:
        score -= min(0.35, (edge_density - 0.42) * 1.1)
    if fill_ratio > 0.78:
        score -= min(0.4, (fill_ratio - 0.78) * 1.8)

    target_aspect = float(target.get("aspect_ratio", 1) or 1)
    goose_aspect = (bbox[2] - bbox[0]) / max(1, bbox[3] - bbox[1])
    if target_aspect < 0.36 and goose_aspect > 0.8:
        score -= min(0.5, (goose_aspect - 0.8) * 0.55)

    return max(0.0, min(1.0, score))


def connected_component_count(mask: Image.Image, min_area: int = 12) -> int:
    binary = mask.convert("1")
    width, height = binary.size
    pixels = binary.load()
    seen: set[tuple[int, int]] = set()
    count = 0
    for y in range(height):
        for x in range(width):
            if (x, y) in seen or not pixels[x, y]:
                continue
            area = flood_area(pixels, x, y, width, height, seen)
            if area >= min_area:
                count += 1
    return count


def flood_area(pixels, start_x: int, start_y: int, width: int, height: int, seen: set[tuple[int, int]]) -> int:
    stack = [(start_x, start_y)]
    seen.add((start_x, start_y))
    area = 0
    while stack:
        x, y = stack.pop()
        area += 1
        for nx in (x - 1, x, x + 1):
            for ny in (y - 1, y, y + 1):
                if nx == x and ny == y:
                    continue
                if nx < 0 or ny < 0 or nx >= width or ny >= height or (nx, ny) in seen:
                    continue
                if pixels[nx, ny]:
                    seen.add((nx, ny))
                    stack.append((nx, ny))
    return area


def is_edge_pixel(pixels, x: int, y: int, width: int, height: int) -> bool:
    if x == 0 or y == 0 or x == width - 1 or y == height - 1:
        return True
    return (
        pixels[x - 1, y] <= 127
        or pixels[x + 1, y] <= 127
        or pixels[x, y - 1] <= 127
        or pixels[x, y + 1] <= 127
    )


def preselect_distance(goose: dict, target: dict) -> float:
    return global_feature_distance(goose, target) * 0.42 + structure_distance(goose, target) * 0.58


def global_feature_distance(goose: dict, target: dict) -> float:
    weights = {
        "aspect_ratio": 0.28,
        "curvature_score": 0.18,
        "boldness_score": 0.17,
        "slant_score": 0.17,
        "fill_ratio": 0.2,
    }
    total = 0.0
    for key in FEATURE_KEYS:
        total += weights[key] * abs(float(goose.get(key, 0)) - float(target.get(key, 0)))
    return total


def structure_distance(goose: dict, target: dict) -> float:
    return (
        vector_distance(goose.get("grid_3x3"), target.get("grid_3x3"), fallback=0.5) * 0.24
        + vector_distance(goose.get("projection_x"), target.get("projection_x"), fallback=0.5) * 0.22
        + vector_distance(goose.get("projection_y"), target.get("projection_y"), fallback=0.5) * 0.22
        + vector_distance(goose.get("contour_grid_4x4"), target.get("contour_grid_4x4"), fallback=0.5) * 0.16
        + vector_distance(goose.get("shape_context"), target.get("shape_context"), fallback=0.5) * 0.16
    )


def grid_distance(a: object, b: object) -> float:
    return vector_distance(a, b, fallback=0.5)


def vector_distance(a: object, b: object, fallback: float = 0.5) -> float:
    if not isinstance(a, list) or not isinstance(b, list):
        return fallback
    length = min(len(a), len(b))
    if length == 0:
        return fallback
    return sum(abs(float(a[index]) - float(b[index])) for index in range(length)) / length


if __name__ == "__main__":
    main()
