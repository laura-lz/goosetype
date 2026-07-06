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
    args = parser.parse_args()

    feature_paths = args.features or ["data/processed/features.json"]
    geese = load_feature_records(feature_paths)
    targets = json.loads(Path(args.font_targets).read_text(encoding="utf-8"))["targets"]
    output = {}
    mask_cache: dict[str, Image.Image] = {}

    for target in targets:
        ranked = sorted(
            geese,
            key=lambda goose: preselect_distance(goose, target),
        )[: args.preselect]
        scored = []
        target_mask = load_mask(target["mask_path"], mask_cache)
        for goose in ranked:
            goose_mask = load_mask(goose["mask_path"], mask_cache)
            normal_score = score_mask_variant(goose_mask, target_mask, args.mask_size, flip_x=False)
            flipped_score = score_mask_variant(ImageOps.mirror(goose_mask), target_mask, args.mask_size, flip_x=True)
            mask_score = max((normal_score, flipped_score), key=lambda item: item["mask_total"])
            feature_score = 1 - min(global_feature_distance(goose, target), 1)
            structure_score = 1 - min(structure_distance(goose, target), 1)
            total = (
                feature_score * 0.22
                + structure_score * 0.34
                + mask_score["aligned_iou"] * 0.23
                + mask_score["aligned_dice"] * 0.14
                + mask_score["centered_iou"] * 0.07
            )
            scored.append(
                {
                    "goose_id": goose["id"],
                    "letter": target["letter"],
                    "score": round(total, 4),
                    "feature_score": round(feature_score, 4),
                    "structure_score": round(structure_score, 4),
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


def score_mask_variant(goose_mask: Image.Image, target_mask: Image.Image, size: int, flip_x: bool) -> dict:
    centered_score = mask_iou(goose_mask, target_mask, size=size)
    aligned_score = aligned_mask_similarity(goose_mask, target_mask, size=size)
    mask_total = aligned_score["aligned_iou"] * 0.58 + aligned_score["aligned_dice"] * 0.3 + centered_score["iou"] * 0.12
    return {
        "mask_total": mask_total,
        "flip_x": flip_x,
        "centered_iou": centered_score["iou"],
        "centered_identical_pixel_ratio": centered_score["identical_pixel_ratio"],
        **aligned_score,
    }


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
