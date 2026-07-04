#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from PIL import Image
except ImportError as error:
    raise SystemExit("Pillow is required: python3 -m pip install Pillow") from error

from goosetype.image_features import mask_iou


FEATURE_KEYS = ("aspect_ratio", "curvature_score", "boldness_score", "slant_score", "fill_ratio")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select goose candidates for each letter by cheap feature similarity first, "
            "then apply slower mask IoU only to the narrowed set."
        )
    )
    parser.add_argument("--features", default="data/processed/features.json")
    parser.add_argument("--font-targets", default="data/processed/font_targets/reference.json")
    parser.add_argument("--output", default="data/processed/candidates.json")
    parser.add_argument("--top-k", type=int, default=8, help="Final candidates to keep per letter.")
    parser.add_argument("--preselect", type=int, default=24, help="Feature-ranked candidates to send to mask comparison.")
    args = parser.parse_args()

    geese = json.loads(Path(args.features).read_text(encoding="utf-8"))
    targets = json.loads(Path(args.font_targets).read_text(encoding="utf-8"))["targets"]
    output = {}
    mask_cache: dict[str, Image.Image] = {}

    for target in targets:
        ranked = sorted(
            geese,
            key=lambda goose: feature_distance(goose, target),
        )[: args.preselect]
        scored = []
        target_mask = load_mask(target["mask_path"], mask_cache)
        for goose in ranked:
            goose_mask = load_mask(goose["mask_path"], mask_cache)
            image_score = mask_iou(goose_mask, target_mask)
            feature_score = 1 - min(feature_distance(goose, target), 1)
            total = feature_score * 0.62 + image_score["iou"] * 0.3 + image_score["identical_pixel_ratio"] * 0.08
            scored.append(
                {
                    "goose_id": goose["id"],
                    "letter": target["letter"],
                    "score": round(total, 4),
                    "feature_score": round(feature_score, 4),
                    **image_score,
                    "cutout_path": goose.get("cutout_path"),
                    "mask_path": goose.get("mask_path"),
                }
            )
        output[target["letter"]] = sorted(scored, key=lambda item: item["score"], reverse=True)[: args.top_k]
        print(f"{target['letter']}: kept {len(output[target['letter']])} candidates from {len(geese)} geese")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote candidate shortlist to {args.output}")


def load_mask(path: str, cache: dict[str, Image.Image]) -> Image.Image:
    if path not in cache:
        cache[path] = Image.open(path).convert("L")
    return cache[path]


def feature_distance(goose: dict, target: dict) -> float:
    weights = {
        "aspect_ratio": 0.28,
        "curvature_score": 0.26,
        "boldness_score": 0.2,
        "slant_score": 0.16,
        "fill_ratio": 0.1,
    }
    total = 0.0
    for key in FEATURE_KEYS:
        total += weights[key] * abs(float(goose.get(key, 0)) - float(target.get(key, 0)))
    return total


if __name__ == "__main__":
    main()

