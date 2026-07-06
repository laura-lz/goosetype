#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from PIL import Image, ImageChops, ImageOps
except ImportError as error:
    raise SystemExit("Pillow is required: python3 -m pip install Pillow") from error

from goosetype.image_features import aligned_mask_similarity, mask_iou, silhouette_from_mask


LAYOUTS = {
    "side_by_side": ((0.02, 0.18, 0.56, 0.74), (0.42, 0.18, 0.56, 0.74)),
    "overlap_wide": ((0.00, 0.16, 0.66, 0.78), (0.34, 0.16, 0.66, 0.78)),
    "stacked": ((0.16, 0.00, 0.68, 0.56), (0.16, 0.44, 0.68, 0.56)),
    "diagonal_down": ((0.02, 0.04, 0.62, 0.68), (0.36, 0.32, 0.62, 0.68)),
    "diagonal_up": ((0.02, 0.32, 0.62, 0.68), (0.36, 0.04, 0.62, 0.68)),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prototype two-goose composite candidates for hard letters from an existing single-goose shortlist."
    )
    parser.add_argument("--candidates", default="data/font_candidates/arial_candidates.json")
    parser.add_argument("--font-targets", default="data/font_targets/arial.json")
    parser.add_argument("--output", default="data/font_candidates/arial_composite_candidates.json")
    parser.add_argument("--image-dir", default="data/font_candidates/composites/arial")
    parser.add_argument("--top-n", type=int, default=5, help="Single candidates per letter to combine.")
    parser.add_argument("--top-k", type=int, default=4, help="Composite candidates to keep per letter.")
    parser.add_argument("--mask-size", type=int, default=180)
    args = parser.parse_args()

    candidates = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    targets = json.loads(Path(args.font_targets).read_text(encoding="utf-8"))["targets"]
    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    output: dict[str, list[dict]] = {}
    mask_cache: dict[str, Image.Image] = {}

    for target in targets:
        letter = target["letter"]
        target_mask = load_mask(target["mask_path"], mask_cache)
        singles = candidates.get(letter, [])[: args.top_n]
        scored = []
        for left_index, left in enumerate(singles):
            for right_index, right in enumerate(singles):
                if left_index == right_index:
                    continue
                left_mask = candidate_mask(left, mask_cache)
                right_mask = candidate_mask(right, mask_cache)
                for layout_name, layout in LAYOUTS.items():
                    composite = compose_pair(left_mask, right_mask, layout, size=args.mask_size)
                    aligned = aligned_mask_similarity(composite, target_mask, size=args.mask_size)
                    centered = mask_iou(composite, target_mask, size=args.mask_size)
                    score = aligned["aligned_iou"] * 0.48 + aligned["aligned_dice"] * 0.34 + centered["iou"] * 0.18
                    scored.append(
                        {
                            "letter": letter,
                            "score": round(score, 4),
                            "layout": layout_name,
                            "left": candidate_summary(left),
                            "right": candidate_summary(right),
                            "centered_iou": centered["iou"],
                            **aligned,
                            "_mask": composite,
                        }
                    )

        kept = sorted(scored, key=lambda item: item["score"], reverse=True)[: args.top_k]
        output[letter] = []
        for index, item in enumerate(kept, start=1):
            image_path = image_dir / f"{safe_letter_id(letter)}_{index:02d}.png"
            silhouette_from_mask(item.pop("_mask"), color=(0, 0, 0)).save(image_path)
            item["silhouette_path"] = str(image_path)
            output[letter].append(item)
        print(f"{letter}: kept {len(output[letter])} composite candidates")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote composite shortlist to {args.output}")


def load_mask(path: str, cache: dict[str, Image.Image]) -> Image.Image:
    if path not in cache:
        cache[path] = Image.open(path).convert("L")
    return cache[path]


def candidate_mask(candidate: dict, cache: dict[str, Image.Image]) -> Image.Image:
    mask = load_mask(candidate["mask_path"], cache)
    if candidate.get("flip_x"):
        return ImageOps.mirror(mask)
    return mask


def compose_pair(
    left: Image.Image,
    right: Image.Image,
    layout: tuple[tuple[float, float, float, float], tuple[float, float, float, float]],
    size: int,
) -> Image.Image:
    canvas = Image.new("L", (size, size), 0)
    for mask, box in ((left, layout[0]), (right, layout[1])):
        x, y, width, height = box
        placed = ImageOps.contain(mask.convert("L"), (max(1, int(width * size)), max(1, int(height * size))))
        layer = Image.new("L", (size, size), 0)
        layer.paste(placed, (int(x * size), int(y * size)))
        canvas = ImageChops.lighter(canvas, layer)
    return canvas.point(lambda value: 255 if value > 96 else 0)


def candidate_summary(candidate: dict) -> dict:
    return {
        "goose_id": candidate.get("goose_id"),
        "score": candidate.get("score"),
        "flip_x": candidate.get("flip_x"),
        "cutout_path": candidate.get("cutout_path"),
        "silhouette_path": candidate.get("silhouette_path"),
        "mask_path": candidate.get("mask_path"),
    }


def safe_letter_id(letter: str) -> str:
    return f"u{ord(letter):04x}"


if __name__ == "__main__":
    main()
