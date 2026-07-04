#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    from PIL import Image
except ImportError as error:
    raise SystemExit("Pillow is required: python3 -m pip install Pillow") from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute GooseType geometry features from processed masks.")
    parser.add_argument("--metadata", default="data/processed/metadata.json")
    parser.add_argument("--output", default="data/processed/features.json")
    args = parser.parse_args()

    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    features = []
    for item in metadata:
        mask = Image.open(item["mask_path"]).convert("L")
        feature = {**item, **measure_mask(mask)}
        features.append(feature)
        print(f"{item['id']}: slant={feature['slant_score']:.2f} bold={feature['boldness_score']:.2f} curve={feature['curvature_score']:.2f}")

    Path(args.output).write_text(json.dumps(features, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


def measure_mask(mask: Image.Image) -> dict:
    width, height = mask.size
    values = mask.load()
    points = []
    perimeter = 0
    for y in range(height):
        for x in range(width):
            if values[x, y] <= 127:
                continue
            points.append((x, y))
            if is_edge(values, x, y, width, height):
                perimeter += 1

    if not points:
        return empty_features(width, height)

    area = len(points)
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    cx = sx / area
    cy = sy / area
    cov_xx = sum((x - cx) ** 2 for x, _ in points) / area
    cov_yy = sum((y - cy) ** 2 for _, y in points) / area
    cov_xy = sum((x - cx) * (y - cy) for x, y in points) / area
    angle = 0.5 * math.atan2(2 * cov_xy, cov_xx - cov_yy)
    major, minor = eigenvalues(cov_xx, cov_yy, cov_xy)
    bbox = mask.getbbox() or (0, 0, width, height)
    bw = max(1, bbox[2] - bbox[0])
    bh = max(1, bbox[3] - bbox[1])
    bbox_area = bw * bh
    boldness = clamp(area / bbox_area)
    thinness = clamp(perimeter / max(area, 1) * 4.8)
    curvature = clamp((perimeter * perimeter) / (max(area, 1) * 42))

    return {
        "area": area,
        "perimeter": perimeter,
        "solidity": round(boldness, 4),
        "orientation_angle": round(math.degrees(angle), 4),
        "center_of_mass": [round(cx / width, 4), round(cy / height, 4)],
        "major_axis": round(math.sqrt(max(major, 0)) / max(width, height), 4),
        "minor_axis": round(math.sqrt(max(minor, 0)) / max(width, height), 4),
        "curvature_score": round(curvature, 4),
        "thinness_score": round(thinness, 4),
        "boldness_score": round(boldness, 4),
        "slant_score": round(clamp(math.degrees(angle) / 45, -1, 1), 4),
        "pose_label": "unknown",
    }


def is_edge(values, x: int, y: int, width: int, height: int) -> bool:
    if x == 0 or y == 0 or x == width - 1 or y == height - 1:
        return True
    return (
        values[x - 1, y] <= 127
        or values[x + 1, y] <= 127
        or values[x, y - 1] <= 127
        or values[x, y + 1] <= 127
    )


def eigenvalues(a: float, d: float, b: float) -> tuple[float, float]:
    trace = a + d
    determinant = a * d - b * b
    root = math.sqrt(max(trace * trace / 4 - determinant, 0))
    return trace / 2 + root, trace / 2 - root


def empty_features(width: int, height: int) -> dict:
    return {
        "area": 0,
        "perimeter": 0,
        "solidity": 0,
        "orientation_angle": 0,
        "center_of_mass": [0.5, 0.5],
        "major_axis": 0,
        "minor_axis": 0,
        "curvature_score": 0,
        "thinness_score": 0,
        "boldness_score": 0,
        "slant_score": 0,
        "pose_label": "unknown",
    }


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


if __name__ == "__main__":
    main()

