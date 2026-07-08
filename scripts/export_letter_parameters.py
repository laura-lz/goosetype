#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export readable per-letter matching parameters from a font target JSON.")
    parser.add_argument("--font-targets", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = json.loads(Path(args.font_targets).read_text(encoding="utf-8"))
    report = {
        "font_name": source.get("font_name"),
        "parameters": [summarize_target(target) for target in source.get("targets", [])],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {len(report['parameters'])} letter parameter records to {args.output}")


def summarize_target(target: dict) -> dict:
    horizontal_bands = target.get("horizontal_bands") or []
    vertical_bands = target.get("vertical_bands") or []
    return {
        "letter": target.get("letter"),
        "dimensions": {
            "aspect_ratio": target.get("aspect_ratio"),
            "bbox_width_ratio": target.get("bbox_width_ratio"),
            "bbox_height_ratio": target.get("bbox_height_ratio"),
            "fill_ratio": target.get("fill_ratio"),
        },
        "stroke_bands": {
            "horizontal": unpack_bands(horizontal_bands),
            "vertical": unpack_bands(vertical_bands),
            "axis_strength": target.get("stroke_axis_strength"),
        },
        "runs": {
            "horizontal_7": target.get("horizontal_run_signature"),
            "vertical_5": target.get("vertical_run_signature"),
        },
        "diagonals": {
            "grid_signature": target.get("diagonal_grid_signature"),
            "grid_binary": target.get("diagonal_grid_binary"),
            "band_signature": target.get("diagonal_band_signature"),
            "axis_strength": target.get("diagonal_axis_strength"),
            "corner_balance": target.get("corner_balance"),
        },
        "occupancy": {
            "grid_3x3_binary": as_grid(target.get("grid_3x3_binary") or [], rows=3, cols=3),
            "grid_3x3": as_grid(target.get("grid_3x3") or [], rows=3, cols=3),
            "grid_5x7_binary": as_grid(target.get("grid_5x7_binary") or [], rows=7, cols=5),
            "grid_5x7": as_grid(target.get("grid_5x7") or [], rows=7, cols=5),
        },
        "legacy_shape": {
            "slant_score": target.get("slant_score"),
            "boldness_score": target.get("boldness_score"),
            "curvature_score": target.get("curvature_score"),
        },
    }


def unpack_bands(values: list[float]) -> list[dict]:
    bands = []
    for index in range(0, len(values), 3):
        chunk = values[index : index + 3]
        if len(chunk) < 3 or not any(chunk):
            continue
        bands.append(
            {
                "center": chunk[0],
                "thickness": chunk[1],
                "strength": chunk[2],
            }
        )
    return bands


def as_grid(values: list, rows: int, cols: int) -> list[list]:
    grid = []
    for row in range(rows):
        start = row * cols
        grid.append(values[start : start + cols])
    return grid


if __name__ == "__main__":
    main()
