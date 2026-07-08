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

from goosetype.image_features import measure_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute features.json beside every processed metadata.json.")
    parser.add_argument("--root", default="data", help="Data root containing processed folders.")
    parser.add_argument(
        "--metadata",
        action="append",
        help="Specific metadata.json file to process. Repeat to override auto-discovery.",
    )
    args = parser.parse_args()

    paths = [Path(path) for path in args.metadata] if args.metadata else discover_metadata_paths(Path(args.root))

    total = 0
    for metadata_path in paths:
        if not metadata_path.exists():
            continue
        records = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            continue
        features = []
        repaired_records = []
        for item in records:
            mask_path = Path(item.get("mask_path") or inferred_asset_path(metadata_path, item, "masks"))
            if not mask_path.exists():
                continue
            measured = measure_mask(Image.open(mask_path).convert("L"))
            item = with_asset_paths(metadata_path, item)
            source_bbox = item.get("bbox")
            measured_bbox = measured.pop("bbox", None)
            repaired_records.append(item)
            features.append(
                {
                    **item,
                    **measured,
                    "bbox": source_bbox,
                    "mask_bbox": measured_bbox,
                    "pose_label": item.get("pose_label", "unknown"),
                }
            )
        output_path = metadata_path.with_name("features.json")
        metadata_path.write_text(json.dumps(repaired_records, indent=2), encoding="utf-8")
        output_path.write_text(json.dumps(features, indent=2), encoding="utf-8")
        total += len(features)
        print(f"{output_path}: {len(features)}")

    print(f"Recomputed {total} feature records.")


def discover_metadata_paths(root: Path) -> list[Path]:
    candidates = sorted(path for path in root.glob("instances_v2*/metadata.json") if path.is_file())
    seen = set()
    paths = []
    for path in candidates:
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)
    return paths


def with_asset_paths(metadata_path: Path, item: dict) -> dict:
    result = dict(item)
    result.setdefault("mask_path", str(inferred_asset_path(metadata_path, item, "masks")))
    result.setdefault("cutout_path", str(inferred_asset_path(metadata_path, item, "cutouts")))
    result.setdefault("silhouette_path", str(inferred_asset_path(metadata_path, item, "silhouettes")))
    return result


def inferred_asset_path(metadata_path: Path, item: dict, folder: str) -> Path:
    return metadata_path.parent / folder / f"{item['id']}.png"


if __name__ == "__main__":
    main()
