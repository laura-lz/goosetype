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
    parser = argparse.ArgumentParser(description="Compute GooseType geometry features from extracted goose masks.")
    parser.add_argument("--metadata", default="data/processed/metadata.json")
    parser.add_argument("--output", default="data/processed/features.json")
    args = parser.parse_args()

    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    features = []
    for item in metadata:
        mask = Image.open(item["mask_path"]).convert("L")
        measured = measure_mask(mask)
        feature = {
            **item,
            **measured,
            "pose_label": item.get("pose_label", "unknown"),
        }
        features.append(feature)
        print(
            f"{item['id']}: aspect={feature['aspect_ratio']:.2f} "
            f"slant={feature['slant_score']:.2f} bold={feature['boldness_score']:.2f} "
            f"curve={feature['curvature_score']:.2f}"
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(features, indent=2), encoding="utf-8")
    print(f"Wrote {len(features)} feature records to {args.output}")


if __name__ == "__main__":
    main()
