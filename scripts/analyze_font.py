#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goosetype.image_features import measure_mask, render_glyph_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Rasterize a reference font and compute per-letter feature targets.")
    parser.add_argument("--font", default=None, help="Path to a .ttf/.otf font. If omitted, a system sans fallback is used.")
    parser.add_argument("--name", default="reference", help="Name to store in the generated metadata.")
    parser.add_argument("--letters", default=string.ascii_uppercase + string.ascii_lowercase, help="Letters to analyze.")
    parser.add_argument("--size", type=int, default=220, help="Raster size for each glyph target.")
    parser.add_argument("--output", default="data/processed/font_targets/reference.json")
    parser.add_argument("--mask-dir", default="data/processed/font_targets/masks")
    args = parser.parse_args()

    output = Path(args.output)
    mask_dir = Path(args.mask_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    targets = []
    seen = set()
    for letter in args.letters:
        if not letter.isalpha():
            continue
        if letter in seen:
            continue
        seen.add(letter)
        mask = render_glyph_mask(letter, args.font, size=args.size)
        glyph_id = f"u{ord(letter):04x}"
        mask_path = mask_dir / f"{args.name}_{glyph_id}.png"
        mask.save(mask_path)
        features = measure_mask(mask)
        targets.append(
            {
                "letter": letter,
                "font_name": args.name,
                "font_path": args.font,
                "mask_path": str(mask_path),
                **features,
            }
        )
        print(f"{letter}: aspect={features['aspect_ratio']:.2f} fill={features['fill_ratio']:.2f} curve={features['curvature_score']:.2f}")

    output.write_text(json.dumps({"font_name": args.name, "font_path": args.font, "targets": targets}, indent=2), encoding="utf-8")
    print(f"Wrote {len(targets)} glyph targets to {output}")


if __name__ == "__main__":
    main()
