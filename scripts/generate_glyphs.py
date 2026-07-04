#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from goosetype import PoseLibrary, StyleConfig, generate_glyph


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate debug GooseType glyph compositions.")
    parser.add_argument("--features", default="data/processed/features.json")
    parser.add_argument("--text", default="GOOSE")
    parser.add_argument("--readability", type=float, default=0.8)
    parser.add_argument("--abstraction", type=float, default=0.2)
    parser.add_argument("--boldness", type=float, default=0.5)
    parser.add_argument("--slant", type=float, default=0.0)
    parser.add_argument("--width", type=float, default=0.5)
    parser.add_argument("--cursive", type=float, default=0.0)
    parser.add_argument("--max-geese", type=int, default=2)
    args = parser.parse_args()

    style = StyleConfig(
        readability=args.readability,
        abstraction=args.abstraction,
        boldness=args.boldness,
        slant=args.slant,
        width=args.width,
        cursive=args.cursive,
        max_geese_per_glyph=args.max_geese,
    )
    library = PoseLibrary.from_json(args.features)
    compositions = [
        generate_glyph(letter, library, style, seed=index).to_dict()
        for index, letter in enumerate(args.text.upper())
        if letter.isalpha()
    ]
    print(json.dumps(compositions, indent=2))


if __name__ == "__main__":
    main()

