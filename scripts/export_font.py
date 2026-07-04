#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Reserve the GooseType static font export step.")
    parser.add_argument("--output", default="outputs/fonts/README.txt")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "Static font export is intentionally deferred until glyph compositions stabilize.\n"
        "Use the web renderer and scripts/generate_glyphs.py for the first milestone.\n",
        encoding="utf-8",
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

