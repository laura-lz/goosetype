#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Wrap feature records in a pose-library JSON object.")
    parser.add_argument("--features", default="data/processed/features.json")
    parser.add_argument("--output", default="data/processed/pose_library.json")
    args = parser.parse_args()

    geese = json.loads(Path(args.features).read_text(encoding="utf-8"))
    payload = {"version": 1, "geese": geese}
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

