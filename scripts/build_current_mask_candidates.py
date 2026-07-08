#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

TARGETS = {
    "i": {"aspect": 0.28, "slant": 0.0, "fill": 0.48},
    "l": {"aspect": 0.3, "slant": 0.0, "fill": 0.5},
    "t": {"aspect": 0.55, "slant": 0.0, "fill": 0.48},
    "f": {"aspect": 0.45, "slant": 0.04, "fill": 0.5},
    "j": {"aspect": 0.34, "slant": -0.03, "fill": 0.5},
    "r": {"aspect": 0.48, "slant": 0.02, "fill": 0.58},
    "m": {"aspect": 1.45, "slant": 0.0, "fill": 0.56},
    "w": {"aspect": 1.65, "slant": 0.0, "fill": 0.54},
    "n": {"aspect": 1.0, "slant": 0.0, "fill": 0.56},
    "u": {"aspect": 0.9, "slant": 0.0, "fill": 0.52},
    "o": {"aspect": 0.9, "slant": 0.0, "fill": 0.72},
    "e": {"aspect": 0.95, "slant": 0.0, "fill": 0.62},
    "a": {"aspect": 0.86, "slant": 0.0, "fill": 0.65},
    "s": {"aspect": 0.75, "slant": 0.0, "fill": 0.6},
    "c": {"aspect": 0.9, "slant": 0.0, "fill": 0.48},
    "v": {"aspect": 0.9, "slant": 0.0, "fill": 0.42},
    "x": {"aspect": 0.92, "slant": 0.0, "fill": 0.46},
    "y": {"aspect": 0.78, "slant": 0.08, "fill": 0.48},
    "z": {"aspect": 1.05, "slant": -0.05, "fill": 0.52},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a frontend candidate JSON from current validated GooseType masks.")
    parser.add_argument("--metadata", action="append", required=True, help="Mask metadata.json file. Repeat for batches.")
    parser.add_argument("--output", default="data/font_candidates/goosetype_candidates.json")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--include-review", action="store_true")
    args = parser.parse_args()

    records = load_records([Path(path) for path in args.metadata], include_review=args.include_review)
    output = build_candidates(records, top_k=args.top_k)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} mask assets into {path}")


def load_records(paths: list[Path], include_review: bool) -> list[dict]:
    rows = []
    seen = set()
    allowed = {"pass", "review"} if include_review else {"pass"}
    for path in paths:
        for item in json.loads(path.read_text(encoding="utf-8")):
            if item.get("mask_review_status", "pass") not in allowed:
                continue
            key = item.get("silhouette_path") or item.get("mask_path") or item.get("id")
            if key in seen:
                continue
            seen.add(key)
            rows.append(item)
    return rows


def build_candidates(records: list[dict], top_k: int) -> dict:
    geese = {}
    letters = {}
    for item in records:
        geese[asset_id(item)] = {
            "source_image": item.get("source_image"),
            "bbox": item.get("bbox"),
            "cutout_path": item.get("cutout_path"),
            "silhouette_path": item.get("silhouette_path"),
            "mask_path": item.get("mask_path"),
            "source_crop_id": item.get("source_crop_id"),
            "mask_review_status": item.get("mask_review_status", "pass"),
        }

    for letter in LETTERS:
        ranked = []
        for item in records:
            score = glyphish_score(item, letter)
            ranked.append(
                {
                    "goose_id": asset_id(item),
                    "source_goose_id": item.get("id"),
                    "letter": letter,
                    "letter_family": "current-mask-preview",
                    "score": round(score, 4),
                    "feature_score": round(score, 4),
                    "geometry_score": round(score, 4),
                    "quality_score": round(float(item.get("mask_quality_score") or 0), 4),
                    "flip_x": should_flip(item, letter),
                    "mask_review_status": item.get("mask_review_status", "pass"),
                }
            )
        letters[letter] = sorted(ranked, key=lambda row: row["score"], reverse=True)[:top_k]

    return {
        "schema": "goosetype-candidates-v2",
        "source": "current_mask_batches",
        "note": "Lightweight preview candidates generated from current validated masks, not final glyph matching.",
        "geese": geese,
        "letters": letters,
    }


def glyphish_score(item: dict, letter: str) -> float:
    lower = letter.lower()
    target = TARGETS.get(lower, default_target(lower))
    aspect = float(item.get("aspect_ratio") or 1)
    slant = float(item.get("slant_score") or 0)
    fill = float(item.get("fill_ratio") or 0.5)
    quality = float(item.get("mask_quality_score") or 0.8)
    goose = float(item.get("goose_candidate_score") or 0.8)
    flying_bonus = 0.08 if aspect > 1.15 and lower in "acefklrstvwxyz" else 0.0
    upright_bonus = 0.06 if aspect < 0.62 and lower in "fijlt" else 0.0
    round_bonus = 0.06 if fill > 0.62 and lower in "abcdegopqsu" else 0.0

    aspect_score = 1 - min(1, abs(aspect - target["aspect"]) / max(target["aspect"], 0.2))
    slant_score = 1 - min(1, abs(slant - target["slant"]) / 1.25)
    fill_score = 1 - min(1, abs(fill - target["fill"]) / 0.75)
    review_penalty = 0.12 if item.get("mask_review_status") == "review" else 0.0
    score = (
        aspect_score * 0.38
        + slant_score * 0.14
        + fill_score * 0.16
        + quality * 0.18
        + goose * 0.14
        + flying_bonus
        + upright_bonus
        + round_bonus
        - review_penalty
    )
    return max(0.0, min(1.0, score))


def default_target(letter: str) -> dict:
    if letter in "mw":
        return {"aspect": 1.55, "slant": 0.0, "fill": 0.52}
    if letter in "fijlt":
        return {"aspect": 0.38, "slant": 0.0, "fill": 0.5}
    if letter in "oO":
        return {"aspect": 0.9, "slant": 0.0, "fill": 0.68}
    return {"aspect": 0.9, "slant": 0.0, "fill": 0.55}


def should_flip(item: dict, letter: str) -> bool:
    slant = float(item.get("slant_score") or 0)
    return letter.lower() in "cgjsz" and slant > 0.22


def asset_id(item: dict) -> str:
    raw = f"{item.get('id')}|{item.get('silhouette_path')}|{item.get('source_crop_id')}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"{item.get('id', 'mask')}_{digest}"


if __name__ == "__main__":
    main()
