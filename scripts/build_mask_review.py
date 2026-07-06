#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a review queue for borderline goose masks.")
    parser.add_argument(
        "--metadata",
        action="append",
        default=None,
        help="Processed metadata.json file. Repeat to merge local and external processed folders.",
    )
    parser.add_argument("--output-json", default="data/review/mask_review.json")
    parser.add_argument("--output-html", default="data/review/mask_review.html")
    parser.add_argument("--limit", type=int, default=160)
    args = parser.parse_args()

    paths = args.metadata or ["data/processed/metadata.json"]
    records = []
    for path in paths:
        source = Path(path)
        if not source.exists():
            continue
        for item in json.loads(source.read_text(encoding="utf-8")):
            record = {**item, "metadata_path": str(source)}
            record["review_priority"] = round(review_priority(record), 4)
            record["review_reasons"] = review_reasons(record)
            records.append(record)

    records = sorted(records, key=lambda item: item["review_priority"], reverse=True)[: args.limit]
    output_json = Path(args.output_json)
    output_html = Path(args.output_html)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(records, indent=2), encoding="utf-8")
    output_html.write_text(render_html(records), encoding="utf-8")
    print(f"Wrote {len(records)} review records to {output_json}")
    print(f"Wrote review sheet to {output_html}")


def review_priority(item: dict) -> float:
    score = 0.0
    goose_score = float(item.get("goose_candidate_score", 1) or 0)
    score += max(0.0, 0.75 - goose_score) * 1.3

    detection = item.get("detection") or {}
    detector_score = float(detection.get("score", 1) or 0)
    score += max(0.0, 0.42 - detector_score) * 1.1

    coverage = float(item.get("detector_mask_coverage", 1) or 0)
    score += max(0.0, 0.28 - coverage) * 1.8

    sam_support = item.get("sam_support_coverage")
    if sam_support is not None:
        score += max(0.0, 0.22 - float(sam_support or 0)) * 1.4

    reasons = item.get("goose_candidate_reasons") or []
    score += min(0.4, 0.1 * len(reasons))
    return score


def review_reasons(item: dict) -> list[str]:
    reasons = []
    goose_score = float(item.get("goose_candidate_score", 1) or 0)
    if goose_score < 0.75:
        reasons.append(f"goose_score={goose_score:.2f}")
    detection = item.get("detection") or {}
    detector_score = float(detection.get("score", 1) or 0)
    if detector_score < 0.42:
        reasons.append(f"detector_score={detector_score:.2f}")
    coverage = item.get("detector_mask_coverage")
    if coverage is not None and float(coverage) < 0.28:
        reasons.append(f"coverage={float(coverage):.2f}")
    sam_support = item.get("sam_support_coverage")
    if sam_support is not None and float(sam_support) < 0.22:
        reasons.append(f"sam_support={float(sam_support):.2f}")
    reasons.extend(item.get("goose_candidate_reasons") or [])
    return reasons or ["boundary_review"]


def render_html(records: list[dict]) -> str:
    cards = "\n".join(render_card(item) for item in records)
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>GooseType Mask Review</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 24px; color: #171b1f; }}
      h1 {{ margin: 0 0 16px; }}
      .grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); }}
      article {{ border: 1px solid #d6d0c2; border-radius: 8px; padding: 10px; background: #fff; }}
      img {{ display: block; height: 132px; object-fit: contain; width: 100%; }}
      code, small {{ color: #697178; display: block; font-size: 12px; overflow-wrap: anywhere; }}
      strong {{ display: block; margin-top: 8px; }}
      .checks {{ display: flex; gap: 8px; margin-top: 8px; }}
      label {{ font-size: 12px; }}
    </style>
  </head>
  <body>
    <h1>GooseType Mask Review</h1>
    <section class="grid">{cards}</section>
  </body>
</html>
"""


def render_card(item: dict) -> str:
    image = html.escape(item.get("silhouette_path") or item.get("cutout_path") or "")
    source = html.escape(str(item.get("source_image", "")))
    reasons = html.escape(", ".join(item.get("review_reasons", [])))
    item_id = html.escape(str(item.get("id", "")))
    priority = float(item.get("review_priority", 0))
    return f"""
      <article>
        <img src="../../{image}" alt="{item_id}" />
        <strong>{item_id}</strong>
        <small>priority {priority:.2f}</small>
        <code>{reasons}</code>
        <code>{source}</code>
        <div class="checks">
          <label><input type="checkbox" /> good</label>
          <label><input type="checkbox" /> bad</label>
          <label><input type="checkbox" /> partial</label>
        </div>
      </article>
    """


if __name__ == "__main__":
    main()
