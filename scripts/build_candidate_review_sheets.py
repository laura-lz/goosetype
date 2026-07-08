#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.select_candidates import segmentation_quality_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Build review sheets for GooseType candidate and rejected masks.")
    parser.add_argument("--features", action="append", required=True)
    parser.add_argument("--candidates", default="data/font_candidates/goosetype_candidates.json")
    parser.add_argument("--output-dir", default="reports/candidate_review")
    parser.add_argument("--quality-threshold", type=float, default=0.55)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    output_dir = repo / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    features = load_features(repo, args.features)
    candidates = json.loads((repo / args.candidates).read_text(encoding="utf-8"))
    selected_assets = selected_candidate_assets(repo, candidates, features)
    candidate_pool = sorted(
        [item for item in features if item["_segmentation_quality"] >= args.quality_threshold],
        key=lambda item: (-(item.get("mask_quality_score") or 0), item.get("_features_path", ""), item.get("id", "")),
    )
    filtered = sorted(
        [item for item in features if item["_segmentation_quality"] < args.quality_threshold],
        key=lambda item: (item["_segmentation_quality"], item.get("_features_path", ""), item.get("id", "")),
    )

    sheets = {
        "selected_alphabet_assets": draw_selected_sheet(output_dir, selected_assets),
        "candidate_pool": draw_feature_sheets(
            output_dir,
            "candidate_pool",
            candidate_pool,
            f"Quality-passing candidate pool, threshold {args.quality_threshold}",
        ),
        "filtered_out_non_candidates": draw_feature_sheets(
            output_dir,
            "filtered_out_non_candidates",
            filtered,
            f"Filtered-out extracted masks below threshold {args.quality_threshold}",
            columns=4,
            per_sheet=48,
        ),
    }
    manifest = {
        "schema": "goosetype-candidate-review-v1",
        "candidate_source": args.candidates,
        "feature_sources": args.features,
        "thresholds": {"segmentation_quality": args.quality_threshold},
        "counts": {
            "extracted_total": len(features),
            "candidate_pool": len(candidate_pool),
            "filtered_out": len(filtered),
            "selected_unique_assets": len(selected_assets),
            "ranked_letter_candidates": sum(len(items) for items in candidates.get("letters", {}).values()),
        },
        "sheets": sheets,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_index(output_dir, manifest)
    print(json.dumps(manifest, indent=2))


def load_features(repo: Path, paths: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        feature_path = repo / path
        if not feature_path.exists():
            continue
        for item in json.loads(feature_path.read_text(encoding="utf-8")):
            record = dict(item)
            record["_features_path"] = str(feature_path.relative_to(repo))
            record["_segmentation_quality"] = segmentation_quality_score(record)
            record["mask_path"] = repo_relative(record.get("mask_path"), repo)
            record["cutout_path"] = repo_relative(record.get("cutout_path"), repo)
            record["silhouette_path"] = repo_relative(record.get("silhouette_path"), repo)
            rows.append(record)
    return rows


def selected_candidate_assets(repo: Path, candidates: dict, features: list[dict]) -> list[dict]:
    by_mask = {str((repo / item["mask_path"]).resolve()): item for item in features if item.get("mask_path")}
    selected: dict[str, dict] = {}
    for letter, ranked in candidates.get("letters", {}).items():
        for candidate in ranked:
            goose_id = candidate["goose_id"]
            asset = candidates.get("geese", {}).get(goose_id, {})
            row = selected.setdefault(
                goose_id,
                {
                    "goose_id": goose_id,
                    "letters": set(),
                    "score": 0.0,
                    "asset": asset,
                    "feature": None,
                },
            )
            row["letters"].add(letter)
            row["score"] = max(row["score"], float(candidate.get("score") or 0))
            mask_path = asset.get("mask_path")
            if mask_path:
                row["feature"] = by_mask.get(str((repo / mask_path).resolve()))

    rows = []
    for row in selected.values():
        rows.append({**row, "letters": "".join(sorted(row["letters"]))})
    return sorted(rows, key=lambda item: (-len(item["letters"]), item["goose_id"]))


def draw_selected_sheet(output_dir: Path, items: list[dict]) -> str:
    return draw_sheet(
        output_dir,
        "selected_alphabet_assets.png",
        f"Selected alphabet assets: {len(items)} unique geese used by ranked letters",
        items,
        label_fn=lambda item: [
            item["goose_id"][-14:],
            item["letters"][:30],
            f"score {item['score']:.3f}",
            source_label(item.get("feature")),
        ],
        image_path_fn=lambda item: item.get("asset", {}).get("silhouette_path"),
    )


def draw_feature_sheets(
    output_dir: Path,
    prefix: str,
    items: list[dict],
    title: str,
    columns: int = 8,
    per_sheet: int = 64,
) -> list[str]:
    if not items:
        return []
    paths = []
    for start in range(0, len(items), per_sheet):
        chunk = items[start : start + per_sheet]
        name = f"{prefix}_{start // per_sheet + 1:02d}.png"
        paths.append(
            draw_sheet(
                output_dir,
                name,
                f"{title}: {start + 1}-{start + len(chunk)} of {len(items)}",
                chunk,
                label_fn=lambda item: [
                    f"{item.get('id')} q={item['_segmentation_quality']:.2f}",
                    f"mask={float(item.get('mask_quality_score') or 0):.2f} goose={float(item.get('goose_candidate_score') or 0):.2f}",
                    ",".join((item.get("mask_quality_reasons") or item.get("goose_candidate_reasons") or [])[:2]),
                    source_label(item),
                ],
                image_path_fn=lambda item: item.get("silhouette_path"),
                columns=columns,
            )
        )
    return paths


def draw_sheet(
    output_dir: Path,
    filename: str,
    title: str,
    items: list[dict],
    label_fn,
    image_path_fn,
    columns: int = 8,
) -> str:
    repo = Path(__file__).resolve().parents[1]
    cell_w = 170
    image_h = 118
    label_h = 58
    header_h = 42
    rows = max(1, math.ceil(len(items) / columns))
    sheet = Image.new("RGB", (columns * cell_w, header_h + rows * (image_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    title_font, label_font = load_fonts()
    draw.text((12, 10), title, fill=(20, 20, 20), font=title_font)

    for index, item in enumerate(items):
        x = (index % columns) * cell_w
        y = header_h + (index // columns) * (image_h + label_h)
        draw.rectangle((x, y, x + cell_w - 1, y + image_h + label_h - 1), outline=(225, 225, 225))
        preview = load_preview(repo, image_path_fn(item))
        preview.thumbnail((cell_w - 18, image_h - 12), Image.Resampling.LANCZOS)
        cell = Image.new("RGBA", (cell_w, image_h), (255, 255, 255, 255))
        cell.alpha_composite(preview, ((cell_w - preview.width) // 2, (image_h - preview.height) // 2))
        sheet.paste(cell.convert("RGB"), (x, y))
        for line_index, line in enumerate(label_fn(item)[:4]):
            draw.text((x + 6, y + image_h + 4 + line_index * 13), str(line)[:28], fill=(60, 66, 72), font=label_font)

    path = output_dir / filename
    sheet.save(path)
    return str(path.relative_to(repo))


def load_preview(repo: Path, path: str | None) -> Image.Image:
    if not path:
        return Image.new("RGBA", (120, 120), (255, 255, 255, 0))
    return Image.open(repo / path).convert("RGBA")


def load_fonts():
    try:
        return (
            ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 20),
            ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 11),
        )
    except Exception:
        return None, None


def source_label(item: dict | None) -> str:
    if not item:
        return ""
    path = item.get("_features_path", "")
    return Path(path).parent.name[:28]


def repo_relative(path: str | None, repo: Path) -> str | None:
    if not path:
        return None
    value = Path(path)
    if value.is_absolute():
        try:
            return str(value.relative_to(repo))
        except ValueError:
            return str(value)
    return str(value)


def write_index(output_dir: Path, manifest: dict) -> None:
    links = []
    for value in manifest["sheets"].values():
        if isinstance(value, list):
            links.extend(value)
        elif value:
            links.append(value)
    rows = "\n".join(f'<li><a href="{Path(path).name}">{Path(path).name}</a></li>' for path in links)
    counts = manifest["counts"]
    html = f"""<!doctype html>
<meta charset="utf-8">
<title>GooseType Candidate Review</title>
<style>
body {{ font-family: Arial, sans-serif; line-height: 1.4; margin: 24px; }}
a {{ color: #16745c; }}
</style>
<h1>GooseType Candidate Review</h1>
<p>Extracted: {counts['extracted_total']}. Candidate pool: {counts['candidate_pool']}. Filtered out: {counts['filtered_out']}. Selected unique assets: {counts['selected_unique_assets']}.</p>
<ul>{rows}</ul>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
