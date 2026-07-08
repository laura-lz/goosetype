#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw, ImageFont


LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GooseType matching audit reports from the canonical candidate JSON.")
    parser.add_argument("--candidates", default="data/font_candidates/goosetype_candidates.json")
    parser.add_argument("--output-dir", default="reports/matching")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    data = json.loads((repo / args.candidates).read_text(encoding="utf-8"))
    candidates = normalize_candidate_data(data)
    output_dir = repo / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    final_picks = pick_by(candidates, lambda item: float(item.get("score") or 0))
    mask_picks = pick_by(candidates, mask_score)
    emnist_picks = pick_by(candidates, lambda item: float(item.get("classifier_score") or 0))
    quality_picks = pick_by(candidates, lambda item: float(item.get("quality_score") or 0))
    disagreements = build_disagreements(candidates, final_picks, mask_picks, emnist_picks)

    render_sheet(repo, output_dir / "final_rank_sheet.png", "Final GooseType ranking", final_picks)
    render_sheet(repo, output_dir / "mask_rank_sheet.png", "Mask-only ranking", mask_picks, score_fn=mask_score)
    render_sheet(repo, output_dir / "emnist_rank_sheet.png", "EMNIST-only ranking", emnist_picks, score_fn=lambda item: float(item.get("classifier_score") or 0))
    render_triad_sheet(repo, output_dir / "ranker_disagreement_sheet.png", final_picks, mask_picks, emnist_picks)
    render_sheet(repo, output_dir / "quality_rank_sheet.png", "Best retained mask quality by letter", quality_picks, score_fn=lambda item: float(item.get("quality_score") or 0))

    summary = {
        "schema": "goosetype-matching-audit-v1",
        "candidate_source": args.candidates,
        "letter_count": len(candidates),
        "candidate_count": sum(len(items) for items in candidates.values()),
        "metrics": {
            "mask_score": "0.34*positive_border + 0.30*negative_space + 0.14*precision + 0.12*iou + 0.06*dice + 0.04*centered_iou",
            "disagreement": "rank-position spread among final, mask-only, and EMNIST-only picks",
        },
        "disagreements": disagreements,
        "best_by_letter": {
            letter: {
                "final": slim_candidate(final_picks.get(letter)),
                "mask": slim_candidate(mask_picks.get(letter), extra_score=mask_score),
                "emnist": slim_candidate(emnist_picks.get(letter), extra_score=lambda item: float(item.get("classifier_score") or 0)),
                "quality": slim_candidate(quality_picks.get(letter), extra_score=lambda item: float(item.get("quality_score") or 0)),
            }
            for letter in LETTERS
            if letter in candidates
        },
    }
    (output_dir / "matching_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_html(output_dir / "index.html", summary)
    print(f"Wrote matching reports to {output_dir.relative_to(repo)}")


def normalize_candidate_data(data: dict) -> dict[str, list[dict]]:
    geese = data.get("geese", {})
    normalized = {}
    for letter, ranked in (data.get("letters") or {}).items():
        normalized[letter] = [
            {
                **(geese.get(candidate.get("goose_id")) or {}),
                **candidate,
                "letter": letter,
                "rank_index": index,
            }
            for index, candidate in enumerate(ranked)
        ]
    return normalized


def pick_by(candidates: dict[str, list[dict]], score_fn) -> dict[str, dict]:
    picks = {}
    for letter, items in candidates.items():
        if not items:
            continue
        picks[letter] = max(items, key=score_fn)
    return picks


def mask_score(item: dict) -> float:
    return (
        float(item.get("positive_border_score") or 0) * 0.34
        + float(item.get("negative_space_score") or 0) * 0.30
        + float(item.get("aligned_precision") or 0) * 0.14
        + float(item.get("aligned_iou") or 0) * 0.12
        + float(item.get("aligned_dice") or 0) * 0.06
        + float(item.get("centered_iou") or 0) * 0.04
    )


def build_disagreements(candidates: dict[str, list[dict]], final_picks: dict, mask_picks: dict, emnist_picks: dict) -> list[dict]:
    rows = []
    for letter, items in candidates.items():
        rank_by_id = {candidate_key(item): index for index, item in enumerate(items)}
        chosen = {
            "final": final_picks.get(letter),
            "mask": mask_picks.get(letter),
            "emnist": emnist_picks.get(letter),
        }
        ranks = [rank_by_id.get(candidate_key(item), 99) for item in chosen.values() if item]
        spread = max(ranks) - min(ranks) if ranks else 0
        distinct = len({candidate_key(item) for item in chosen.values() if item})
        rows.append(
            {
                "letter": letter,
                "spread": spread,
                "distinct_picks": distinct,
                "final": slim_candidate(chosen["final"]),
                "mask": slim_candidate(chosen["mask"], extra_score=mask_score),
                "emnist": slim_candidate(chosen["emnist"], extra_score=lambda item: float(item.get("classifier_score") or 0)),
            }
        )
    return sorted(rows, key=lambda row: (row["distinct_picks"], row["spread"]), reverse=True)


def candidate_key(item: dict | None) -> tuple:
    if not item:
        return ("", False)
    return (item.get("goose_id"), bool(item.get("flip_x")))


def slim_candidate(item: dict | None, extra_score=None) -> dict | None:
    if not item:
        return None
    data = {
        "goose_id": item.get("goose_id"),
        "score": round(float(item.get("score") or 0), 4),
        "rank_index": item.get("rank_index"),
        "flip_x": bool(item.get("flip_x")),
        "quality_score": round(float(item.get("quality_score") or 0), 4),
        "mask_score": round(mask_score(item), 4),
        "classifier_score": round(float(item.get("classifier_score") or 0), 4),
        "silhouette_path": item.get("silhouette_path"),
        "mask_path": item.get("mask_path"),
        "cutout_path": item.get("cutout_path"),
    }
    if extra_score:
        data["report_score"] = round(float(extra_score(item)), 4)
    return data


def render_sheet(repo: Path, path: Path, title: str, picks: dict[str, dict], score_fn=None) -> None:
    cell = 128
    label_h = 42
    cols = 13
    rows = 4
    header_h = 48
    canvas = Image.new("RGB", (cols * cell, header_h + rows * (cell + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    title_font, font = load_fonts()
    draw.text((14, 12), title, fill=(20, 20, 20), font=title_font)

    for index, letter in enumerate(LETTERS):
        item = picks.get(letter)
        if not item:
            continue
        x = (index % cols) * cell
        y = header_h + (index // cols) * (cell + label_h)
        canvas.paste(fit_image(load_preview(repo, item), cell), (x, y))
        value = score_fn(item) if score_fn else float(item.get("score") or 0)
        draw.text((x + 6, y + cell + 3), f"{letter} {value:.3f}", fill=(63, 72, 82), font=font)
        draw.text((x + 6, y + cell + 20), str(item.get("goose_id", ""))[-8:], fill=(119, 126, 136), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def render_triad_sheet(repo: Path, path: Path, final_picks: dict, mask_picks: dict, emnist_picks: dict) -> None:
    cell = 92
    label_w = 42
    row_h = cell + 20
    header_h = 58
    width = label_w + cell * 3
    height = header_h + len(LETTERS) * row_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font, font = load_fonts()
    draw.text((10, 10), "Per-letter ranker disagreement: final / mask / EMNIST", fill=(20, 20, 20), font=title_font)
    draw.text((label_w + 16, 36), "final", fill=(63, 72, 82), font=font)
    draw.text((label_w + cell + 18, 36), "mask", fill=(63, 72, 82), font=font)
    draw.text((label_w + cell * 2 + 12, 36), "emnist", fill=(63, 72, 82), font=font)
    for row, letter in enumerate(LETTERS):
        y = header_h + row * row_h
        draw.text((12, y + 34), letter, fill=(20, 20, 20), font=title_font)
        for col, item in enumerate((final_picks.get(letter), mask_picks.get(letter), emnist_picks.get(letter))):
            if not item:
                continue
            x = label_w + col * cell
            canvas.paste(fit_image(load_preview(repo, item), cell), (x, y))
            draw.text((x + 5, y + cell + 1), str(item.get("goose_id", ""))[-8:], fill=(119, 126, 136), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def load_preview(repo: Path, item: dict) -> Image.Image:
    source = item.get("silhouette_path") or item.get("mask_path") or item.get("cutout_path")
    image = Image.open(repo / source).convert("RGBA")
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    background.alpha_composite(image)
    if item.get("flip_x"):
        background = background.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return background.convert("RGB")


def fit_image(image: Image.Image, size: int) -> Image.Image:
    canvas = Image.new("RGB", (size, size), "white")
    image = image.copy()
    image.thumbnail((size - 12, size - 12), Image.Resampling.LANCZOS)
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def load_fonts():
    try:
        return (
            ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 18),
            ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 13),
        )
    except OSError:
        font = ImageFont.load_default()
        return font, font


def write_html(path: Path, summary: dict) -> None:
    worst = summary["disagreements"][:18]
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(row['letter'])}</td>"
        f"<td>{row['distinct_picks']}</td>"
        f"<td>{row['spread']}</td>"
        f"<td>{html.escape(str((row['final'] or {}).get('goose_id', '')))}</td>"
        f"<td>{html.escape(str((row['mask'] or {}).get('goose_id', '')))}</td>"
        f"<td>{html.escape(str((row['emnist'] or {}).get('goose_id', '')))}</td>"
        "</tr>"
        for row in worst
    )
    body = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>GooseType Matching Audit</title>
    <style>
      body {{ font-family: Arial, Helvetica, sans-serif; margin: 24px; color: #171b1f; }}
      main {{ max-width: 1120px; margin: 0 auto; }}
      img {{ display: block; max-width: 100%; border: 1px solid #d8dee5; margin: 12px 0 28px; }}
      table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
      td, th {{ border-bottom: 1px solid #e4e8ee; padding: 8px; text-align: left; }}
      code {{ background: #f2f4f7; padding: 2px 4px; border-radius: 4px; }}
    </style>
  </head>
  <body>
    <main>
      <h1>GooseType Matching Audit</h1>
      <p>{summary['candidate_count']} retained candidates across {summary['letter_count']} letters.</p>
      <p><a href="matching_audit.json">Download JSON audit</a></p>
      <h2>Final Ranking</h2>
      <img src="final_rank_sheet.png" alt="Final GooseType ranking sheet" />
      <h2>Mask-Only Ranking</h2>
      <img src="mask_rank_sheet.png" alt="Mask-only ranking sheet" />
      <h2>EMNIST-Only Ranking</h2>
      <img src="emnist_rank_sheet.png" alt="EMNIST-only ranking sheet" />
      <h2>Ranker Disagreement</h2>
      <img src="ranker_disagreement_sheet.png" alt="Final, mask, and EMNIST comparison sheet" />
      <h2>Highest Quality Retained Masks</h2>
      <img src="quality_rank_sheet.png" alt="Quality ranking sheet" />
      <h2>Largest Disagreements</h2>
      <table>
        <thead><tr><th>Letter</th><th>Distinct picks</th><th>Rank spread</th><th>Final</th><th>Mask</th><th>EMNIST</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
      <p>Mask score: <code>{html.escape(summary['metrics']['mask_score'])}</code></p>
    </main>
  </body>
</html>
"""
    path.write_text(body, encoding="utf-8")


if __name__ == "__main__":
    main()
