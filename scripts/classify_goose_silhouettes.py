#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw, ImageFont

from goosetype.classifier import load_glyph_classifier, mask_to_emnist_tensor
from scripts.select_candidates import load_feature_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the EMNIST-style glyph classifier directly on goose silhouettes."
    )
    parser.add_argument("--features", action="append", required=True, help="Feature JSON file. Repeat to merge datasets.")
    parser.add_argument("--model", default="data/models/emnist_glyph_classifier.pt")
    parser.add_argument("--output", default="reports/emnist_goose_predictions.json")
    parser.add_argument("--sheet", default="reports/emnist_goose_predictions.png")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--per-letter", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    classifier = load_glyph_classifier(repo / args.model, device=args.device)
    records = load_feature_records(args.features)
    predictions = []
    per_letter: dict[str, list[dict]] = {label: [] for label in classifier.labels}

    for record in records:
        path = record.get("mask_path") or record.get("silhouette_path")
        if not path:
            continue
        image_path = repo / path
        if not image_path.exists():
            continue
        mask = Image.open(image_path).convert("L")
        probabilities = score_all_labels(classifier, mask)
        top = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[: args.top_k]
        item = {
            "goose_id": record.get("id"),
            "source_image": record.get("source_image"),
            "silhouette_path": record.get("silhouette_path"),
            "mask_path": record.get("mask_path"),
            "cutout_path": record.get("cutout_path"),
            "top_predictions": [{"letter": label, "score": round(score, 6)} for label, score in top],
        }
        predictions.append(item)
        for label, score in probabilities.items():
            per_letter[label].append({**item, "letter_score": score})

    for label, items in per_letter.items():
        items.sort(key=lambda item: item["letter_score"], reverse=True)
        per_letter[label] = items[: args.per_letter]

    output = {
        "schema": "goosetype-emnist-goose-predictions-v1",
        "model": args.model,
        "feature_files": args.features,
        "goose_count": len(predictions),
        "predictions": predictions,
        "best_by_letter": {
            label: [
                {
                    "goose_id": item.get("goose_id"),
                    "score": round(item["letter_score"], 6),
                    "silhouette_path": item.get("silhouette_path"),
                    "mask_path": item.get("mask_path"),
                    "cutout_path": item.get("cutout_path"),
                    "source_image": item.get("source_image"),
                }
                for item in items
            ]
            for label, items in per_letter.items()
        },
    }
    write_json(repo / args.output, output)
    render_sheet(repo, repo / args.sheet, per_letter, args.per_letter)
    print(f"Wrote predictions to {args.output}")
    print(f"Wrote contact sheet to {args.sheet}")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def render_sheet(repo: Path, path: Path, per_letter: dict[str, list[dict]], per_letter_count: int) -> None:
    labels = list(per_letter)
    cell = 104
    label_h = 34
    cols = 13
    rows = ((len(labels) * per_letter_count) + cols - 1) // cols
    header_h = 48
    canvas = Image.new("RGB", (cols * cell, header_h + rows * (cell + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    title_font, font = load_fonts()
    draw.text((14, 12), "EMNIST-style classifier: best goose silhouettes by predicted letter", fill=(20, 20, 20), font=title_font)

    index = 0
    for label in labels:
        for item in per_letter[label]:
            col = index % cols
            row = index // cols
            x = col * cell
            y = header_h + row * (cell + label_h)
            image = load_preview(repo, item)
            canvas.paste(fit_image(image, cell), (x, y))
            draw.text((x + 6, y + cell + 2), f"{label} {item['letter_score']:.3f}", fill=(63, 72, 82), font=font)
            draw.text((x + 6, y + cell + 18), str(item.get("goose_id", ""))[-8:], fill=(119, 126, 136), font=font)
            index += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def load_preview(repo: Path, item: dict) -> Image.Image:
    image = Image.open(repo / (item.get("silhouette_path") or item.get("mask_path"))).convert("RGBA")
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    background.alpha_composite(image)
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
            ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 20),
            ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 13),
        )
    except OSError:
        font = ImageFont.load_default()
        return font, font


def score_all_labels(classifier, mask: Image.Image) -> dict[str, float]:
    import torch

    tensor = mask_to_emnist_tensor(mask).to(classifier.device)
    with torch.no_grad():
        probabilities = torch.softmax(classifier.model(tensor), dim=1)[0]
    return {label: float(probabilities[index].item()) for index, label in enumerate(classifier.labels)}


if __name__ == "__main__":
    main()
