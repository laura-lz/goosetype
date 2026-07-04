#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from PIL import Image, ImageChops, ImageFilter, ImageStat
except ImportError as error:
    raise SystemExit("Pillow is required: python3 -m pip install Pillow") from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Create first-pass goose masks, cutouts, silhouettes, and metadata.")
    parser.add_argument("--input", default="goose_photos_square", help="Folder of raw or square goose images.")
    parser.add_argument("--output", default="data/processed", help="Processed data folder.")
    parser.add_argument("--threshold", type=float, default=54.0, help="Foreground threshold for background-distance masking.")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    masks_dir = output_dir / "masks"
    cutouts_dir = output_dir / "cutouts"
    silhouettes_dir = output_dir / "silhouettes"
    for directory in (masks_dir, cutouts_dir, silhouettes_dir):
        directory.mkdir(parents=True, exist_ok=True)

    images = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    metadata = []
    for index, image_path in enumerate(images, start=1):
        goose_id = f"goose_{index:04d}"
        image = Image.open(image_path).convert("RGBA")
        mask = build_mask(image, args.threshold)
        bbox = mask.getbbox() or (0, 0, image.width, image.height)
        cutout = Image.new("RGBA", image.size, (0, 0, 0, 0))
        cutout.paste(image, mask=mask)
        silhouette = Image.new("RGBA", image.size, (22, 28, 29, 255))
        silhouette.putalpha(mask)

        mask_path = masks_dir / f"{goose_id}.png"
        cutout_path = cutouts_dir / f"{goose_id}.png"
        silhouette_path = silhouettes_dir / f"{goose_id}.png"
        mask.save(mask_path)
        cutout.save(cutout_path)
        silhouette.save(silhouette_path)

        x0, y0, x1, y1 = bbox
        width = x1 - x0
        height = y1 - y0
        metadata.append(
            {
                "id": goose_id,
                "source_image": str(image_path),
                "cutout_path": str(cutout_path),
                "mask_path": str(mask_path),
                "silhouette_path": str(silhouette_path),
                "bbox": [x0, y0, width, height],
                "width": width,
                "height": height,
                "aspect_ratio": round(width / max(height, 1), 4),
            }
        )
        print(f"{goose_id}: {image_path.name} -> bbox={metadata[-1]['bbox']}")

    output_path = output_dir / "metadata.json"
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


def build_mask(image: Image.Image, threshold: float) -> Image.Image:
    small = image.resize((220, 220))
    rgb = small.convert("RGB")
    bg = estimate_background(rgb)
    bg_image = Image.new("RGB", rgb.size, tuple(int(value) for value in bg))
    diff = ImageChops.difference(rgb, bg_image).convert("L")

    saturation = rgb.convert("HSV").split()[1]
    center = center_weight(rgb.size)
    score = ImageChops.add(diff, saturation.point(lambda value: value * 0.5))
    score = ImageChops.add(score, center)
    mask_small = score.point(lambda value: 255 if value > threshold else 0)
    mask_small = mask_small.filter(ImageFilter.MedianFilter(5))
    mask_small = mask_small.filter(ImageFilter.MaxFilter(5))
    mask = mask_small.resize(image.size, Image.Resampling.LANCZOS)
    return mask.point(lambda value: 255 if value > 80 else 0)


def estimate_background(image: Image.Image) -> tuple[float, float, float]:
    width, height = image.size
    sample_boxes = [
        (0, 0, 18, 18),
        (width - 18, 0, width, 18),
        (0, height - 18, 18, height),
        (width - 18, height - 18, width, height),
        (width // 2 - 9, 0, width // 2 + 9, 18),
        (width // 2 - 9, height - 18, width // 2 + 9, height),
    ]
    samples = []
    for box in sample_boxes:
        stat = ImageStat.Stat(image.crop(box))
        samples.append(stat.mean)
    return tuple(sum(sample[channel] for sample in samples) / len(samples) for channel in range(3))


def center_weight(size: tuple[int, int]) -> Image.Image:
    width, height = size
    pixels = []
    for y in range(height):
        for x in range(width):
            dx = (x / width - 0.5) * 2
            dy = (y / height - 0.5) * 2
            pixels.append(int(max(0, 1 - (dx * dx + dy * dy) ** 0.5) * 26))
    image = Image.new("L", size)
    image.putdata(pixels)
    return image


if __name__ == "__main__":
    main()

