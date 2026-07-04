from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps, ImageStat


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass(frozen=True)
class Component:
    bbox: tuple[int, int, int, int]
    area: int


def build_foreground_mask(image: Image.Image, threshold: float = 54.0) -> Image.Image:
    small = image.convert("RGBA").resize((220, 220), Image.Resampling.LANCZOS)
    rgb = small.convert("RGB")
    bg = estimate_background(rgb)
    diff = ImageChops.difference(rgb, Image.new("RGB", rgb.size, tuple(int(value) for value in bg))).convert("L")
    saturation = rgb.convert("HSV").split()[1]
    score = ImageChops.add(diff, saturation.point(lambda value: value * 0.5))
    score = ImageChops.add(score, center_weight(rgb.size))
    mask = score.point(lambda value: 255 if value > threshold else 0)
    mask = mask.filter(ImageFilter.MedianFilter(5)).filter(ImageFilter.MaxFilter(5))
    mask = mask.resize(image.size, Image.Resampling.LANCZOS)
    return mask.point(lambda value: 255 if value > 80 else 0)


def estimate_background(image: Image.Image) -> tuple[float, float, float]:
    width, height = image.size
    edge = max(8, min(width, height) // 14)
    sample_boxes = [
        (0, 0, edge, edge),
        (width - edge, 0, width, edge),
        (0, height - edge, edge, height),
        (width - edge, height - edge, width, height),
        (width // 2 - edge // 2, 0, width // 2 + edge // 2, edge),
        (width // 2 - edge // 2, height - edge, width // 2 + edge // 2, height),
    ]
    means = [ImageStat.Stat(image.crop(box)).mean for box in sample_boxes]
    return tuple(sum(mean[channel] for mean in means) / len(means) for channel in range(3))


def center_weight(size: tuple[int, int]) -> Image.Image:
    width, height = size
    pixels = []
    for y in range(height):
        for x in range(width):
            dx = (x / width - 0.5) * 2
            dy = (y / height - 0.5) * 2
            pixels.append(int(max(0, 1 - math.hypot(dx, dy)) * 26))
    image = Image.new("L", size)
    image.putdata(pixels)
    return image


def connected_components(mask: Image.Image, min_area: int = 900) -> list[Component]:
    binary = mask.convert("1")
    width, height = binary.size
    pixels = binary.load()
    seen: set[tuple[int, int]] = set()
    components: list[Component] = []

    for y in range(height):
        for x in range(width):
            if (x, y) in seen or not pixels[x, y]:
                continue
            area, bbox = flood_component(pixels, x, y, width, height, seen)
            if area >= min_area:
                components.append(Component(bbox=bbox, area=area))

    return sorted(components, key=lambda component: component.area, reverse=True)


def flood_component(pixels, start_x: int, start_y: int, width: int, height: int, seen: set[tuple[int, int]]) -> tuple[int, tuple[int, int, int, int]]:
    queue = deque([(start_x, start_y)])
    seen.add((start_x, start_y))
    area = 0
    min_x = max_x = start_x
    min_y = max_y = start_y
    while queue:
        x, y = queue.popleft()
        area += 1
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if nx < 0 or ny < 0 or nx >= width or ny >= height or (nx, ny) in seen:
                continue
            if pixels[nx, ny]:
                seen.add((nx, ny))
                queue.append((nx, ny))
    return area, (min_x, min_y, max_x + 1, max_y + 1)


def component_mask(mask: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    out = Image.new("L", mask.size, 0)
    out.paste(mask.crop(bbox), bbox)
    return out


def crop_with_padding(image: Image.Image, mask: Image.Image, bbox: tuple[int, int, int, int], padding: int) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int]]:
    x0, y0, x1, y1 = bbox
    padded = (
        max(0, x0 - padding),
        max(0, y0 - padding),
        min(image.width, x1 + padding),
        min(image.height, y1 + padding),
    )
    return image.crop(padded), mask.crop(padded), padded


def cutout_from_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    cutout = Image.new("RGBA", image.size, (0, 0, 0, 0))
    cutout.paste(image.convert("RGBA"), mask=mask)
    return cutout


def silhouette_from_mask(mask: Image.Image, color: tuple[int, int, int] = (22, 28, 29)) -> Image.Image:
    silhouette = Image.new("RGBA", mask.size, (*color, 255))
    silhouette.putalpha(mask)
    return silhouette


def measure_mask(mask: Image.Image) -> dict:
    width, height = mask.size
    values = mask.convert("L").load()
    points = []
    perimeter = 0
    for y in range(height):
        for x in range(width):
            if values[x, y] <= 127:
                continue
            points.append((x, y))
            if is_edge(values, x, y, width, height):
                perimeter += 1

    if not points:
        return empty_features()

    area = len(points)
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    cx = sx / area
    cy = sy / area
    cov_xx = sum((x - cx) ** 2 for x, _ in points) / area
    cov_yy = sum((y - cy) ** 2 for _, y in points) / area
    cov_xy = sum((x - cx) * (y - cy) for x, y in points) / area
    angle = 0.5 * math.atan2(2 * cov_xy, cov_xx - cov_yy)
    major, minor = eigenvalues(cov_xx, cov_yy, cov_xy)
    bbox = mask.getbbox() or (0, 0, width, height)
    bbox_width = max(1, bbox[2] - bbox[0])
    bbox_height = max(1, bbox[3] - bbox[1])
    bbox_area = bbox_width * bbox_height
    boldness = clamp(area / bbox_area)
    thinness = clamp(perimeter / max(area, 1) * 4.8)
    curvature = clamp((perimeter * perimeter) / (max(area, 1) * 42))

    return {
        "area": area,
        "perimeter": perimeter,
        "solidity": round(boldness, 4),
        "orientation_angle": round(math.degrees(angle), 4),
        "center_of_mass": [round(cx / width, 4), round(cy / height, 4)],
        "major_axis": round(math.sqrt(max(major, 0)) / max(width, height), 4),
        "minor_axis": round(math.sqrt(max(minor, 0)) / max(width, height), 4),
        "curvature_score": round(curvature, 4),
        "thinness_score": round(thinness, 4),
        "boldness_score": round(boldness, 4),
        "slant_score": round(clamp(math.degrees(angle) / 45, -1, 1), 4),
        "aspect_ratio": round(bbox_width / max(bbox_height, 1), 4),
        "fill_ratio": round(area / bbox_area, 4),
        "bbox": [bbox[0], bbox[1], bbox_width, bbox_height],
    }


def is_edge(values, x: int, y: int, width: int, height: int) -> bool:
    if x == 0 or y == 0 or x == width - 1 or y == height - 1:
        return True
    return (
        values[x - 1, y] <= 127
        or values[x + 1, y] <= 127
        or values[x, y - 1] <= 127
        or values[x, y + 1] <= 127
    )


def eigenvalues(a: float, d: float, b: float) -> tuple[float, float]:
    trace = a + d
    determinant = a * d - b * b
    root = math.sqrt(max(trace * trace / 4 - determinant, 0))
    return trace / 2 + root, trace / 2 - root


def empty_features() -> dict:
    return {
        "area": 0,
        "perimeter": 0,
        "solidity": 0,
        "orientation_angle": 0,
        "center_of_mass": [0.5, 0.5],
        "major_axis": 0,
        "minor_axis": 0,
        "curvature_score": 0,
        "thinness_score": 0,
        "boldness_score": 0,
        "slant_score": 0,
        "aspect_ratio": 1,
        "fill_ratio": 0,
        "bbox": [0, 0, 1, 1],
    }


def render_glyph_mask(letter: str, font_path: str | Path | None, size: int = 220) -> Image.Image:
    font = load_font(font_path, int(size * 0.78))
    image = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), letter, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (size - text_width) // 2 - bbox[0]
    y = (size - text_height) // 2 - bbox[1]
    draw.text((x, y), letter, fill=255, font=font)
    return image.point(lambda value: 255 if value > 20 else 0)


def load_font(font_path: str | Path | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if font_path:
        return ImageFont.truetype(str(font_path), size=size)
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def mask_iou(a: Image.Image, b: Image.Image, size: int = 160) -> dict:
    mask_a = ImageOps.contain(a.convert("L"), (size, size))
    mask_b = ImageOps.contain(b.convert("L"), (size, size))
    canvas_a = Image.new("L", (size, size), 0)
    canvas_b = Image.new("L", (size, size), 0)
    canvas_a.paste(mask_a, ((size - mask_a.width) // 2, (size - mask_a.height) // 2))
    canvas_b.paste(mask_b, ((size - mask_b.width) // 2, (size - mask_b.height) // 2))
    pixels_a = canvas_a.load()
    pixels_b = canvas_b.load()
    intersection = union = identical = 0
    total = size * size
    for y in range(size):
        for x in range(size):
            on_a = pixels_a[x, y] > 127
            on_b = pixels_b[x, y] > 127
            if on_a and on_b:
                intersection += 1
            if on_a or on_b:
                union += 1
            if on_a == on_b:
                identical += 1
    return {
        "iou": round(intersection / union, 4) if union else 0,
        "identical_pixel_ratio": round(identical / total, 4),
    }


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))

