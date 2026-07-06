from __future__ import annotations

import math
import os
from io import BytesIO
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps, ImageStat

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif"}
_REMBG_SESSIONS = {}
_OWL_VIT_DETECTOR = None
_SAM_MODELS = {}


@dataclass(frozen=True)
class Component:
    bbox: tuple[int, int, int, int]
    area: int


@dataclass(frozen=True)
class Detection:
    bbox: tuple[int, int, int, int]
    score: float
    label: str


def build_foreground_mask(
    image: Image.Image,
    threshold: float = 54.0,
    backend: str = "auto",
    rembg_model: str = "isnet-general-use",
    max_segmentation_side: int = 1600,
) -> Image.Image:
    backend = backend.lower()
    if backend not in {"auto", "sam", "rembg", "grabcut", "heuristic"}:
        raise ValueError(f"Unknown segmentation backend: {backend}")

    errors = []
    if backend in {"auto", "rembg"}:
        try:
            return build_rembg_mask(image, model_name=rembg_model, max_side=max_segmentation_side)
        except Exception as error:
            if backend == "rembg":
                raise
            errors.append(f"rembg unavailable: {error}")

    if backend in {"auto", "grabcut"}:
        try:
            return build_grabcut_mask(image)
        except Exception as error:
            if backend == "grabcut":
                raise
            errors.append(f"grabcut unavailable: {error}")

    if errors and backend == "auto":
        print("Segmentation fallback:", "; ".join(errors))
    return build_heuristic_foreground_mask(image, threshold=threshold)


def build_instance_mask(
    image: Image.Image,
    bbox: tuple[int, int, int, int] | None = None,
    backend: str = "auto",
    threshold: float = 54.0,
    rembg_model: str = "isnet-general-use",
    sam_model: str = "facebook/sam-vit-base",
    max_segmentation_side: int = 1600,
) -> Image.Image:
    backend = backend.lower()
    if backend == "sam":
        if bbox is None:
            raise ValueError("SAM segmentation requires a box prompt.")
        return build_sam_box_mask(image, bbox=bbox, model_name=sam_model, max_side=max_segmentation_side)
    return build_foreground_mask(
        image,
        threshold=threshold,
        backend=backend,
        rembg_model=rembg_model,
        max_segmentation_side=max_segmentation_side,
    )


def build_detector_guided_mask(
    image: Image.Image,
    detector_backend: str = "owlvit",
    detection_queries: tuple[str, ...] = ("goose", "geese"),
    detection_threshold: float = 0.25,
    segmentation_backend: str = "rembg",
    rembg_model: str = "isnet-general-use",
    max_segmentation_side: int = 1600,
    box_padding_ratio: float = 0.08,
) -> tuple[Image.Image, list[Detection]]:
    detections = detect_goose_boxes(
        image,
        backend=detector_backend,
        queries=detection_queries,
        threshold=detection_threshold,
        max_side=max_segmentation_side,
    )
    if not detections:
        return Image.new("L", image.size, 0), []

    full_mask = Image.new("L", image.size, 0)
    for detection in detections:
        crop_box = pad_bbox(detection.bbox, image.size, box_padding_ratio)
        crop = image.crop(crop_box)
        crop_mask = build_foreground_mask(
            crop,
            backend=segmentation_backend,
            rembg_model=rembg_model,
            max_segmentation_side=max_segmentation_side,
        )
        full_mask.paste(crop_mask, crop_box)
    return clean_mask(full_mask), detections


def detect_goose_boxes(
    image: Image.Image,
    backend: str = "owlvit",
    queries: tuple[str, ...] = ("goose", "geese"),
    threshold: float = 0.25,
    max_side: int = 1600,
) -> list[Detection]:
    backend = backend.lower()
    if backend in {"none", "off"}:
        return []
    if backend != "owlvit":
        raise ValueError(f"Unknown detector backend: {backend}")
    return detect_goose_boxes_owlvit(image, queries=queries, threshold=threshold, max_side=max_side)


def detect_goose_boxes_owlvit(
    image: Image.Image,
    queries: tuple[str, ...] = ("goose", "geese"),
    threshold: float = 0.25,
    max_side: int = 1600,
) -> list[Detection]:
    global _OWL_VIT_DETECTOR

    try:
        from transformers import pipeline
    except ImportError as error:
        raise RuntimeError(
            "OWL-ViT detection requires the optional ML dependencies: "
            "python3 -m pip install -r requirements-ml.txt"
        ) from error

    original_size = image.size
    source = image.convert("RGB")
    scale_x = scale_y = 1.0
    if max_side > 0 and max(source.size) > max_side:
        resized = ImageOps.contain(source, (max_side, max_side), Image.Resampling.LANCZOS)
        scale_x = original_size[0] / resized.width
        scale_y = original_size[1] / resized.height
        source = resized

    if _OWL_VIT_DETECTOR is None:
        _OWL_VIT_DETECTOR = pipeline(
            task="zero-shot-object-detection",
            model="google/owlvit-base-patch32",
            device=-1,
        )

    results = _OWL_VIT_DETECTOR(source, candidate_labels=list(queries))
    detections: list[Detection] = []
    for result in results:
        score = float(result.get("score", 0))
        if score < threshold:
            continue
        box = result.get("box", {})
        x0 = int(round(float(box.get("xmin", 0)) * scale_x))
        y0 = int(round(float(box.get("ymin", 0)) * scale_y))
        x1 = int(round(float(box.get("xmax", source.width)) * scale_x))
        y1 = int(round(float(box.get("ymax", source.height)) * scale_y))
        bbox = clamp_bbox((x0, y0, x1, y1), original_size)
        if bbox[2] - bbox[0] < 12 or bbox[3] - bbox[1] < 12:
            continue
        detections.append(Detection(bbox=bbox, score=round(score, 4), label=str(result.get("label", ""))))

    return merge_overlapping_detections(detections)


def build_rembg_mask(image: Image.Image, model_name: str = "isnet-general-use", max_side: int = 1600) -> Image.Image:
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/goosetype_numba_cache")
    from rembg import new_session, remove

    session = _REMBG_SESSIONS.get(model_name)
    if session is None:
        session = new_session(model_name)
        _REMBG_SESSIONS[model_name] = session

    original_size = image.size
    source = image.convert("RGBA")
    if max_side > 0 and max(source.size) > max_side:
        source = ImageOps.contain(source, (max_side, max_side), Image.Resampling.LANCZOS)

    result = remove(source, session=session)
    if isinstance(result, bytes):
        result = Image.open(BytesIO(result)).convert("RGBA")
    alpha = result.convert("RGBA").getchannel("A")
    if alpha.size != original_size:
        alpha = alpha.resize(original_size, Image.Resampling.LANCZOS)
    return clean_mask(alpha)


def build_sam_box_mask(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    model_name: str = "facebook/sam-vit-base",
    max_side: int = 1600,
) -> Image.Image:
    global _SAM_MODELS

    try:
        import torch
        from transformers import SamModel, SamProcessor
    except ImportError as error:
        raise RuntimeError(
            "SAM segmentation requires the optional ML dependencies: "
            "python3 -m pip install -r requirements-ml.txt"
        ) from error

    source = image.convert("RGB")
    prompt_box = clamp_bbox(bbox, source.size)
    if max_side > 0 and max(source.size) > max_side:
        resized = ImageOps.contain(source, (max_side, max_side), Image.Resampling.LANCZOS)
        scale_x = resized.width / source.width
        scale_y = resized.height / source.height
        prompt_box = (
            int(round(prompt_box[0] * scale_x)),
            int(round(prompt_box[1] * scale_y)),
            int(round(prompt_box[2] * scale_x)),
            int(round(prompt_box[3] * scale_y)),
        )
        source = resized

    cached = _SAM_MODELS.get(model_name)
    if cached is None:
        processor = SamProcessor.from_pretrained(model_name)
        model = SamModel.from_pretrained(model_name)
        model.eval()
        cached = (processor, model)
        _SAM_MODELS[model_name] = cached
    processor, model = cached

    inputs = processor(source, input_boxes=[[[list(prompt_box)]]], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    original_sizes = inputs["original_sizes"]
    reshaped_input_sizes = inputs["reshaped_input_sizes"]
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        original_sizes.cpu(),
        reshaped_input_sizes.cpu(),
    )[0]
    scores = outputs.iou_scores.cpu()[0, 0]
    best_index = int(scores.argmax().item())
    mask_tensor = masks[0, best_index]
    mask = Image.fromarray((mask_tensor.numpy().astype("uint8") * 255), mode="L")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)
    return clean_mask(mask)


def build_grabcut_mask(image: Image.Image) -> Image.Image:
    import cv2
    import numpy as np

    rgb = image.convert("RGB")
    array = np.array(rgb)
    height, width = array.shape[:2]
    inset_x = max(8, int(width * 0.06))
    inset_y = max(8, int(height * 0.06))
    rect = (inset_x, inset_y, max(1, width - inset_x * 2), max(1, height - inset_y * 2))
    mask = np.zeros((height, width), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(array, mask, rect, bgd_model, fgd_model, 6, cv2.GC_INIT_WITH_RECT)
    binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
    kernel_size = max(3, min(width, height) // 90)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return clean_mask(Image.fromarray(binary, mode="L"))


def build_heuristic_foreground_mask(image: Image.Image, threshold: float = 54.0) -> Image.Image:
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
    return clean_mask(mask.point(lambda value: 255 if value > 80 else 0))


def clean_mask(mask: Image.Image) -> Image.Image:
    cleaned = mask.convert("L")
    cleaned = cleaned.filter(ImageFilter.MedianFilter(5))
    cleaned = cleaned.filter(ImageFilter.MaxFilter(3))
    cleaned = cleaned.filter(ImageFilter.MinFilter(3))
    return cleaned.point(lambda value: 255 if value > 96 else 0)


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
    try:
        return connected_components_cv2(mask, min_area=min_area)
    except Exception:
        return connected_components_python(mask, min_area=min_area)


def connected_components_cv2(mask: Image.Image, min_area: int = 900) -> list[Component]:
    import cv2
    import numpy as np

    array = np.array(mask.convert("L"))
    binary = np.where(array > 127, 255, 0).astype("uint8")
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components: list[Component] = []
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        if int(area) >= min_area:
            components.append(Component(bbox=(int(x), int(y), int(x + width), int(y + height)), area=int(area)))
    return sorted(components, key=lambda component: component.area, reverse=True)


def connected_components_python(mask: Image.Image, min_area: int = 900) -> list[Component]:
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


def pad_bbox(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    padding = int(round(max(x1 - x0, y1 - y0) * padding_ratio))
    return clamp_bbox((x0 - padding, y0 - padding, x1 + padding, y1 + padding), image_size)


def clamp_bbox(bbox: tuple[int, int, int, int], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = image_size
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(x0 + 1, min(width, x1))
    y1 = max(y0 + 1, min(height, y1))
    return x0, y0, x1, y1


def merge_overlapping_detections(detections: list[Detection], iou_threshold: float = 0.55) -> list[Detection]:
    merged: list[Detection] = []
    for detection in sorted(detections, key=lambda item: item.score, reverse=True):
        if any(bbox_iou(detection.bbox, existing.bbox) >= iou_threshold for existing in merged):
            continue
        merged.append(detection)
    return merged


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    intersection = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = max(1, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1, (bx1 - bx0) * (by1 - by0))
    return intersection / max(1, area_a + area_b - intersection)


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
    grid_3x3 = mask_grid_signature(mask)
    projection_x, projection_y = mask_projection_signature(mask)
    contour_grid_4x4 = mask_contour_grid_signature(mask)
    shape_context = mask_shape_context_signature(mask)

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
        "grid_3x3": grid_3x3,
        "grid_3x3_binary": [1 if value >= 0.16 else 0 for value in grid_3x3],
        "projection_x": projection_x,
        "projection_y": projection_y,
        "contour_grid_4x4": contour_grid_4x4,
        "shape_context": shape_context,
        "bbox": [bbox[0], bbox[1], bbox_width, bbox_height],
    }


def score_goose_candidate(
    features: dict,
    image_size: tuple[int, int],
    bbox: tuple[int, int, int, int] | None = None,
) -> dict:
    image_width, image_height = image_size
    image_area = max(1, image_width * image_height)
    bbox_values = bbox or tuple(features.get("bbox", [0, 0, 1, 1]))
    x0, y0, x1, y1 = normalize_bbox(bbox_values)
    bbox_width = max(1, x1 - x0)
    bbox_height = max(1, y1 - y0)
    bbox_area_ratio = (bbox_width * bbox_height) / image_area
    aspect_ratio = float(features.get("aspect_ratio", bbox_width / bbox_height))
    fill_ratio = float(features.get("fill_ratio", 0))
    thinness = float(features.get("thinness_score", 0))
    curvature = float(features.get("curvature_score", 0))
    area = float(features.get("area", 0))
    border_contact = border_contact_ratio((x0, y0, x1, y1), image_size)

    score = 1.0
    reasons: list[str] = []

    checks = [
        (area >= image_area * 0.0015, 0.45, "too_small"),
        (bbox_area_ratio <= 0.82, 0.45, "too_large_for_single_goose"),
        (0.22 <= aspect_ratio <= 4.8, 0.55, "implausible_aspect_ratio"),
        (0.07 <= fill_ratio <= 0.78, 0.45, "implausible_fill_ratio"),
        (thinness <= 0.95, 0.55, "too_thin_or_branchlike"),
        (border_contact <= 0.52, 0.3, "touches_image_border_too_much"),
    ]
    for passed, penalty, reason in checks:
        if not passed:
            score -= penalty
            reasons.append(reason)

    # Moderately curved/irregular silhouettes are useful for geese; extreme contour
    # complexity is often grass, shrubs, or noisy background.
    if curvature > 0.98 and fill_ratio < 0.18:
        score -= 0.35
        reasons.append("noisy_sparse_contour")
    if bbox_width < 24 or bbox_height < 24:
        score -= 0.3
        reasons.append("bbox_too_small")

    return {
        "goose_candidate_score": round(clamp(score), 4),
        "goose_candidate_reasons": reasons,
        "bbox_area_ratio": round(bbox_area_ratio, 4),
        "border_contact_ratio": round(border_contact, 4),
        "is_probable_goose": score >= 0.5,
    }


def normalize_bbox(bbox: tuple[int, int, int, int] | list[int]) -> tuple[int, int, int, int]:
    x0, y0, third, fourth = [int(value) for value in bbox]
    if third <= x0 or fourth <= y0:
        return x0, y0, x0 + max(1, third), y0 + max(1, fourth)
    return x0, y0, third, fourth


def border_contact_ratio(bbox: tuple[int, int, int, int], image_size: tuple[int, int]) -> float:
    x0, y0, x1, y1 = bbox
    width, height = image_size
    touches = 0
    touches += int(x0 <= 1)
    touches += int(y0 <= 1)
    touches += int(x1 >= width - 1)
    touches += int(y1 >= height - 1)
    return touches / 4


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
        "grid_3x3": [0.0] * 9,
        "grid_3x3_binary": [0] * 9,
        "projection_x": [0.0] * 16,
        "projection_y": [0.0] * 16,
        "contour_grid_4x4": [0.0] * 16,
        "shape_context": [0.0] * 32,
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


def aligned_mask_similarity(a: Image.Image, b: Image.Image, size: int = 160) -> dict:
    canvas_a = normalize_mask_to_bbox(a, size=size)
    canvas_b = normalize_mask_to_bbox(b, size=size)
    pixels_a = canvas_a.load()
    pixels_b = canvas_b.load()
    intersection = union = identical_on = 0
    area_a = area_b = 0
    total = size * size
    for y in range(size):
        for x in range(size):
            on_a = pixels_a[x, y] > 127
            on_b = pixels_b[x, y] > 127
            if on_a:
                area_a += 1
            if on_b:
                area_b += 1
            if on_a and on_b:
                intersection += 1
                identical_on += 1
            if on_a or on_b:
                union += 1
    precision = intersection / area_a if area_a else 0
    recall = intersection / area_b if area_b else 0
    dice = (2 * intersection) / (area_a + area_b) if area_a + area_b else 0
    return {
        "aligned_iou": round(intersection / union, 4) if union else 0,
        "aligned_dice": round(dice, 4),
        "aligned_precision": round(precision, 4),
        "aligned_recall": round(recall, 4),
        "aligned_identical_on_ratio": round(identical_on / max(1, total), 4),
    }


def normalize_mask_to_bbox(mask: Image.Image, size: int = 160, margin: int = 8) -> Image.Image:
    source = mask.convert("L").point(lambda value: 255 if value > 127 else 0)
    bbox = source.getbbox()
    canvas = Image.new("L", (size, size), 0)
    if not bbox:
        return canvas

    cropped = source.crop(bbox)
    target_size = max(1, size - margin * 2)
    normalized = cropped.resize((target_size, target_size), Image.Resampling.BILINEAR)
    normalized = normalized.point(lambda value: 255 if value > 96 else 0)
    canvas.paste(normalized, (margin, margin))
    return canvas


def mask_grid_signature(mask: Image.Image, rows: int = 3, cols: int = 3, size: int = 96) -> list[float]:
    normalized = normalize_mask_to_bbox(mask, size=size, margin=0)
    pixels = normalized.load()
    signature: list[float] = []
    for row in range(rows):
        y0 = row * size // rows
        y1 = (row + 1) * size // rows
        for col in range(cols):
            x0 = col * size // cols
            x1 = (col + 1) * size // cols
            total = max(1, (x1 - x0) * (y1 - y0))
            occupied = 0
            for y in range(y0, y1):
                for x in range(x0, x1):
                    if pixels[x, y] > 127:
                        occupied += 1
            signature.append(round(occupied / total, 4))
    return signature


def mask_projection_signature(mask: Image.Image, bins: int = 16, size: int = 96) -> tuple[list[float], list[float]]:
    normalized = normalize_mask_to_bbox(mask, size=size, margin=0)
    pixels = normalized.load()
    x_counts = [0] * bins
    y_counts = [0] * bins
    for y in range(size):
        y_bin = min(bins - 1, y * bins // size)
        for x in range(size):
            if pixels[x, y] <= 127:
                continue
            x_bin = min(bins - 1, x * bins // size)
            x_counts[x_bin] += 1
            y_counts[y_bin] += 1
    max_x = max(1, max(x_counts))
    max_y = max(1, max(y_counts))
    return (
        [round(value / max_x, 4) for value in x_counts],
        [round(value / max_y, 4) for value in y_counts],
    )


def mask_contour_grid_signature(mask: Image.Image, rows: int = 4, cols: int = 4, size: int = 96) -> list[float]:
    normalized = normalize_mask_to_bbox(mask, size=size, margin=0)
    pixels = normalized.load()
    counts = [0] * (rows * cols)
    total_edges = 0
    for y in range(size):
        for x in range(size):
            if pixels[x, y] <= 127 or not is_edge(pixels, x, y, size, size):
                continue
            row = min(rows - 1, y * rows // size)
            col = min(cols - 1, x * cols // size)
            counts[row * cols + col] += 1
            total_edges += 1
    total_edges = max(1, total_edges)
    return [round(value / total_edges, 4) for value in counts]


def mask_shape_context_signature(mask: Image.Image, radial_bins: int = 4, angular_bins: int = 8, size: int = 96) -> list[float]:
    normalized = normalize_mask_to_bbox(mask, size=size, margin=0)
    pixels = normalized.load()
    points: list[tuple[int, int]] = []
    edge_points: list[tuple[int, int]] = []
    for y in range(size):
        for x in range(size):
            if pixels[x, y] <= 127:
                continue
            points.append((x, y))
            if is_edge(pixels, x, y, size, size):
                edge_points.append((x, y))
    if not points or not edge_points:
        return [0.0] * (radial_bins * angular_bins)

    cx = sum(x for x, _ in points) / len(points)
    cy = sum(y for _, y in points) / len(points)
    max_radius = max(1.0, max(math.hypot(x - cx, y - cy) for x, y in edge_points))
    counts = [0] * (radial_bins * angular_bins)
    for x, y in edge_points:
        dx = x - cx
        dy = y - cy
        radius = math.hypot(dx, dy) / max_radius
        radial_index = min(radial_bins - 1, int(radius * radial_bins))
        angle = (math.atan2(dy, dx) + math.pi) / (math.pi * 2)
        angular_index = min(angular_bins - 1, int(angle * angular_bins))
        counts[radial_index * angular_bins + angular_index] += 1

    total = max(1, sum(counts))
    return [round(value / total, 4) for value in counts]


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))
