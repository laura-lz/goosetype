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
    point_prompts: list[tuple[int, int, int]] | None = None,
    support_mask: Image.Image | None = None,
) -> Image.Image:
    backend = backend.lower()
    if backend == "sam":
        if bbox is None:
            raise ValueError("SAM segmentation requires a box prompt.")
        return build_sam_box_mask(
            image,
            bbox=bbox,
            model_name=sam_model,
            max_side=max_segmentation_side,
            point_prompts=point_prompts,
            support_mask=support_mask,
        )
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
    point_prompts: list[tuple[int, int, int]] | None = None,
    support_mask: Image.Image | None = None,
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
    prompt_points = point_prompts or build_sam_prompt_points(source, prompt_box, support_mask=support_mask)
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
        prompt_points = [
            (int(round(x * scale_x)), int(round(y * scale_y)), label)
            for x, y, label in prompt_points
        ]
        source = resized

    cached = _SAM_MODELS.get(model_name)
    if cached is None:
        local_only = os.environ.get("GOOSETYPE_SAM_ALLOW_DOWNLOAD", "").lower() not in {"1", "true", "yes"}
        try:
            processor = SamProcessor.from_pretrained(model_name, local_files_only=local_only)
            model = SamModel.from_pretrained(model_name, local_files_only=local_only)
        except Exception as error:
            if local_only:
                raise RuntimeError(
                    f"SAM model {model_name!r} is not fully available in the local Hugging Face cache. "
                    "Connect to the network once with GOOSETYPE_SAM_ALLOW_DOWNLOAD=1, or install/copy the "
                    "model files into the Hugging Face cache."
                ) from error
            raise
        model.eval()
        cached = (processor, model)
        _SAM_MODELS[model_name] = cached
    processor, model = cached

    processor_kwargs = {"input_boxes": [[[list(prompt_box)]]]}
    if prompt_points:
        processor_kwargs["input_points"] = [[[[x, y] for x, y, _ in prompt_points]]]
        processor_kwargs["input_labels"] = [[[label for _, _, label in prompt_points]]]

    inputs = processor(source, **processor_kwargs, return_tensors="pt")
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
    best_index = choose_sam_mask_index(masks[0], scores, prompt_box, source.size, support_mask=support_mask)
    mask_tensor = masks[0, best_index]
    mask = Image.fromarray((mask_tensor.numpy().astype("uint8") * 255), mode="L")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)
    return clean_mask(mask)


def build_sam_prompt_points(
    image: Image.Image,
    prompt_box: tuple[int, int, int, int],
    support_mask: Image.Image | None = None,
) -> list[tuple[int, int, int]]:
    x0, y0, x1, y1 = clamp_bbox(prompt_box, image.size)
    points: list[tuple[int, int, int]] = []

    support = None
    if support_mask is not None:
        support = support_mask.convert("L")
        if support.size != image.size:
            support = support.resize(image.size, Image.Resampling.NEAREST)
        support_point = mask_centroid_point(support, bbox=(x0, y0, x1, y1))
        if support_point is not None:
            points.append((support_point[0], support_point[1], 1))

    points.append(((x0 + x1) // 2, (y0 + y1) // 2, 1))

    dark_point = darkest_component_point(image, bbox=(x0, y0, x1, y1), support_mask=support)
    if dark_point is not None and all(point_distance(dark_point, (px, py)) > 8 for px, py, label in points if label == 1):
        points.append((dark_point[0], dark_point[1], 1))

    neg_margin = max(3, int(round(min(x1 - x0, y1 - y0) * 0.08)))
    negative_points = [
        (max(0, x0 - neg_margin), max(0, y0 - neg_margin)),
        (min(image.width - 1, x1 + neg_margin), max(0, y0 - neg_margin)),
        (max(0, x0 - neg_margin), min(image.height - 1, y1 + neg_margin)),
        (min(image.width - 1, x1 + neg_margin), min(image.height - 1, y1 + neg_margin)),
    ]
    support_pixels = support.load() if support is not None else None
    for nx, ny in negative_points:
        if support_pixels is not None and support_pixels[nx, ny] > 127:
            continue
        points.append((nx, ny, 0))

    return dedupe_sam_points(points, max_points=8)


def mask_centroid_point(mask: Image.Image, bbox: tuple[int, int, int, int]) -> tuple[int, int] | None:
    binary = mask.convert("L")
    pixels = binary.load()
    x0, y0, x1, y1 = clamp_bbox(bbox, binary.size)
    total_x = total_y = count = 0
    for y in range(y0, y1):
        for x in range(x0, x1):
            if pixels[x, y] <= 127:
                continue
            total_x += x
            total_y += y
            count += 1
    if count < 12:
        return None
    return int(round(total_x / count)), int(round(total_y / count))


def darkest_component_point(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    support_mask: Image.Image | None = None,
) -> tuple[int, int] | None:
    x0, y0, x1, y1 = clamp_bbox(bbox, image.size)
    crop = image.convert("RGB").crop((x0, y0, x1, y1))
    gray = ImageOps.grayscale(crop)
    values = list(gray.getdata())
    if not values:
        return None
    threshold = min(105, sorted(values)[max(0, int(len(values) * 0.22) - 1)] + 18)
    dark = gray.point(lambda value: 255 if value <= threshold else 0)
    if support_mask is not None:
        support = support_mask.convert("L").crop((x0, y0, x1, y1)).filter(ImageFilter.MaxFilter(9))
        dark = ImageChops.multiply(dark, support.point(lambda value: 255 if value > 24 else 0))
    components = connected_components(dark, min_area=max(8, int((x1 - x0) * (y1 - y0) * 0.002)))
    if not components:
        return None
    component = components[0]
    local_mask = component_mask(dark, component.bbox)
    point = mask_centroid_point(local_mask, component.bbox)
    if point is None:
        return None
    return x0 + point[0], y0 + point[1]


def dedupe_sam_points(points: list[tuple[int, int, int]], max_points: int) -> list[tuple[int, int, int]]:
    deduped: list[tuple[int, int, int]] = []
    for point in points:
        x, y, label = point
        if any(label == other_label and point_distance((x, y), (other_x, other_y)) < 6 for other_x, other_y, other_label in deduped):
            continue
        deduped.append(point)
        if len(deduped) >= max_points:
            break
    return deduped


def point_distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def choose_sam_mask_index(
    masks,
    scores,
    prompt_box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    support_mask: Image.Image | None = None,
) -> int:
    best_index = 0
    best_score = float("-inf")
    for index in range(masks.shape[0]):
        mask = Image.fromarray((masks[index].numpy().astype("uint8") * 255), mode="L")
        score = sam_candidate_score(mask, float(scores[index].item()), prompt_box, image_size, support_mask=support_mask)
        if score > best_score:
            best_score = score
            best_index = index
    return best_index


def sam_candidate_score(
    mask: Image.Image,
    sam_score: float,
    prompt_box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    support_mask: Image.Image | None = None,
) -> float:
    binary = mask.convert("L").point(lambda value: 255 if value > 127 else 0)
    bbox = binary.getbbox()
    if not bbox:
        return -10.0
    area = sum(1 for value in binary.getdata() if value > 127)
    x0, y0, x1, y1 = prompt_box
    box_area = max(1, (x1 - x0) * (y1 - y0))
    bbox_area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    coverage = area / box_area
    fill = area / bbox_area
    edge_contact = mask_edge_contact_ratio(binary)
    bbox_aspect = (bbox[2] - bbox[0]) / max(1, bbox[3] - bbox[1])
    support_iou = binary_mask_iou(binary, support_mask) if support_mask is not None else None

    score = sam_score
    if coverage < 0.05:
        score -= (0.05 - coverage) * 5.0
    if coverage > 0.72:
        score -= (coverage - 0.72) * 2.8
    if fill > 0.82:
        score -= (fill - 0.82) * 1.8
    if edge_contact > 0.06:
        score -= (edge_contact - 0.06) * 3.0
    if bbox_aspect < 0.16 or bbox_aspect > 6.0:
        score -= 0.35
    if bbox[0] <= 1 or bbox[1] <= 1 or bbox[2] >= image_size[0] - 1 or bbox[3] >= image_size[1] - 1:
        score -= 0.22
    if support_iou is not None:
        score += min(0.18, support_iou * 0.22)
        if support_iou < 0.12:
            score -= 0.18
    return score


def binary_mask_iou(mask: Image.Image, other_mask: Image.Image) -> float:
    a = mask.convert("L").point(lambda value: 255 if value > 127 else 0)
    b = other_mask.convert("L")
    if b.size != a.size:
        b = b.resize(a.size, Image.Resampling.NEAREST)
    b = b.point(lambda value: 255 if value > 127 else 0)
    intersection = union = 0
    for left, right in zip(a.getdata(), b.getdata()):
        left_on = left > 127
        right_on = right > 127
        intersection += int(left_on and right_on)
        union += int(left_on or right_on)
    return intersection / max(1, union)


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
    binary = cleaned.point(lambda value: 255 if value > 96 else 0)
    return refine_instance_mask(binary)


def refine_instance_mask(mask: Image.Image, min_island_area: int = 18) -> Image.Image:
    binary = mask.convert("L").point(lambda value: 255 if value > 127 else 0)
    binary = fill_internal_holes(binary)
    binary = remove_tiny_islands(binary, min_area=min_island_area)
    return binary.point(lambda value: 255 if value > 127 else 0)


def color_refine_instance_mask(
    image: Image.Image,
    mask: Image.Image,
    min_island_area: int = 18,
) -> Image.Image:
    source = image.convert("RGB")
    if source.size != mask.size:
        source = source.resize(mask.size, Image.Resampling.LANCZOS)
    rough = refine_instance_mask(mask, min_island_area=min_island_area)
    foreground = rough.filter(ImageFilter.MinFilter(5))
    if not foreground.getbbox():
        foreground = rough
    background = rough.filter(ImageFilter.MaxFilter(9))

    foreground_mean = masked_rgb_mean(source, foreground, foreground=True)
    background_mean = masked_rgb_mean(source, background, foreground=False)
    if foreground_mean is None or background_mean is None:
        return rough

    width, height = rough.size
    rough_pixels = rough.load()
    bg_pixels = background.load()
    rgb_pixels = source.load()
    refined = Image.new("L", rough.size, 0)
    out = refined.load()

    for y in range(height):
        for x in range(width):
            mask_value = rough_pixels[x, y]
            if mask_value <= 96:
                continue
            rgb = rgb_pixels[x, y]
            fg_distance = rgb_distance(rgb, foreground_mean)
            bg_distance = rgb_distance(rgb, background_mean)
            color_margin = bg_distance - fg_distance
            strong_mask = mask_value > 220 and bg_pixels[x, y] > 0
            if color_margin > -18 or (strong_mask and color_margin > -42):
                out[x, y] = 255

    refined = refined.filter(ImageFilter.MedianFilter(3))
    return refine_instance_mask(refined, min_island_area=min_island_area)


def masked_rgb_mean(image: Image.Image, mask: Image.Image, foreground: bool) -> tuple[float, float, float] | None:
    rgb_pixels = image.load()
    mask_pixels = mask.convert("L").load()
    width, height = image.size
    totals = [0.0, 0.0, 0.0]
    count = 0
    edge_stride = max(1, min(width, height) // 160)

    for y in range(0, height, edge_stride):
        for x in range(0, width, edge_stride):
            is_foreground = mask_pixels[x, y] > 127
            if foreground != is_foreground:
                continue
            if not foreground and not is_edge_or_background_sample(x, y, width, height, mask_pixels):
                continue
            pixel = rgb_pixels[x, y]
            totals[0] += pixel[0]
            totals[1] += pixel[1]
            totals[2] += pixel[2]
            count += 1

    if count < 12:
        return None
    return (totals[0] / count, totals[1] / count, totals[2] / count)


def is_edge_or_background_sample(x: int, y: int, width: int, height: int, mask_pixels) -> bool:
    edge = max(3, min(width, height) // 18)
    return x < edge or y < edge or x >= width - edge or y >= height - edge or mask_pixels[x, y] <= 20


def rgb_distance(pixel: tuple[int, int, int], mean: tuple[float, float, float]) -> float:
    red = float(pixel[0]) - mean[0]
    green = float(pixel[1]) - mean[1]
    blue = float(pixel[2]) - mean[2]
    return math.sqrt(red * red + green * green + blue * blue)


def fill_internal_holes(
    mask: Image.Image,
    max_hole_foreground_ratio: float = 0.12,
    max_hole_bbox_ratio: float = 0.08,
    max_absolute_hole_area: int = 2500,
) -> Image.Image:
    binary = mask.convert("1")
    width, height = binary.size
    pixels = binary.load()
    outside: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque()

    for x in range(width):
        for y in (0, height - 1):
            if not pixels[x, y] and (x, y) not in outside:
                outside.add((x, y))
                queue.append((x, y))
    for y in range(height):
        for x in (0, width - 1):
            if not pixels[x, y] and (x, y) not in outside:
                outside.add((x, y))
                queue.append((x, y))

    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if nx < 0 or ny < 0 or nx >= width or ny >= height or (nx, ny) in outside:
                continue
            if not pixels[nx, ny]:
                outside.add((nx, ny))
                queue.append((nx, ny))

    bbox = mask.convert("L").point(lambda value: 255 if value > 127 else 0).getbbox()
    bbox_area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) if bbox else width * height
    foreground_area = sum(1 for value in mask.convert("L").getdata() if value > 127)
    ratio_cap = min(int(foreground_area * max_hole_foreground_ratio), int(bbox_area * max_hole_bbox_ratio))
    max_hole_area = max(48, min(max_absolute_hole_area, ratio_cap))

    fill_pixels: set[tuple[int, int]] = set()
    seen = set(outside)
    for y in range(height):
        for x in range(width):
            if pixels[x, y] or (x, y) in seen:
                continue
            hole = flood_background_hole(pixels, x, y, width, height, seen)
            if len(hole) <= max_hole_area:
                fill_pixels.update(hole)

    filled = Image.new("L", (width, height), 0)
    out = filled.load()
    for y in range(height):
        for x in range(width):
            if pixels[x, y] or (x, y) in fill_pixels:
                out[x, y] = 255
    return filled


def flood_background_hole(
    pixels,
    start_x: int,
    start_y: int,
    width: int,
    height: int,
    seen: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    hole = {(start_x, start_y)}
    queue: deque[tuple[int, int]] = deque([(start_x, start_y)])
    seen.add((start_x, start_y))
    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if nx < 0 or ny < 0 or nx >= width or ny >= height or (nx, ny) in seen:
                continue
            if not pixels[nx, ny]:
                seen.add((nx, ny))
                hole.add((nx, ny))
                queue.append((nx, ny))
    return hole


def remove_tiny_islands(mask: Image.Image, min_area: int = 18) -> Image.Image:
    try:
        return remove_tiny_islands_cv2(mask, min_area=min_area)
    except Exception:
        return remove_tiny_islands_python(mask, min_area=min_area)


def remove_tiny_islands_cv2(mask: Image.Image, min_area: int = 18) -> Image.Image:
    import cv2
    import numpy as np

    array = np.array(mask.convert("L"))
    binary = np.where(array > 127, 255, 0).astype("uint8")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return Image.fromarray(binary, mode="L")

    largest_index = max(range(1, count), key=lambda index: int(stats[index, cv2.CC_STAT_AREA]))
    refined = np.zeros_like(binary)
    refined[labels == largest_index] = 255
    return Image.fromarray(refined, mode="L")


def remove_tiny_islands_python(mask: Image.Image, min_area: int = 18) -> Image.Image:
    binary = mask.convert("1")
    width, height = binary.size
    pixels = binary.load()
    seen: set[tuple[int, int]] = set()
    components: list[set[tuple[int, int]]] = []

    for y in range(height):
        for x in range(width):
            if (x, y) in seen or not pixels[x, y]:
                continue
            component = flood_component_pixels(pixels, x, y, width, height, seen)
            components.append(component)

    if not components:
        return Image.new("L", mask.size, 0)

    largest = max(components, key=len)
    refined = Image.new("L", mask.size, 0)
    out = refined.load()
    for x, y in largest:
        out[x, y] = 255
    return refined


def flood_component_pixels(
    pixels,
    start_x: int,
    start_y: int,
    width: int,
    height: int,
    seen: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    component = {(start_x, start_y)}
    queue: deque[tuple[int, int]] = deque([(start_x, start_y)])
    seen.add((start_x, start_y))
    while queue:
        x, y = queue.popleft()
        for nx, ny in (
            (x - 1, y),
            (x + 1, y),
            (x, y - 1),
            (x, y + 1),
            (x - 1, y - 1),
            (x + 1, y - 1),
            (x - 1, y + 1),
            (x + 1, y + 1),
        ):
            if nx < 0 or ny < 0 or nx >= width or ny >= height or (nx, ny) in seen:
                continue
            if pixels[nx, ny]:
                seen.add((nx, ny))
                component.add((nx, ny))
                queue.append((nx, ny))
    return component


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
    grid_5x7 = mask_grid_signature(mask, rows=7, cols=5, size=140, preserve_aspect=True)
    projection_x, projection_y = mask_projection_signature(mask)
    contour_grid_4x4 = mask_contour_grid_signature(mask)
    shape_context = mask_shape_context_signature(mask)
    stroke_geometry = mask_stroke_geometry_signature(mask)
    diagonal_geometry = mask_diagonal_geometry_signature(mask, grid_3x3=grid_3x3)
    quality = mask_quality_metrics(mask, points=points, bbox=bbox, area=area, perimeter=perimeter)

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
        "bbox_width_ratio": round(bbox_width / max(width, 1), 4),
        "bbox_height_ratio": round(bbox_height / max(height, 1), 4),
        "fill_ratio": round(area / bbox_area, 4),
        "grid_3x3": grid_3x3,
        "grid_3x3_binary": [1 if value >= 0.16 else 0 for value in grid_3x3],
        "grid_5x7": grid_5x7,
        "grid_5x7_binary": [1 if value >= 0.12 else 0 for value in grid_5x7],
        "projection_x": projection_x,
        "projection_y": projection_y,
        "contour_grid_4x4": contour_grid_4x4,
        "shape_context": shape_context,
        **quality,
        **stroke_geometry,
        **diagonal_geometry,
        "bbox": [bbox[0], bbox[1], bbox_width, bbox_height],
    }


def mask_quality_metrics(
    mask: Image.Image,
    points: list[tuple[int, int]] | None = None,
    bbox: tuple[int, int, int, int] | None = None,
    area: int | None = None,
    perimeter: int | None = None,
) -> dict:
    binary = mask.convert("L").point(lambda value: 255 if value > 127 else 0)
    width, height = binary.size
    if max(width, height) > 220:
        scale = 220 / max(width, height)
        small_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        small = binary.resize(small_size, Image.Resampling.NEAREST)
        return mask_quality_metrics(small)

    bbox = bbox or binary.getbbox()
    if not bbox:
        return {
            "mask_quality_score": 0.0,
            "mask_quality_reasons": ["empty_mask"],
            "mask_component_count": 0,
            "mask_largest_component_ratio": 0.0,
            "mask_crop_edge_contact_ratio": 1.0,
            "mask_hole_ratio": 0.0,
            "mask_padding_balance": 0.0,
            "mask_patchiness_score": 1.0,
        }

    if points is None:
        pixels = binary.load()
        points = [(x, y) for y in range(height) for x in range(width) if pixels[x, y] > 127]
    area = int(area if area is not None else len(points))
    perimeter = int(perimeter if perimeter is not None else mask_perimeter(binary))
    components = connected_components(binary, min_area=1)
    large_components = [component for component in components if component.area >= max(16, area * 0.015)]
    largest_area = max((component.area for component in components), default=0)
    largest_ratio = largest_area / max(1, area)
    edge_contact = mask_edge_contact_ratio(binary)
    hole_ratio = internal_hole_ratio(binary, area)
    padding_balance = crop_padding_balance(bbox, (width, height))
    patchiness = mask_patchiness_score(binary, area=area, perimeter=perimeter, bbox=bbox)
    edge_density = perimeter / max(1, area)
    fill_ratio = area / max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

    score = 1.0
    reasons: list[str] = []
    penalties = [
        (largest_ratio >= 0.72, 0.28, "fragmented_components"),
        (len(large_components) <= 3, 0.18, "too_many_large_components"),
        (edge_contact <= 0.045, 0.22, "foreground_touches_crop_edge"),
        (hole_ratio <= 0.2, 0.18, "large_internal_holes"),
        (padding_balance <= 0.35, 0.12, "uneven_crop_padding"),
        (patchiness >= 0.58, 0.28, "giant_or_noisy_patch"),
        (edge_density <= 0.18 or fill_ratio <= 0.38, 0.2, "jagged_or_shredded_mask"),
        (fill_ratio <= 0.74, 0.18, "overfilled_blob_mask"),
    ]
    for passed, penalty, reason in penalties:
        if not passed:
            score -= penalty
            reasons.append(reason)

    return {
        "mask_quality_score": round(clamp(score), 4),
        "mask_quality_reasons": reasons,
        "mask_component_count": len(large_components),
        "mask_largest_component_ratio": round(largest_ratio, 4),
        "mask_crop_edge_contact_ratio": round(edge_contact, 4),
        "mask_hole_ratio": round(hole_ratio, 4),
        "mask_padding_balance": round(padding_balance, 4),
        "mask_patchiness_score": round(patchiness, 4),
        "mask_edge_density": round(edge_density, 4),
    }


def mask_perimeter(mask: Image.Image) -> int:
    binary = mask.convert("L")
    pixels = binary.load()
    width, height = binary.size
    return sum(
        1
        for y in range(height)
        for x in range(width)
        if pixels[x, y] > 127 and is_edge(pixels, x, y, width, height)
    )


def mask_edge_contact_ratio(mask: Image.Image) -> float:
    binary = mask.convert("L")
    pixels = binary.load()
    width, height = binary.size
    edge_pixels = foreground_edge_pixels = 0
    for x in range(width):
        for y in (0, height - 1):
            edge_pixels += 1
            foreground_edge_pixels += int(pixels[x, y] > 127)
    for y in range(1, height - 1):
        for x in (0, width - 1):
            edge_pixels += 1
            foreground_edge_pixels += int(pixels[x, y] > 127)
    return foreground_edge_pixels / max(1, edge_pixels)


def internal_hole_ratio(mask: Image.Image, foreground_area: int) -> float:
    try:
        return internal_hole_ratio_cv2(mask, foreground_area)
    except Exception:
        pass
    before = mask.convert("L").point(lambda value: 255 if value > 127 else 0)
    filled = fill_internal_holes(
        before,
        max_hole_foreground_ratio=1.0,
        max_hole_bbox_ratio=1.0,
        max_absolute_hole_area=before.width * before.height,
    )
    added = sum(1 for left, right in zip(before.getdata(), filled.getdata()) if left <= 127 and right > 127)
    return added / max(1, foreground_area)


def internal_hole_ratio_cv2(mask: Image.Image, foreground_area: int) -> float:
    import cv2
    import numpy as np

    array = np.array(mask.convert("L"))
    background = np.where(array > 127, 0, 255).astype("uint8")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(background, connectivity=4)
    height, width = background.shape
    hole_area = 0
    for index in range(1, count):
        x, y, component_width, component_height, area = stats[index]
        touches_border = x == 0 or y == 0 or x + component_width >= width or y + component_height >= height
        if not touches_border:
            hole_area += int(area)
    return hole_area / max(1, foreground_area)


def crop_padding_balance(bbox: tuple[int, int, int, int], size: tuple[int, int]) -> float:
    width, height = size
    x0, y0, x1, y1 = bbox
    left = x0 / max(1, width)
    right = (width - x1) / max(1, width)
    top = y0 / max(1, height)
    bottom = (height - y1) / max(1, height)
    return max(abs(left - right), abs(top - bottom))


def mask_patchiness_score(mask: Image.Image, area: int, perimeter: int, bbox: tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = bbox
    bbox_area = max(1, (x1 - x0) * (y1 - y0))
    fill = area / bbox_area
    edge_density = perimeter / max(1, area)
    score = 1.0
    if fill > 0.82:
        score -= min(0.55, (fill - 0.82) * 2.2)
    if fill < 0.045:
        score -= min(0.55, (0.045 - fill) * 7.0)
    if edge_density > 0.48:
        score -= min(0.5, (edge_density - 0.48) * 1.3)
    return clamp(score)


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
    mask_quality = float(features.get("mask_quality_score", 1))
    border_contact = border_contact_ratio((x0, y0, x1, y1), image_size)

    score = 1.0
    reasons: list[str] = []

    checks = [
        (area >= image_area * 0.0015, 0.45, "too_small"),
        (bbox_area_ratio <= 0.82, 0.45, "too_large_for_single_goose"),
        (0.22 <= aspect_ratio <= 4.8, 0.55, "implausible_aspect_ratio"),
        (0.07 <= fill_ratio <= 0.76, 0.6, "implausible_fill_ratio"),
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
    if mask_quality < 0.45:
        score -= 0.35
        reasons.append("low_mask_quality")
    elif mask_quality < 0.65:
        score -= 0.15
        reasons.append("borderline_mask_quality")
    if fill_ratio > 0.76 and "overfilled_blob_mask" in features.get("mask_quality_reasons", []):
        score -= 0.2
        reasons.append("overfilled_mask")

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
        "bbox_width_ratio": 0,
        "bbox_height_ratio": 0,
        "fill_ratio": 0,
        "grid_3x3": [0.0] * 9,
        "grid_3x3_binary": [0] * 9,
        "grid_5x7": [0.0] * 35,
        "grid_5x7_binary": [0] * 35,
        "projection_x": [0.0] * 16,
        "projection_y": [0.0] * 16,
        "contour_grid_4x4": [0.0] * 16,
        "shape_context": [0.0] * 32,
        "horizontal_bands": [0.0] * 12,
        "vertical_bands": [0.0] * 12,
        "horizontal_run_signature": [0.0] * 7,
        "vertical_run_signature": [0.0] * 5,
        "stroke_axis_strength": [0.0, 0.0],
        "diagonal_grid_signature": [0.0] * 6,
        "diagonal_grid_binary": [0] * 6,
        "diagonal_band_signature": [0.0] * 6,
        "diagonal_axis_strength": [0.0, 0.0],
        "corner_balance": [0.0] * 4,
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
    canvas_a = normalize_mask_to_bbox(a, size=size, preserve_aspect=True)
    canvas_b = normalize_mask_to_bbox(b, size=size, preserve_aspect=True)
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


def normalize_mask_to_bbox(
    mask: Image.Image,
    size: int = 160,
    margin: int = 8,
    preserve_aspect: bool = False,
) -> Image.Image:
    source = mask.convert("L").point(lambda value: 255 if value > 127 else 0)
    bbox = source.getbbox()
    canvas = Image.new("L", (size, size), 0)
    if not bbox:
        return canvas

    cropped = source.crop(bbox)
    target_size = max(1, size - margin * 2)
    if preserve_aspect:
        normalized = ImageOps.contain(cropped, (target_size, target_size), Image.Resampling.BILINEAR)
    else:
        normalized = cropped.resize((target_size, target_size), Image.Resampling.BILINEAR)
    normalized = normalized.point(lambda value: 255 if value > 96 else 0)
    canvas.paste(normalized, ((size - normalized.width) // 2, (size - normalized.height) // 2))
    return canvas


def mask_grid_signature(
    mask: Image.Image,
    rows: int = 3,
    cols: int = 3,
    size: int = 96,
    preserve_aspect: bool = False,
) -> list[float]:
    normalized = normalize_mask_to_bbox(mask, size=size, margin=0, preserve_aspect=preserve_aspect)
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


def mask_stroke_geometry_signature(mask: Image.Image, size: int = 112) -> dict:
    normalized = normalize_mask_to_bbox(mask, size=size, margin=0, preserve_aspect=True)
    pixels = normalized.load()
    row_counts = [0] * size
    col_counts = [0] * size
    horizontal_runs = [0] * size
    vertical_runs = [0] * size

    for y in range(size):
        current = longest = 0
        for x in range(size):
            if pixels[x, y] > 127:
                row_counts[y] += 1
                col_counts[x] += 1
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        horizontal_runs[y] = longest

    for x in range(size):
        current = longest = 0
        for y in range(size):
            if pixels[x, y] > 127:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        vertical_runs[x] = longest

    max_row = max(1, max(row_counts))
    max_col = max(1, max(col_counts))
    return {
        "horizontal_bands": projection_bands(row_counts, max_value=max_row, max_bands=4),
        "vertical_bands": projection_bands(col_counts, max_value=max_col, max_bands=4),
        "horizontal_run_signature": binned_run_signature(horizontal_runs, rows_or_cols=7, normalize_by=size),
        "vertical_run_signature": binned_run_signature(vertical_runs, rows_or_cols=5, normalize_by=size),
        "stroke_axis_strength": [
            round(sum(horizontal_runs) / max(1, size * size), 4),
            round(sum(vertical_runs) / max(1, size * size), 4),
        ],
    }


def mask_diagonal_geometry_signature(mask: Image.Image, grid_3x3: list[float] | None = None, size: int = 112) -> dict:
    normalized = normalize_mask_to_bbox(mask, size=size, margin=0, preserve_aspect=True)
    pixels = normalized.load()
    grid = grid_3x3 or mask_grid_signature(mask)
    main_grid = [grid[0], grid[4], grid[8]] if len(grid) >= 9 else [0.0, 0.0, 0.0]
    anti_grid = [grid[2], grid[4], grid[6]] if len(grid) >= 9 else [0.0, 0.0, 0.0]

    band_width = max(2, round(size * 0.085))
    main_counts = [0, 0, 0]
    main_totals = [0, 0, 0]
    anti_counts = [0, 0, 0]
    anti_totals = [0, 0, 0]
    main_run = anti_run = 0

    for diagonal_index in range(size):
        main_has_pixel = False
        anti_has_pixel = False
        for offset in range(-band_width, band_width + 1):
            x = diagonal_index + offset
            y = diagonal_index
            if 0 <= x < size:
                segment = min(2, y * 3 // size)
                main_totals[segment] += 1
                if pixels[x, y] > 127:
                    main_counts[segment] += 1
                    main_has_pixel = True

            anti_x = size - 1 - diagonal_index + offset
            anti_y = diagonal_index
            if 0 <= anti_x < size:
                segment = min(2, anti_y * 3 // size)
                anti_totals[segment] += 1
                if pixels[anti_x, anti_y] > 127:
                    anti_counts[segment] += 1
                    anti_has_pixel = True

        if main_has_pixel:
            main_run += 1
        if anti_has_pixel:
            anti_run += 1

    main_band = [round(main_counts[index] / max(1, main_totals[index]), 4) for index in range(3)]
    anti_band = [round(anti_counts[index] / max(1, anti_totals[index]), 4) for index in range(3)]
    corners = [grid[index] if len(grid) > index else 0.0 for index in (0, 2, 6, 8)]
    return {
        "diagonal_grid_signature": [round(value, 4) for value in main_grid + anti_grid],
        "diagonal_grid_binary": [1 if value >= 0.16 else 0 for value in main_grid + anti_grid],
        "diagonal_band_signature": main_band + anti_band,
        "diagonal_axis_strength": [round(main_run / size, 4), round(anti_run / size, 4)],
        "corner_balance": [round(value, 4) for value in corners],
    }


def projection_bands(
    values: list[int],
    max_value: int,
    max_bands: int = 4,
    threshold_ratio: float = 0.38,
) -> list[float]:
    if not values or max_value <= 0:
        return [0.0] * (max_bands * 3)

    threshold = max_value * threshold_ratio
    bands: list[tuple[float, float, float]] = []
    start: int | None = None
    total = 0
    peak = 0
    for index, value in enumerate(values + [0]):
        if value >= threshold:
            if start is None:
                start = index
                total = 0
                peak = 0
            total += value
            peak = max(peak, value)
            continue
        if start is None:
            continue
        end = index
        thickness = end - start
        center = (start + end - 1) / 2
        strength = total / max(1, thickness * max_value)
        bands.append((center / max(1, len(values) - 1), thickness / len(values), strength))
        start = None

    bands.sort(key=lambda item: item[2], reverse=True)
    bands = sorted(bands[:max_bands], key=lambda item: item[0])
    signature: list[float] = []
    for center, thickness, strength in bands:
        signature.extend([round(center, 4), round(thickness, 4), round(strength, 4)])
    signature.extend([0.0] * (max_bands * 3 - len(signature)))
    return signature


def binned_run_signature(runs: list[int], rows_or_cols: int, normalize_by: int) -> list[float]:
    signature: list[float] = []
    length = len(runs)
    for index in range(rows_or_cols):
        start = index * length // rows_or_cols
        end = (index + 1) * length // rows_or_cols
        bucket = runs[start:end]
        signature.append(round((max(bucket) if bucket else 0) / max(1, normalize_by), 4))
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
