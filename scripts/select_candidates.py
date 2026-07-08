#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from PIL import Image, ImageOps
except ImportError as error:
    raise SystemExit("Pillow is required: python3 -m pip install Pillow") from error

from goosetype.classifier import load_glyph_classifier
from goosetype.image_features import aligned_mask_similarity, mask_iou, normalize_mask_to_bbox


FEATURE_KEYS = ("aspect_ratio", "curvature_score", "boldness_score", "slant_score", "fill_ratio")
STROKE_LETTERS = set("AEFHIKLNTVXYZfkltvxyz")
CURVED_LETTERS = set("BCDGJOPQRSUabcdegopqrsu")
SIMPLE_STEM_LETTERS = set("Iijlr")
WIDE_HUMP_LETTERS = set("MWhmnw")

STRUCTURE_WEIGHTS = {
    "stroke": {
        "grid_3x3": 0.18,
        "grid_3x3_binary": 0.1,
        "diagonal_grid_signature": 0.08,
        "diagonal_band_signature": 0.1,
        "diagonal_axis_strength": 0.07,
        "corner_balance": 0.04,
        "horizontal_bands": 0.11,
        "vertical_bands": 0.11,
        "horizontal_run_signature": 0.06,
        "vertical_run_signature": 0.06,
        "projection_x": 0.04,
        "projection_y": 0.04,
        "contour_grid_4x4": 0.01,
    },
    "curved": {
        "grid_3x3": 0.12,
        "grid_3x3_binary": 0.06,
        "diagonal_grid_signature": 0.03,
        "diagonal_band_signature": 0.03,
        "diagonal_axis_strength": 0.02,
        "corner_balance": 0.03,
        "horizontal_bands": 0.12,
        "vertical_bands": 0.12,
        "horizontal_run_signature": 0.06,
        "vertical_run_signature": 0.06,
        "projection_x": 0.11,
        "projection_y": 0.11,
        "contour_grid_4x4": 0.16,
    },
    "simple_stem": {
        "grid_3x3": 0.12,
        "grid_3x3_binary": 0.08,
        "diagonal_grid_signature": 0.02,
        "diagonal_band_signature": 0.02,
        "diagonal_axis_strength": 0.01,
        "corner_balance": 0.03,
        "horizontal_bands": 0.08,
        "vertical_bands": 0.2,
        "horizontal_run_signature": 0.05,
        "vertical_run_signature": 0.22,
        "projection_x": 0.04,
        "projection_y": 0.08,
        "contour_grid_4x4": 0.05,
    },
    "wide_hump": {
        "grid_3x3": 0.15,
        "grid_3x3_binary": 0.08,
        "diagonal_grid_signature": 0.04,
        "diagonal_band_signature": 0.04,
        "diagonal_axis_strength": 0.03,
        "corner_balance": 0.04,
        "horizontal_bands": 0.09,
        "vertical_bands": 0.19,
        "horizontal_run_signature": 0.04,
        "vertical_run_signature": 0.13,
        "projection_x": 0.08,
        "projection_y": 0.06,
        "contour_grid_4x4": 0.03,
    },
}

LETTER_RECIPES = {
    "A": {
        "require": {"diag_up": 0.75, "diag_down": 0.75, "middle_h": 0.55, "bottom_void": 0.35},
        "forbid": {"top_h": 0.45, "center_v": 0.35, "round": 0.55},
    },
    "B": {
        "require": {"left_v": 0.85, "upper_round": 0.75, "lower_round": 0.75, "center_void": 0.35},
        "forbid": {"diag_up": 0.45, "diag_down": 0.45},
    },
    "C": {"require": {"round": 0.85, "center_void": 0.55}, "forbid": {"right_v": 0.38, "center_v": 0.45}},
    "D": {"require": {"left_v": 0.85, "round": 0.8, "right_v": 0.45, "center_void": 0.35}, "forbid": {"diag_up": 0.45}},
    "E": {
        "require": {"left_v": 0.9, "top_h": 0.78, "middle_h": 0.65, "bottom_h": 0.78},
        "forbid": {"right_v": 0.4, "round": 0.62, "diag_up": 0.45, "diag_down": 0.45},
    },
    "F": {
        "require": {"left_v": 0.9, "top_h": 0.78, "middle_h": 0.62},
        "forbid": {"bottom_h": 0.42, "right_v": 0.4, "round": 0.62},
    },
    "G": {"require": {"round": 0.8, "middle_h": 0.45, "center_void": 0.45}, "forbid": {"right_v": 0.68}},
    "H": {
        "require": {"left_v": 0.85, "right_v": 0.85, "middle_h": 0.72},
        "forbid": {"top_h": 0.72, "bottom_h": 0.72, "round": 0.58},
    },
    "I": {
        "require": {"center_v": 0.9},
        "forbid": {"left_v": 0.42, "right_v": 0.42, "round": 0.5, "diag_up": 0.38, "diag_down": 0.38},
    },
    "J": {"require": {"right_v": 0.8, "bottom_h": 0.45, "lower_round": 0.55}, "forbid": {"left_v": 0.45, "top_h": 0.6}},
    "K": {"require": {"left_v": 0.82, "diag_up": 0.62, "diag_down": 0.62}, "forbid": {"right_v": 0.55, "round": 0.55}},
    "L": {"require": {"left_v": 0.88, "bottom_h": 0.72}, "forbid": {"top_h": 0.45, "right_v": 0.4, "round": 0.55}},
    "M": {
        "require": {"left_v": 0.78, "center_v": 0.62, "right_v": 0.78},
        "forbid": {"round": 0.62, "center_void": 0.72},
    },
    "N": {"require": {"left_v": 0.8, "right_v": 0.8, "diag_down": 0.72}, "forbid": {"round": 0.6, "middle_h": 0.58}},
    "O": {
        "require": {"round": 0.9, "center_void": 0.58},
        "forbid": {"center_v": 0.48, "middle_h": 0.48, "diag_up": 0.42, "diag_down": 0.42},
    },
    "P": {
        "require": {"left_v": 0.85, "upper_round": 0.82, "center_void": 0.35},
        "forbid": {"lower_round": 0.55, "bottom_h": 0.48, "diag_down": 0.45},
    },
    "Q": {"require": {"round": 0.86, "center_void": 0.5}, "forbid": {"center_v": 0.52, "middle_h": 0.52}},
    "R": {
        "require": {"left_v": 0.85, "upper_round": 0.78, "diag_down": 0.55},
        "forbid": {"lower_round": 0.65, "right_v": 0.78},
    },
    "S": {"require": {"upper_round": 0.78, "lower_round": 0.78}, "forbid": {"left_v": 0.78, "right_v": 0.78, "center_v": 0.62}},
    "T": {
        "require": {"top_h": 0.88, "center_v": 0.82},
        "forbid": {"left_v": 0.48, "right_v": 0.48, "bottom_h": 0.48, "round": 0.55},
    },
    "U": {
        "require": {"left_v": 0.72, "right_v": 0.72, "lower_round": 0.62},
        "forbid": {"top_h": 0.5, "center_v": 0.55, "diag_up": 0.45, "diag_down": 0.45},
    },
    "V": {"require": {"diag_up": 0.72, "diag_down": 0.72}, "forbid": {"top_h": 0.45, "bottom_h": 0.6, "center_v": 0.62}},
    "W": {
        "require": {"left_v": 0.72, "center_v": 0.62, "right_v": 0.72},
        "forbid": {"round": 0.62, "top_h": 0.62},
    },
    "X": {
        "require": {"diag_up": 0.86, "diag_down": 0.86},
        "forbid": {"left_v": 0.58, "right_v": 0.58, "top_h": 0.58, "bottom_h": 0.58, "round": 0.55},
    },
    "Y": {
        "require": {"diag_up": 0.68, "diag_down": 0.68, "center_v": 0.72},
        "forbid": {"bottom_h": 0.48, "round": 0.55},
    },
    "Z": {
        "require": {"top_h": 0.82, "bottom_h": 0.82, "diag_up": 0.7},
        "forbid": {"left_v": 0.48, "right_v": 0.48, "round": 0.58},
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select goose candidates for each letter by cheap feature similarity first, "
            "then apply slower mask IoU only to the narrowed set."
        )
    )
    parser.add_argument(
        "--features",
        action="append",
        default=None,
        help="Goose feature file. Repeat to merge local and external candidate libraries.",
    )
    parser.add_argument("--font-targets", default="data/font_targets/reference.json")
    parser.add_argument("--output", default="data/font_candidates/goosetype_candidates.json")
    parser.add_argument("--top-k", type=int, default=8, help="Final candidates to keep per letter.")
    parser.add_argument("--preselect", type=int, default=36, help="Feature-ranked candidates to send to mask comparison.")
    parser.add_argument("--mask-size", type=int, default=180, help="Raster size for mask similarity.")
    parser.add_argument("--classifier-model", default=None, help="Optional EMNIST-style classifier checkpoint.")
    parser.add_argument("--classifier-device", default="cpu")
    parser.add_argument(
        "--min-segmentation-quality",
        type=float,
        default=0.48,
        help="Drop masks below this pre-ranking segmentation quality score.",
    )
    parser.add_argument(
        "--use-letter-recipes",
        action="store_true",
        help="Use the older hand-written per-letter recipe table. Off by default because it can reward mask artifacts.",
    )
    args = parser.parse_args()

    feature_paths = args.features or ["data/masks_v2_from_crops/features.json"]
    geese = [
        goose
        for goose in load_feature_records(feature_paths)
        if segmentation_quality_score(goose) >= args.min_segmentation_quality
    ]
    targets = json.loads(Path(args.font_targets).read_text(encoding="utf-8"))["targets"]
    output = {}
    mask_cache: dict[str, Image.Image] = {}
    classifier = load_glyph_classifier(args.classifier_model, device=args.classifier_device) if args.classifier_model else None

    for target in targets:
        ranked = sorted(
            geese,
            key=lambda goose: preselect_distance(goose, target),
        )[: args.preselect]
        scored = []
        target_mask = load_mask(target["mask_path"], mask_cache)
        for goose in ranked:
            goose_mask = load_mask(goose["mask_path"], mask_cache)
            normal_score = score_mask_variant(
                goose_mask,
                target_mask,
                args.mask_size,
                flip_x=False,
                classifier=classifier,
                target_letter=target["letter"],
            )
            flipped_score = score_mask_variant(
                ImageOps.mirror(goose_mask),
                target_mask,
                args.mask_size,
                flip_x=True,
                classifier=classifier,
                target_letter=target["letter"],
            )
            mask_score = max((normal_score, flipped_score), key=lambda item: item["variant_total"])
            feature_score = 1 - min(global_feature_distance(goose, target), 1)
            family = letter_family(target["letter"])
            dimension_score = dimension_bound_score(goose, target, family)
            geometry_score = 1 - min(geometry_distance(goose, target, family), 1)
            recipe_score = 1 - min(recipe_distance(goose, target, use_letter_recipes=args.use_letter_recipes), 1)
            structure_score = 1 - min(structure_distance(goose, target, family), 1)
            roundness_score = 1 - min(roundness_distance(goose, target), 1)
            quality_score = candidate_quality_score(goose, goose_mask, target)
            total = final_score(
                feature_score,
                geometry_score,
                recipe_score,
                dimension_score,
                mask_score,
                quality_score=quality_score,
                classifier_enabled=classifier is not None,
                family=family,
            )
            scored.append(
                {
                    "goose_id": goose_asset_id(goose),
                    "source_goose_id": goose.get("id"),
                    "letter": target["letter"],
                    "letter_family": family,
                    "score": round(total, 4),
                    "feature_score": round(feature_score, 4),
                    "geometry_score": round(geometry_score, 4),
                    "recipe_score": round(recipe_score, 4),
                    "structure_score": round(structure_score, 4),
                    "roundness_score": round(roundness_score, 4),
                    "dimension_score": round(dimension_score, 4),
                    "quality_score": round(quality_score, 4),
                    **{key: value for key, value in mask_score.items() if key != "mask_total"},
                    "source_image": goose.get("source_image"),
                    "bbox": goose.get("bbox"),
                    "cutout_path": goose.get("cutout_path"),
                    "silhouette_path": goose.get("silhouette_path"),
                    "mask_path": goose.get("mask_path"),
                }
            )
        output[target["letter"]] = sorted(scored, key=lambda item: item["score"], reverse=True)[: args.top_k]
        print(f"{target['letter']}: kept {len(output[target['letter']])} candidates from {len(geese)} geese")

    repo_root = Path(__file__).resolve().parents[1]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(compact_candidate_output(output, repo_root=repo_root), indent=2), encoding="utf-8")
    print(f"Wrote candidate shortlist to {args.output}")


def compact_candidate_output(output: dict[str, list[dict]], repo_root: Path | None = None) -> dict:
    geese: dict[str, dict] = {}
    letters: dict[str, list[dict]] = {}
    shared_keys = ("source_image", "bbox", "cutout_path", "silhouette_path", "mask_path")
    path_keys = {"source_image", "cutout_path", "silhouette_path", "mask_path"}

    for letter, candidates in output.items():
        letters[letter] = []
        for candidate in candidates:
            goose_id = candidate["goose_id"]
            if goose_id not in geese:
                geese[goose_id] = {}
                for key in shared_keys:
                    value = candidate.get(key)
                    if value is None:
                        continue
                    geese[goose_id][key] = web_asset_path(value, repo_root=repo_root) if key in path_keys else value
            letters[letter].append(
                {
                    key: value
                    for key, value in candidate.items()
                    if key not in shared_keys and key != "letter"
                }
            )

    return {
        "schema": "goosetype-candidates-v2",
        "geese": geese,
        "letters": letters,
    }


def web_asset_path(path: str | None, repo_root: Path | None = None) -> str | None:
    if not path:
        return None
    asset = Path(path)
    if asset.is_absolute() and repo_root is not None:
        try:
            return str(asset.relative_to(repo_root))
        except ValueError:
            return str(asset)
    return str(asset)


def goose_asset_id(goose: dict) -> str:
    digest = hashlib.sha1(goose_identity_key(goose).encode("utf-8")).hexdigest()[:10]
    return f"{goose.get('id', 'goose')}_{digest}"


def goose_identity_key(goose: dict) -> str:
    source = Path(str(goose.get("source_image") or "")).name
    bbox = tuple(int(round(float(value))) for value in (goose.get("bbox") or []))
    if source and bbox:
        return f"{source}|{bbox}"
    mask_path = str(goose.get("mask_path") or "")
    return f"{mask_path}|{source}"


def score_mask_variant(
    goose_mask: Image.Image,
    target_mask: Image.Image,
    size: int,
    flip_x: bool,
    classifier=None,
    target_letter: str | None = None,
) -> dict:
    centered_score = mask_iou(goose_mask, target_mask, size=size)
    aligned_score = aligned_mask_similarity(goose_mask, target_mask, size=size)
    negative_score = negative_mask_similarity(goose_mask, target_mask, size=size)
    border_score = positive_border_similarity(goose_mask, target_mask, size=size)
    mask_total = (
        border_score["positive_border_score"] * 0.34
        + negative_score["negative_space_score"] * 0.3
        + aligned_score["aligned_precision"] * 0.14
        + aligned_score["aligned_iou"] * 0.12
        + aligned_score["aligned_dice"] * 0.06
        + centered_score["iou"] * 0.04
    )
    classifier_score = classifier.score_mask(goose_mask, target_letter) if classifier and target_letter else None
    variant_total = mask_total if classifier_score is None else mask_total * 0.55 + classifier_score * 0.45
    return {
        "mask_total": mask_total,
        "variant_total": variant_total,
        "flip_x": flip_x,
        "classifier_score": classifier_score,
        "centered_iou": centered_score["iou"],
        "centered_identical_pixel_ratio": centered_score["identical_pixel_ratio"],
        **border_score,
        **negative_score,
        **aligned_score,
    }


def negative_mask_similarity(a: Image.Image, b: Image.Image, size: int = 96, rows: int = 5, cols: int = 5) -> dict:
    candidate = normalize_for_negative_mask(a, size=size)
    target = normalize_for_negative_mask(b, size=size)
    candidate_pixels = candidate.load()
    target_pixels = target.load()
    empty_penalty = 0.0
    empty_weight = 0.0
    filled_reward = 0.0
    filled_weight = 0.0

    for row in range(rows):
        y0 = row * size // rows
        y1 = (row + 1) * size // rows
        for col in range(cols):
            x0 = col * size // cols
            x1 = (col + 1) * size // cols
            area = max(1, (x1 - x0) * (y1 - y0))
            target_fill = 0
            candidate_fill = 0
            for y in range(y0, y1):
                for x in range(x0, x1):
                    if target_pixels[x, y] > 127:
                        target_fill += 1
                    if candidate_pixels[x, y] > 127:
                        candidate_fill += 1
            target_ratio = target_fill / area
            candidate_ratio = candidate_fill / area
            if target_ratio <= 0.08:
                empty_weight += 1
                empty_penalty += min(1.0, candidate_ratio / 0.22)
            elif target_ratio >= 0.28:
                filled_weight += 1
                filled_reward += max(0.0, 1 - abs(candidate_ratio - target_ratio))

    empty_score = 1 - empty_penalty / max(1.0, empty_weight)
    filled_score = filled_reward / max(1.0, filled_weight)
    score = empty_score * 0.68 + filled_score * 0.32
    return {
        "negative_space_score": round(max(0.0, min(1.0, score)), 4),
        "negative_empty_score": round(max(0.0, min(1.0, empty_score)), 4),
        "negative_filled_score": round(max(0.0, min(1.0, filled_score)), 4),
    }


def normalize_for_negative_mask(mask: Image.Image, size: int) -> Image.Image:
    return normalize_mask_to_bbox(mask, size=size, margin=6, preserve_aspect=True)


def positive_border_similarity(a: Image.Image, b: Image.Image, size: int = 96, radius: int = 2) -> dict:
    candidate = normalize_for_negative_mask(a, size=size)
    target = normalize_for_negative_mask(b, size=size)
    candidate_pixels = candidate.load()
    target_pixels = target.load()
    border_points = [
        (x, y)
        for y in range(size)
        for x in range(size)
        if target_pixels[x, y] > 127 and is_edge_pixel(target_pixels, x, y, size, size)
    ]
    if not border_points:
        return {
            "positive_border_score": 0.0,
            "positive_border_recall": 0.0,
            "positive_border_weakest_band": 0.0,
        }

    covered = sum(1 for x, y in border_points if has_nearby_pixel(candidate_pixels, x, y, size, radius))
    recall = covered / len(border_points)
    band_scores = []
    band_specs = (
        lambda x, y: y < size * 0.28,
        lambda x, y: y >= size * 0.72,
        lambda x, y: x < size * 0.28,
        lambda x, y: x >= size * 0.72,
    )
    for contains in band_specs:
        band = [(x, y) for x, y in border_points if contains(x, y)]
        if len(band) < 8:
            continue
        band_covered = sum(1 for x, y in band if has_nearby_pixel(candidate_pixels, x, y, size, radius))
        band_scores.append(band_covered / len(band))

    weakest_band = min(band_scores) if band_scores else recall
    score = recall * 0.55 + weakest_band * 0.45
    return {
        "positive_border_score": round(max(0.0, min(1.0, score)), 4),
        "positive_border_recall": round(max(0.0, min(1.0, recall)), 4),
        "positive_border_weakest_band": round(max(0.0, min(1.0, weakest_band)), 4),
    }


def has_nearby_pixel(pixels, x: int, y: int, size: int, radius: int) -> bool:
    for yy in range(max(0, y - radius), min(size, y + radius + 1)):
        for xx in range(max(0, x - radius), min(size, x + radius + 1)):
            if pixels[xx, yy] > 127:
                return True
    return False


def final_score(
    feature_score: float,
    geometry_score: float,
    recipe_score: float,
    dimension_score: float,
    mask_score: dict,
    quality_score: float,
    classifier_enabled: bool,
    family: str,
) -> float:
    if classifier_enabled:
        classifier_score = float(mask_score.get("classifier_score") or 0)
        weights = classifier_final_weights(family)
        return (
            classifier_score * weights["classifier"]
            + dimension_score * weights["dimension"]
            + quality_score * weights["quality"]
            + geometry_score * weights["geometry"]
            + recipe_score * weights["recipe"]
            + feature_score * weights["feature"]
            + mask_score["positive_border_score"] * weights["positive_border"]
            + mask_score["negative_space_score"] * weights["negative_space"]
            + mask_score["aligned_iou"] * weights["aligned_iou"]
            + mask_score["aligned_dice"] * weights["aligned_dice"]
        ) * dimension_gate_multiplier(dimension_score, family)

    weights = geometry_final_weights(family)
    return (
        feature_score * weights["feature"]
        + geometry_score * weights["geometry"]
        + recipe_score * weights["recipe"]
        + dimension_score * weights["dimension"]
        + quality_score * weights["quality"]
        + mask_score["positive_border_score"] * weights["positive_border"]
        + mask_score["negative_space_score"] * weights["negative_space"]
        + mask_score["aligned_iou"] * weights["aligned_iou"]
        + mask_score["aligned_dice"] * weights["aligned_dice"]
        + mask_score["centered_iou"] * weights["centered_iou"]
    ) * dimension_gate_multiplier(dimension_score, family)


def load_feature_records(paths: list[str]) -> list[dict]:
    records = []
    seen = set()
    for path in paths:
        for item in json.loads(Path(path).read_text(encoding="utf-8")):
            key = goose_identity_key(item)
            if key in seen:
                continue
            seen.add(key)
            records.append(item)
    return records


def load_mask(path: str, cache: dict[str, Image.Image]) -> Image.Image:
    if path not in cache:
        cache[path] = Image.open(path).convert("L")
    return cache[path]


def candidate_quality_score(goose: dict, goose_mask: Image.Image, target: dict) -> float:
    quality = float(goose.get("goose_candidate_score", 1) or 0)
    mask_quality = float(goose.get("mask_quality_score", 1) or 0)
    if goose.get("goose_candidate_reasons"):
        quality -= min(0.35, 0.16 * len(goose.get("goose_candidate_reasons") or []))
    if goose.get("mask_quality_reasons"):
        quality -= min(0.35, 0.1 * len(goose.get("mask_quality_reasons") or []))

    detection = goose.get("detection") or {}
    detector_score = detection.get("score")
    if detector_score is not None:
        quality -= max(0.0, 0.38 - float(detector_score)) * 0.9

    coverage = goose.get("detector_mask_coverage")
    if coverage is not None:
        quality -= max(0.0, 0.24 - float(coverage)) * 1.1

    sam_support = goose.get("sam_support_coverage")
    if sam_support is not None:
        quality -= max(0.0, 0.18 - float(sam_support)) * 0.8

    hygiene = segmentation_quality_score(goose)
    quality = (
        quality * 0.24
        + mask_quality * 0.18
        + mask_cleanliness_score(goose_mask, target) * 0.22
        + hygiene * 0.36
    )
    return max(0.0, min(1.0, quality))


def segmentation_quality_score(goose: dict) -> float:
    score = float(goose.get("mask_quality_score", 1) or 0)
    score = min(score, float(goose.get("goose_candidate_score", 1) or 0))

    component_count = int(float(goose.get("mask_component_count", 1) or 0))
    largest_component_ratio = float(goose.get("mask_largest_component_ratio", 1) or 0)
    edge_contact = float(goose.get("mask_crop_edge_contact_ratio", 0) or 0)
    hole_ratio = float(goose.get("mask_hole_ratio", 0) or 0)
    fill_ratio = float(goose.get("fill_ratio", 0) or 0)
    thinness = float(goose.get("thinness_score", 0) or 0)
    aspect = float(goose.get("aspect_ratio", 1) or 1)
    area = float(goose.get("area", 0) or 0)

    if component_count > 2:
        score -= min(0.45, 0.16 * (component_count - 2))
    if largest_component_ratio < 0.92:
        score -= min(0.35, (0.92 - largest_component_ratio) * 1.1)
    if edge_contact > 0.035:
        score -= min(0.4, (edge_contact - 0.035) * 4.0)
    if hole_ratio > 0.08:
        score -= min(0.4, (hole_ratio - 0.08) * 1.8)
    if fill_ratio > 0.72:
        score -= min(0.38, (fill_ratio - 0.72) * 1.8)
    if fill_ratio < 0.16:
        score -= min(0.35, (0.16 - fill_ratio) * 2.0)
    if thinness > 0.78:
        score -= min(0.42, (thinness - 0.78) * 1.5)
    if area < 900:
        score -= 0.28
    if aspect < 0.18 or aspect > 5.0:
        score -= 0.28

    detection = goose.get("detection") or {}
    if detection.get("score") is not None:
        score -= max(0.0, 0.28 - float(detection.get("score"))) * 0.7
    coverage = goose.get("detector_mask_coverage")
    if coverage is not None:
        coverage_value = float(coverage)
        if coverage_value < 0.1:
            score -= min(0.35, (0.1 - coverage_value) * 2.2)
        if coverage_value > 0.78:
            score -= min(0.28, (coverage_value - 0.78) * 0.9)

    return max(0.0, min(1.0, score))


def mask_cleanliness_score(mask: Image.Image, target: dict, size: int = 96) -> float:
    normalized = ImageOps.contain(mask.convert("L"), (size, size))
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(normalized, ((size - normalized.width) // 2, (size - normalized.height) // 2))
    binary = canvas.point(lambda value: 255 if value > 127 else 0)
    pixels = binary.load()
    bbox = binary.getbbox()
    if not bbox:
        return 0.0

    area = 0
    perimeter = 0
    border_pixels = 0
    for y in range(size):
        for x in range(size):
            if pixels[x, y] <= 127:
                continue
            area += 1
            if x in {0, size - 1} or y in {0, size - 1}:
                border_pixels += 1
            if is_edge_pixel(pixels, x, y, size, size):
                perimeter += 1

    bbox_area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    fill_ratio = area / bbox_area
    edge_density = perimeter / max(1, area)
    border_ratio = border_pixels / max(1, perimeter)
    component_count = connected_component_count(binary, min_area=12)

    score = 1.0
    if component_count > 3:
        score -= min(0.45, (component_count - 3) * 0.08)
    if border_ratio > 0.08:
        score -= min(0.35, (border_ratio - 0.08) * 2.4)
    if edge_density > 0.42 and fill_ratio > 0.42:
        score -= min(0.35, (edge_density - 0.42) * 1.1)
    if fill_ratio > 0.78:
        score -= min(0.4, (fill_ratio - 0.78) * 1.8)

    target_aspect = float(target.get("aspect_ratio", 1) or 1)
    goose_aspect = (bbox[2] - bbox[0]) / max(1, bbox[3] - bbox[1])
    if target_aspect < 0.36 and goose_aspect > 0.8:
        score -= min(0.5, (goose_aspect - 0.8) * 0.55)

    return max(0.0, min(1.0, score))


def connected_component_count(mask: Image.Image, min_area: int = 12) -> int:
    binary = mask.convert("1")
    width, height = binary.size
    pixels = binary.load()
    seen: set[tuple[int, int]] = set()
    count = 0
    for y in range(height):
        for x in range(width):
            if (x, y) in seen or not pixels[x, y]:
                continue
            area = flood_area(pixels, x, y, width, height, seen)
            if area >= min_area:
                count += 1
    return count


def flood_area(pixels, start_x: int, start_y: int, width: int, height: int, seen: set[tuple[int, int]]) -> int:
    stack = [(start_x, start_y)]
    seen.add((start_x, start_y))
    area = 0
    while stack:
        x, y = stack.pop()
        area += 1
        for nx in (x - 1, x, x + 1):
            for ny in (y - 1, y, y + 1):
                if nx == x and ny == y:
                    continue
                if nx < 0 or ny < 0 or nx >= width or ny >= height or (nx, ny) in seen:
                    continue
                if pixels[nx, ny]:
                    seen.add((nx, ny))
                    stack.append((nx, ny))
    return area


def is_edge_pixel(pixels, x: int, y: int, width: int, height: int) -> bool:
    if x == 0 or y == 0 or x == width - 1 or y == height - 1:
        return True
    return (
        pixels[x - 1, y] <= 127
        or pixels[x + 1, y] <= 127
        or pixels[x, y - 1] <= 127
        or pixels[x, y + 1] <= 127
    )


def preselect_distance(goose: dict, target: dict) -> float:
    family = letter_family(target.get("letter", ""))
    weights = preselect_weights(family)
    return (
        (1 - dimension_bound_score(goose, target, family)) * weights["dimension"]
        + recipe_distance(goose, target, use_letter_recipes=False) * weights["recipe"]
        + geometry_distance(goose, target, family) * weights["geometry"]
        + global_feature_distance(goose, target) * weights["feature"]
        + (1 - segmentation_quality_score(goose)) * weights["quality"]
    )


def global_feature_distance(goose: dict, target: dict) -> float:
    weights = {
        "aspect_ratio": 0.28,
        "curvature_score": 0.18,
        "boldness_score": 0.17,
        "slant_score": 0.17,
        "fill_ratio": 0.2,
    }
    total = 0.0
    for key in FEATURE_KEYS:
        total += weights[key] * abs(float(goose.get(key, 0)) - float(target.get(key, 0)))
    return total


def structure_distance(goose: dict, target: dict, family: str) -> float:
    weights = STRUCTURE_WEIGHTS.get(family, STRUCTURE_WEIGHTS["stroke"])
    return sum(
        vector_distance(goose.get(key), target.get(key), fallback=0.5) * weight
        for key, weight in weights.items()
    )


def geometry_distance(goose: dict, target: dict, family: str) -> float:
    weights = geometry_stage_weights(family)
    return (
        structure_distance(goose, target, family) * weights["structure"]
        + roundness_distance(goose, target) * weights["roundness"]
    )


def roundness_distance(goose: dict, target: dict) -> float:
    return (
        abs(float(goose.get("fill_ratio", 0)) - float(target.get("fill_ratio", 0))) * 0.22
        + abs(float(goose.get("curvature_score", 0)) - float(target.get("curvature_score", 0))) * 0.12
        + abs(float(goose.get("boldness_score", 0)) - float(target.get("boldness_score", 0))) * 0.1
        + vector_distance(roundness_profile(goose), roundness_profile(target), fallback=0.5) * 0.28
        + vector_distance(goose.get("contour_grid_4x4"), target.get("contour_grid_4x4"), fallback=0.5) * 0.28
    )


def roundness_profile(record: dict) -> list[float]:
    grid = record.get("grid_3x3") or []
    if not isinstance(grid, list) or len(grid) < 9:
        return [0.0] * 8
    top = sum(float(value) for value in grid[0:3]) / 3
    middle = sum(float(value) for value in grid[3:6]) / 3
    bottom = sum(float(value) for value in grid[6:9]) / 3
    left = (float(grid[0]) + float(grid[3]) + float(grid[6])) / 3
    center = (float(grid[1]) + float(grid[4]) + float(grid[7])) / 3
    right = (float(grid[2]) + float(grid[5]) + float(grid[8])) / 3
    upper_roundness = (top + middle) / 2
    lower_roundness = (middle + bottom) / 2
    return [
        round(top, 4),
        round(middle, 4),
        round(bottom, 4),
        round(left, 4),
        round(center, 4),
        round(right, 4),
        round(upper_roundness, 4),
        round(lower_roundness, 4),
    ]


def recipe_distance(goose: dict, target: dict, use_letter_recipes: bool = False) -> float:
    recipe = LETTER_RECIPES.get(str(target.get("letter", "")))
    if not use_letter_recipes or not recipe:
        return vector_distance(recipe_signature(goose), recipe_signature(target), fallback=0.5)

    signature = recipe_signature(goose)
    required = recipe.get("require", recipe)
    forbidden = recipe.get("forbid", {})

    required_weight = 0.0
    required_total = 0.0
    for key, expected in required.items():
        actual = signature.get(key, 0.0)
        weight = recipe_feature_weight(key)
        required_total += max(0.0, expected - actual) * weight
        required_weight += weight

    forbidden_weight = 0.0
    forbidden_total = 0.0
    for key, threshold in forbidden.items():
        actual = signature.get(key, 0.0)
        weight = recipe_feature_weight(key) * 1.2
        forbidden_total += max(0.0, actual - threshold) * weight
        forbidden_weight += weight

    target_signature = recipe_signature(target)
    recipe_keys = set(required) | set(forbidden)
    support_keys = [key for key in target_signature if key not in recipe_keys]
    support_distance = sum(abs(signature[key] - target_signature[key]) for key in support_keys) / max(1, len(support_keys))
    required_distance = required_total / max(0.001, required_weight)
    forbidden_distance = forbidden_total / max(0.001, forbidden_weight)
    return required_distance * 0.48 + forbidden_distance * 0.36 + support_distance * 0.16


def recipe_feature_weight(key: str) -> float:
    if key in {"left_v", "center_v", "right_v", "top_h", "middle_h", "bottom_h", "diag_up", "diag_down"}:
        return 1.25
    if key in {"round", "upper_round", "lower_round"}:
        return 1.05
    return 0.8


def recipe_signature(record: dict) -> dict[str, float]:
    grid = normalized_grid(record.get("grid_3x3"), 9)
    top = average(grid[0:3])
    middle = average(grid[3:6])
    bottom = average(grid[6:9])
    left = average([grid[0], grid[3], grid[6]])
    center = average([grid[1], grid[4], grid[7]])
    right = average([grid[2], grid[5], grid[8]])

    horizontal_runs = normalized_grid(record.get("horizontal_run_signature"), 7)
    vertical_runs = normalized_grid(record.get("vertical_run_signature"), 5)
    horizontal_bands = normalized_grid(record.get("horizontal_bands"), 12)
    vertical_bands = normalized_grid(record.get("vertical_bands"), 12)
    diagonal_axis = normalized_grid(record.get("diagonal_axis_strength"), 2)

    top_h = max(top, average(horizontal_runs[0:2]), average(horizontal_bands[0:4]))
    middle_h = max(middle, average(horizontal_runs[2:5]), average(horizontal_bands[4:8]))
    bottom_h = max(bottom, average(horizontal_runs[5:7]), average(horizontal_bands[8:12]))
    left_v = max(left, average(vertical_runs[0:2]), average(vertical_bands[0:4]))
    center_v = max(center, average(vertical_runs[2:3]), average(vertical_bands[4:8]))
    right_v = max(right, average(vertical_runs[3:5]), average(vertical_bands[8:12]))

    contour = normalized_grid(record.get("contour_grid_4x4"), 16)
    top_contour = average(contour[0:8])
    bottom_contour = average(contour[8:16])
    round_score = clamp01(float(record.get("curvature_score", 0)) * 0.65 + average([top_contour, bottom_contour]) * 0.35)
    upper_round = clamp01(round_score * 0.55 + top_contour * 0.45)
    lower_round = clamp01(round_score * 0.55 + bottom_contour * 0.45)

    return {
        "top_h": clamp01(top_h),
        "middle_h": clamp01(middle_h),
        "bottom_h": clamp01(bottom_h),
        "left_v": clamp01(left_v),
        "center_v": clamp01(center_v),
        "right_v": clamp01(right_v),
        "diag_down": clamp01(diagonal_axis[0]),
        "diag_up": clamp01(diagonal_axis[1]),
        "round": round_score,
        "upper_round": upper_round,
        "lower_round": lower_round,
        "center_void": clamp01(1 - grid[4]),
        "upper_center_void": clamp01(1 - grid[1]),
        "lower_center_void": clamp01(1 - grid[7]),
        "bottom_void": clamp01(1 - bottom),
    }


def normalized_grid(values: object, length: int) -> list[float]:
    if not isinstance(values, list):
        return [0.0] * length
    normalized = [clamp01(float(value)) for value in values[:length]]
    if len(normalized) < length:
        normalized.extend([0.0] * (length - len(normalized)))
    return normalized


def average(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def letter_family(letter: str) -> str:
    if letter in SIMPLE_STEM_LETTERS:
        return "simple_stem"
    if letter in WIDE_HUMP_LETTERS:
        return "wide_hump"
    if letter in CURVED_LETTERS:
        return "curved"
    return "stroke"


def geometry_stage_weights(family: str) -> dict[str, float]:
    return {
        "stroke": {"structure": 0.78, "roundness": 0.22},
        "curved": {"structure": 0.38, "roundness": 0.62},
        "simple_stem": {"structure": 0.72, "roundness": 0.28},
        "wide_hump": {"structure": 0.66, "roundness": 0.34},
    }.get(family, {"structure": 0.65, "roundness": 0.35})


def preselect_weights(family: str) -> dict[str, float]:
    return {
        "stroke": {"feature": 0.07, "geometry": 0.2, "recipe": 0.14, "dimension": 0.34, "quality": 0.25},
        "curved": {"feature": 0.08, "geometry": 0.17, "recipe": 0.14, "dimension": 0.34, "quality": 0.27},
        "simple_stem": {"feature": 0.05, "geometry": 0.14, "recipe": 0.12, "dimension": 0.44, "quality": 0.25},
        "wide_hump": {"feature": 0.06, "geometry": 0.18, "recipe": 0.13, "dimension": 0.38, "quality": 0.25},
    }.get(family, {"feature": 0.07, "geometry": 0.2, "recipe": 0.14, "dimension": 0.34, "quality": 0.25})


def dimension_bound_score(goose: dict, target: dict, family: str) -> float:
    base_distance = dimension_distance(goose, target)
    score = max(0.0, 1 - min(base_distance * 1.25, 1))

    goose_width = bbox_width_ratio(goose)
    target_width = bbox_width_ratio(target)
    goose_height = bbox_height_ratio(goose)
    target_height = bbox_height_ratio(target)
    goose_aspect = max(0.05, float(goose.get("aspect_ratio", 1) or 1))
    target_aspect = max(0.05, float(target.get("aspect_ratio", 1) or 1))

    if family == "simple_stem":
        max_width = max(target_width * 2.7, target_width + 0.12)
        max_aspect = max(target_aspect * 3.2, target_aspect + 0.28)
        if goose_width > max_width:
            score *= max(0.0, 1 - (goose_width - max_width) / 0.22)
        if goose_aspect > max_aspect:
            score *= max(0.0, 1 - (goose_aspect - max_aspect) / 0.55)
        if goose_height < target_height * 0.62:
            score *= max(0.0, goose_height / max(0.01, target_height * 0.62))
    elif family == "wide_hump":
        if goose_width < target_width * 0.55:
            score *= max(0.0, goose_width / max(0.01, target_width * 0.55))
    else:
        max_width = target_width + (0.22 if family == "curved" else 0.18)
        if goose_width > max_width and target_width < 0.25:
            score *= max(0.0, 1 - (goose_width - max_width) / 0.3)

    return max(0.0, min(1.0, score))


def dimension_gate_multiplier(dimension_score: float, family: str) -> float:
    floor = {
        "simple_stem": 0.12,
        "wide_hump": 0.28,
        "curved": 0.35,
        "stroke": 0.28,
    }.get(family, 0.3)
    return floor + (1 - floor) * max(0.0, min(1.0, dimension_score))


def classifier_final_weights(family: str) -> dict[str, float]:
    weights = {
        "stroke": {
            "classifier": 0.18,
            "dimension": 0.21,
            "quality": 0.08,
            "geometry": 0.08,
            "recipe": 0.12,
            "feature": 0.03,
            "positive_border": 0.16,
            "negative_space": 0.12,
            "aligned_iou": 0.02,
            "aligned_dice": 0.0,
        },
        "curved": {
            "classifier": 0.18,
            "dimension": 0.2,
            "quality": 0.08,
            "geometry": 0.06,
            "recipe": 0.08,
            "feature": 0.03,
            "positive_border": 0.2,
            "negative_space": 0.15,
            "aligned_iou": 0.02,
            "aligned_dice": 0.0,
        },
        "simple_stem": {
            "classifier": 0.16,
            "dimension": 0.36,
            "quality": 0.1,
            "geometry": 0.08,
            "recipe": 0.08,
            "feature": 0.03,
            "positive_border": 0.12,
            "negative_space": 0.05,
            "aligned_iou": 0.02,
            "aligned_dice": 0.0,
        },
        "wide_hump": {
            "classifier": 0.17,
            "dimension": 0.27,
            "quality": 0.08,
            "geometry": 0.08,
            "recipe": 0.12,
            "feature": 0.03,
            "positive_border": 0.15,
            "negative_space": 0.08,
            "aligned_iou": 0.02,
            "aligned_dice": 0.01,
        },
    }
    return weights.get(family, weights["stroke"])


def geometry_final_weights(family: str) -> dict[str, float]:
    weights = {
        "stroke": {
            "feature": 0.08,
            "geometry": 0.15,
            "recipe": 0.19,
            "dimension": 0.22,
            "quality": 0.1,
            "positive_border": 0.14,
            "negative_space": 0.1,
            "aligned_iou": 0.02,
            "aligned_dice": 0.0,
            "centered_iou": 0.03,
        },
        "curved": {
            "feature": 0.1,
            "geometry": 0.11,
            "recipe": 0.15,
            "dimension": 0.22,
            "quality": 0.1,
            "positive_border": 0.18,
            "negative_space": 0.12,
            "aligned_iou": 0.02,
            "aligned_dice": 0.0,
            "centered_iou": 0.03,
        },
        "simple_stem": {
            "feature": 0.06,
            "geometry": 0.1,
            "recipe": 0.16,
            "dimension": 0.38,
            "quality": 0.13,
            "positive_border": 0.1,
            "negative_space": 0.03,
            "aligned_iou": 0.01,
            "aligned_dice": 0.0,
            "centered_iou": 0.03,
        },
        "wide_hump": {
            "feature": 0.09,
            "geometry": 0.14,
            "recipe": 0.17,
            "dimension": 0.31,
            "quality": 0.1,
            "positive_border": 0.1,
            "negative_space": 0.08,
            "aligned_iou": 0.0,
            "aligned_dice": 0.0,
            "centered_iou": 0.01,
        },
    }
    return weights.get(family, weights["stroke"])


def dimension_distance(goose: dict, target: dict) -> float:
    goose_aspect = max(0.05, float(goose.get("aspect_ratio", 1) or 1))
    target_aspect = max(0.05, float(target.get("aspect_ratio", 1) or 1))
    aspect_distance = min(1.0, abs(math_log_ratio(goose_aspect, target_aspect)) / 1.6)
    width_distance = abs(bbox_width_ratio(goose) - bbox_width_ratio(target))
    height_distance = abs(bbox_height_ratio(goose) - bbox_height_ratio(target))
    return aspect_distance * 0.42 + width_distance * 0.28 + height_distance * 0.3


def bbox_width_ratio(record: dict) -> float:
    if record.get("bbox_width_ratio") is not None:
        return float(record.get("bbox_width_ratio") or 0)
    bbox = record.get("bbox") or [0, 0, 1, 1]
    canvas_width = float(record.get("width") or record.get("target_size") or 220)
    return float(bbox[2] if len(bbox) >= 3 else 1) / max(1.0, canvas_width)


def bbox_height_ratio(record: dict) -> float:
    if record.get("bbox_height_ratio") is not None:
        return float(record.get("bbox_height_ratio") or 0)
    bbox = record.get("bbox") or [0, 0, 1, 1]
    canvas_height = float(record.get("height") or record.get("target_size") or 220)
    return float(bbox[3] if len(bbox) >= 4 else 1) / max(1.0, canvas_height)


def math_log_ratio(a: float, b: float) -> float:
    import math

    return math.log(max(0.05, a) / max(0.05, b))


def grid_distance(a: object, b: object) -> float:
    return vector_distance(a, b, fallback=0.5)


def vector_distance(a: object, b: object, fallback: float = 0.5) -> float:
    if not isinstance(a, list) or not isinstance(b, list):
        return fallback
    length = min(len(a), len(b))
    if length == 0:
        return fallback
    return sum(abs(float(a[index]) - float(b[index])) for index in range(length)) / length


if __name__ == "__main__":
    main()
