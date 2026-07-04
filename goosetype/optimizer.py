from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from .config import StyleConfig
from .glyph_targets import get_target
from .pose_library import GooseFeature, PoseLibrary


@dataclass(frozen=True)
class GlyphPlacement:
    id: str
    x: float
    y: float
    scale: float
    rotation: float
    flip_x: bool = False


@dataclass(frozen=True)
class GlyphComposition:
    letter: str
    geese: list[GlyphPlacement]
    score: float

    def to_dict(self) -> dict:
        return {
            "letter": self.letter,
            "geese": [asdict(goose) for goose in self.geese],
            "score": self.score,
        }


def generate_glyph(
    letter: str,
    pose_library: PoseLibrary,
    style: StyleConfig,
    seed: int | None = None,
) -> GlyphComposition:
    style = style.clamped()
    rng = random.Random(seed)
    target = get_target(letter)
    candidates = pose_library.filtered(style.pose_filter).geese
    if not candidates:
        raise ValueError("Pose library is empty.")

    ranked = sorted(
        candidates,
        key=lambda goose: _score_goose(goose, target.aspect_ratio, target.boldness_score, target.curvature_score, style),
        reverse=True,
    )
    parts = min(style.max_geese_per_glyph, max(1, target.preferred_parts))
    if style.abstraction > 0.72:
        parts = style.max_geese_per_glyph

    selected = ranked[: max(parts * 3, 1)]
    rng.shuffle(selected)
    selected = sorted(
        selected[:parts],
        key=lambda goose: _score_goose(goose, target.aspect_ratio, target.boldness_score, target.curvature_score, style),
        reverse=True,
    )

    placements = []
    spread = 0.22 if parts == 2 else 0.0
    for index, goose in enumerate(selected):
        offset = (index - 0.5) * spread if parts == 2 else 0.0
        placements.append(
            GlyphPlacement(
                id=goose.id,
                x=0.5 + offset,
                y=0.5 + style.cursive * 0.05 * index,
                scale=0.82 + style.boldness * 0.18 - abs(target.aspect_ratio - goose.aspect_ratio) * 0.08,
                rotation=goose.orientation_angle * 0.18 + style.slant * 18 + target.slant_score * 12,
                flip_x=(index == 1 and target.preferred_parts == 2),
            )
        )

    score = sum(
        _score_goose(goose, target.aspect_ratio, target.boldness_score, target.curvature_score, style)
        for goose in selected
    ) / len(selected)
    return GlyphComposition(letter=letter.upper(), geese=placements, score=round(score, 4))


def _score_goose(
    goose: GooseFeature,
    target_aspect: float,
    target_boldness: float,
    target_curvature: float,
    style: StyleConfig,
) -> float:
    aspect = 1.0 - min(abs(goose.aspect_ratio - target_aspect) / 2.0, 1.0)
    boldness = 1.0 - abs(goose.boldness_score - (target_boldness * 0.55 + style.boldness * 0.45))
    curvature = 1.0 - abs(goose.curvature_score - target_curvature)
    slant = 1.0 - min(abs(goose.slant_score - style.slant) / 2.0, 1.0)
    readability = style.readability * (aspect * 0.48 + curvature * 0.28 + boldness * 0.24)
    expressiveness = style.abstraction * (0.35 + abs(goose.slant_score) * 0.25 + goose.curvature_score * 0.4)
    return readability + expressiveness + slant * 0.18

