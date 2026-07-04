from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StyleConfig:
    readability: float = 0.8
    abstraction: float = 0.2
    boldness: float = 0.5
    slant: float = 0.0
    width: float = 0.5
    cursive: float = 0.0
    max_geese_per_glyph: int = 2
    render_mode: str = "photo"
    color_filter: str | None = None
    pose_filter: str | None = None

    def clamped(self) -> "StyleConfig":
        return StyleConfig(
            readability=_clamp01(self.readability),
            abstraction=_clamp01(self.abstraction),
            boldness=_clamp01(self.boldness),
            slant=max(-1.0, min(1.0, self.slant)),
            width=_clamp01(self.width),
            cursive=_clamp01(self.cursive),
            max_geese_per_glyph=max(1, min(2, int(self.max_geese_per_glyph))),
            render_mode=self.render_mode,
            color_filter=self.color_filter,
            pose_filter=self.pose_filter,
        )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))

