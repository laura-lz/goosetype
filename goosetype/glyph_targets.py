from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GlyphTarget:
    letter: str
    aspect_ratio: float
    boldness_score: float
    slant_score: float
    curvature_score: float
    preferred_parts: int = 1


_TARGETS = {
    "A": GlyphTarget("A", 0.78, 0.56, 0.05, 0.2, 2),
    "B": GlyphTarget("B", 0.68, 0.7, 0.0, 0.72, 2),
    "C": GlyphTarget("C", 0.76, 0.55, -0.05, 0.95, 1),
    "D": GlyphTarget("D", 0.74, 0.68, 0.0, 0.78, 2),
    "E": GlyphTarget("E", 0.62, 0.62, 0.0, 0.18, 2),
    "F": GlyphTarget("F", 0.58, 0.5, 0.0, 0.18, 1),
    "G": GlyphTarget("G", 0.8, 0.62, -0.04, 0.88, 1),
    "H": GlyphTarget("H", 0.76, 0.66, 0.0, 0.2, 2),
    "I": GlyphTarget("I", 0.28, 0.45, 0.0, 0.08, 1),
    "J": GlyphTarget("J", 0.42, 0.48, 0.08, 0.58, 1),
    "K": GlyphTarget("K", 0.74, 0.56, 0.12, 0.24, 2),
    "L": GlyphTarget("L", 0.58, 0.5, 0.0, 0.18, 1),
    "M": GlyphTarget("M", 0.95, 0.7, 0.0, 0.26, 2),
    "N": GlyphTarget("N", 0.78, 0.62, 0.18, 0.22, 2),
    "O": GlyphTarget("O", 0.82, 0.66, 0.0, 1.0, 1),
    "P": GlyphTarget("P", 0.63, 0.6, 0.0, 0.56, 2),
    "Q": GlyphTarget("Q", 0.84, 0.66, 0.1, 0.92, 2),
    "R": GlyphTarget("R", 0.72, 0.62, 0.12, 0.55, 2),
    "S": GlyphTarget("S", 0.66, 0.58, -0.08, 0.9, 1),
    "T": GlyphTarget("T", 0.72, 0.54, 0.0, 0.12, 1),
    "U": GlyphTarget("U", 0.78, 0.6, 0.0, 0.72, 1),
    "V": GlyphTarget("V", 0.78, 0.48, -0.05, 0.24, 1),
    "W": GlyphTarget("W", 1.05, 0.64, 0.0, 0.3, 2),
    "X": GlyphTarget("X", 0.78, 0.58, 0.0, 0.2, 2),
    "Y": GlyphTarget("Y", 0.78, 0.5, 0.08, 0.28, 1),
    "Z": GlyphTarget("Z", 0.72, 0.54, -0.08, 0.16, 1),
}


def get_target(letter: str) -> GlyphTarget:
    return _TARGETS.get(letter.upper(), GlyphTarget(letter.upper(), 0.7, 0.5, 0.0, 0.5, 1))

