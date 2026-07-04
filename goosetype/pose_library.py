from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class GooseFeature:
    id: str
    source_image: str
    cutout_path: str
    mask_path: str
    silhouette_path: str
    aspect_ratio: float
    area: int
    perimeter: int
    width: int
    height: int
    orientation_angle: float
    center_of_mass: tuple[float, float]
    curvature_score: float
    thinness_score: float
    boldness_score: float
    slant_score: float
    pose_label: str = "unknown"

    @classmethod
    def from_dict(cls, data: dict) -> "GooseFeature":
        return cls(
            id=str(data["id"]),
            source_image=str(data.get("source_image", "")),
            cutout_path=str(data.get("cutout_path", "")),
            mask_path=str(data.get("mask_path", "")),
            silhouette_path=str(data.get("silhouette_path", "")),
            aspect_ratio=float(data.get("aspect_ratio", 1.0)),
            area=int(data.get("area", 0)),
            perimeter=int(data.get("perimeter", 0)),
            width=int(data.get("width", 1)),
            height=int(data.get("height", 1)),
            orientation_angle=float(data.get("orientation_angle", 0.0)),
            center_of_mass=tuple(data.get("center_of_mass", (0.5, 0.5))),
            curvature_score=float(data.get("curvature_score", 0.5)),
            thinness_score=float(data.get("thinness_score", 0.5)),
            boldness_score=float(data.get("boldness_score", 0.5)),
            slant_score=float(data.get("slant_score", 0.0)),
            pose_label=str(data.get("pose_label", "unknown")),
        )


class PoseLibrary:
    def __init__(self, geese: Iterable[GooseFeature]):
        self.geese = list(geese)

    @classmethod
    def from_json(cls, path: str | Path) -> "PoseLibrary":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        items = payload["geese"] if isinstance(payload, dict) and "geese" in payload else payload
        return cls(GooseFeature.from_dict(item) for item in items)

    def filtered(self, pose_label: str | None = None) -> "PoseLibrary":
        if not pose_label:
            return PoseLibrary(self.geese)
        return PoseLibrary(goose for goose in self.geese if goose.pose_label == pose_label)

