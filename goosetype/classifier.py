from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps


DEFAULT_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")


class SimpleGlyphCNN:
    def __new__(cls, *args, **kwargs):
        try:
            import torch.nn as nn
        except ImportError as error:
            raise RuntimeError("Glyph classifier requires torch. Install requirements-ml.txt.") from error

        class _SimpleGlyphCNN(nn.Module):
            def __init__(self, num_classes: int = len(DEFAULT_LABELS)) -> None:
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((3, 3)),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(0.25),
                    nn.Linear(128 * 3 * 3, 192),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.15),
                    nn.Linear(192, num_classes),
                )

            def forward(self, x):
                return self.classifier(self.features(x))

        return _SimpleGlyphCNN(*args, **kwargs)


@dataclass
class GlyphClassifier:
    model: object
    labels: list[str]
    device: str

    def score_mask(self, mask: Image.Image, target_letter: str) -> float:
        if target_letter not in self.labels:
            return 0.0

        import torch

        tensor = mask_to_emnist_tensor(mask).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
        return float(probabilities[self.labels.index(target_letter)].item())

    def top_predictions(self, mask: Image.Image, limit: int = 5) -> list[tuple[str, float]]:
        import torch

        tensor = mask_to_emnist_tensor(mask).to(self.device)
        with torch.no_grad():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0]
        values, indices = torch.topk(probabilities, k=min(limit, len(self.labels)))
        return [(self.labels[int(index)], float(value)) for value, index in zip(values, indices)]


def load_glyph_classifier(path: str | Path, device: str = "cpu") -> GlyphClassifier:
    import torch

    checkpoint = torch.load(str(path), map_location=device)
    labels = list(checkpoint.get("labels") or DEFAULT_LABELS)
    model = SimpleGlyphCNN(num_classes=len(labels))
    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return GlyphClassifier(model=model, labels=labels, device=device)


def mask_to_emnist_image(mask: Image.Image, size: int = 28, margin: int = 3) -> Image.Image:
    source = mask.convert("L").point(lambda value: 255 if value > 127 else 0)
    bbox = source.getbbox()
    canvas = Image.new("L", (size, size), 0)
    if not bbox:
        return canvas

    cropped = source.crop(bbox)
    target = max(1, size - margin * 2)
    contained = ImageOps.contain(cropped, (target, target), Image.Resampling.LANCZOS)
    x = (size - contained.width) // 2
    y = (size - contained.height) // 2
    canvas.paste(contained, (x, y))
    return canvas.point(lambda value: 255 if value > 20 else 0)


def mask_to_emnist_tensor(mask: Image.Image):
    import torch

    image = mask_to_emnist_image(mask)
    values = list(image.getdata())
    tensor = torch.tensor(values, dtype=torch.float32).view(1, 1, 28, 28) / 255.0
    return tensor
