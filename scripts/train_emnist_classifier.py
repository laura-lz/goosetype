#!/usr/bin/env python3
from __future__ import annotations

import argparse
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset
except ImportError as error:
    raise SystemExit("Training requires torch. Install requirements-ml.txt.") from error

try:
    from torchvision.datasets import EMNIST
    from torchvision import transforms
except ImportError as error:
    raise SystemExit("Training on EMNIST requires torchvision: python3 -m pip install torchvision") from error

from PIL import ImageOps

from goosetype.classifier import DEFAULT_LABELS, SimpleGlyphCNN


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an EMNIST-style glyph classifier for GooseType candidate scoring.")
    parser.add_argument("--data-dir", default="data/emnist", help="Where torchvision should download/cache EMNIST.")
    parser.add_argument("--output", default="data/models/emnist_glyph_classifier.pt")
    parser.add_argument("--split", default="byclass", choices=("byclass", "letters"))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--limit-train", type=int, default=0, help="Optional cap for quick local smoke tests.")
    parser.add_argument("--limit-test", type=int, default=0, help="Optional cap for quick local smoke tests.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    labels = labels_for_split(args.split)
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda image: ImageOps.mirror(image.rotate(-90, expand=True))),
            transforms.ToTensor(),
        ]
    )
    train_dataset = EMNIST(root=args.data_dir, split=args.split, train=True, download=True, transform=transform)
    test_dataset = EMNIST(root=args.data_dir, split=args.split, train=False, download=True, transform=transform)
    train_dataset = filter_dataset(train_dataset, labels, split=args.split, limit=args.limit_train)
    test_dataset = filter_dataset(test_dataset, labels, split=args.split, limit=args.limit_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SimpleGlyphCNN(num_classes=len(labels)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = correct = 0
        for images, raw_targets in train_loader:
            targets = remap_targets(raw_targets, labels, args.split).to(args.device)
            images = images.to(args.device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * images.size(0)
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            total += images.size(0)
        accuracy = evaluate(model, test_loader, labels, args.split, args.device)
        print(
            f"epoch={epoch} train_loss={total_loss / max(1, total):.4f} "
            f"train_acc={correct / max(1, total):.4f} test_acc={accuracy:.4f}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"labels": labels, "model_state_dict": model.state_dict()}, output)
    print(f"Wrote classifier checkpoint to {output}")


def labels_for_split(split: str) -> list[str]:
    if split == "letters":
        return list(string.ascii_uppercase)
    return DEFAULT_LABELS


def filter_dataset(dataset, labels: list[str], split: str, limit: int):
    allowed = allowed_raw_targets(labels, split)
    indices = [index for index, target in enumerate(dataset.targets.tolist()) if int(target) in allowed]
    if limit > 0:
        indices = indices[:limit]
    return Subset(dataset, indices)


def allowed_raw_targets(labels: list[str], split: str) -> set[int]:
    if split == "letters":
        return {ord(label.lower()) - ord("a") + 1 for label in labels}
    classes = emnist_byclass_classes()
    return {classes.index(label) for label in labels if label in classes}


def remap_targets(raw_targets, labels: list[str], split: str):
    raw_to_label = raw_target_to_label_map(split)
    label_to_index = {label: index for index, label in enumerate(labels)}
    return torch.tensor([label_to_index[raw_to_label[int(value)]] for value in raw_targets], dtype=torch.long)


def raw_target_to_label_map(split: str) -> dict[int, str]:
    if split == "letters":
        return {index + 1: letter for index, letter in enumerate(string.ascii_uppercase)}
    return {index: label for index, label in enumerate(emnist_byclass_classes())}


def emnist_byclass_classes() -> list[str]:
    return list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")


def evaluate(model, loader, labels: list[str], split: str, device: str) -> float:
    model.eval()
    total = correct = 0
    with torch.no_grad():
        for images, raw_targets in loader:
            targets = remap_targets(raw_targets, labels, split).to(device)
            logits = model(images.to(device))
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            total += images.size(0)
    return correct / max(1, total)


if __name__ == "__main__":
    main()
