#!/usr/bin/env python3
"""Fine-tune EfficientNet-B0 on your own feeder bird crops.

Expects one folder per species under --data-dir.  Folder names become the
class labels, so name them exactly how you want them to appear in detections:

    training-data/
      Blue Jay/
        f000882_...jpg
        f000884_...jpg
      Tufted Titmouse/
        ...
      Dark-eyed Junco/
        ...

How to build that folder from yolo-crops/:
    mkdir -p training-data/"Blue Jay"
    cp yolo-crops/bird/f000882_* training-data/"Blue Jay"/
    # repeat for each species

Training is two-phase:
  Phase 1 — Warm-up:   freeze the backbone, train only the new classification
                        head.  Fast convergence, prevents early over-fitting.
  Phase 2 — Fine-tune: unfreeze the whole network at a lower learning rate.
                        Adapts EfficientNet's features to your feeder imagery.

Output: a single .pt checkpoint containing model weights + label list,
        ready to drop into bird-detector/ as CLASSIFIER_BACKEND=efficientnet.

Usage:
    python scripts/train_efficientnet.py --data-dir training-data/

    # More epochs, custom output path
    python scripts/train_efficientnet.py \\
        --data-dir training-data/ \\
        --warmup-epochs 10 \\
        --finetune-epochs 20 \\
        --output data/models/feeder_birds.pt

    # Smaller batch if you run out of memory
    python scripts/train_efficientnet.py --data-dir training-data/ --batch-size 8

Install deps:
    pip install torch torchvision pillow
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("train-efficientnet")


# ── Data loading ──────────────────────────────────────────────────────────────

def _build_datasets(data_dir: Path, val_split: float, batch_size: int):
    """Return (train_loader, val_loader, class_names)."""
    import torch
    import torchvision.transforms as T
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader, random_split

    # Augmentation for training — helps a lot with small datasets
    train_transform = T.Compose([
        T.Resize(256),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        T.RandomRotation(15),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = ImageFolder(str(data_dir))
    class_names = full_dataset.classes
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    if n_train < 1:
        log.error("Not enough images for a train/val split (found %d total)", n_total)
        sys.exit(1)

    train_subset, val_subset = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # Apply separate transforms by wrapping the subsets
    class _TransformSubset(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            img, label = self.subset[idx]
            # img is a PIL Image at this point (ImageFolder default loader)
            return self.transform(img), label

    # Re-load as PIL (ImageFolder returns tensors by default only if transform set)
    full_dataset_pil = ImageFolder(str(data_dir))  # no transform → PIL images
    train_subset_pil, val_subset_pil = random_split(
        full_dataset_pil, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_ds = _TransformSubset(train_subset_pil, train_transform)
    val_ds   = _TransformSubset(val_subset_pil,   val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                               num_workers=2, pin_memory=True)

    return train_loader, val_loader, class_names


# ── Training loop ─────────────────────────────────────────────────────────────

def _train_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()
    import torch
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (out.argmax(1) == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


def _val_epoch(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    import torch
    total_loss = correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.item() * len(labels)
            correct += (out.argmax(1) == labels).sum().item()
            total += len(labels)
    return total_loss / total, correct / total


def _per_class_accuracy(model, loader, class_names, device) -> dict[str, float]:
    import torch
    model.eval()
    correct = {c: 0 for c in class_names}
    totals  = {c: 0 for c in class_names}
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            for pred, label in zip(preds, labels):
                name = class_names[label.item()]
                totals[name] += 1
                if pred == label:
                    correct[name] += 1
    return {
        c: correct[c] / totals[c] if totals[c] > 0 else 0.0
        for c in class_names
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    import torch
    import torch.nn as nn
    import torchvision.models as models

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        log.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_loader, val_loader, class_names = _build_datasets(
        data_dir, args.val_split, args.batch_size
    )
    n_classes = len(class_names)
    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)

    log.info("Classes (%d): %s", n_classes, ", ".join(class_names))
    log.info("Train: %d images   Val: %d images", n_train, n_val)

    if n_classes < 2:
        log.error("Need at least 2 species folders to train a classifier")
        sys.exit(1)

    # ── Model ─────────────────────────────────────────────────────────────────
    log.info("Loading EfficientNet-B0 (pretrained ImageNet) …")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, n_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # ── Phase 1: Warm-up — head only ──────────────────────────────────────────
    if args.warmup_epochs > 0:
        log.info("── Phase 1: Warm-up (%d epochs, head only) ──", args.warmup_epochs)
        for param in model.features.parameters():
            param.requires_grad = False

        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.warmup_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.warmup_epochs
        )

        for epoch in range(1, args.warmup_epochs + 1):
            t = time.monotonic()
            train_loss, train_acc = _train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss,   val_acc   = _val_epoch(model, val_loader, criterion, device)
            scheduler.step()
            elapsed = time.monotonic() - t
            log.info(
                "  Warm-up epoch %2d/%d  train_loss=%.4f acc=%.1f%%  "
                "val_loss=%.4f acc=%.1f%%  (%.1fs)",
                epoch, args.warmup_epochs,
                train_loss, train_acc * 100,
                val_loss,   val_acc   * 100,
                elapsed,
            )

    # ── Phase 2: Fine-tune — whole network ────────────────────────────────────
    if args.finetune_epochs > 0:
        log.info("── Phase 2: Fine-tune (%d epochs, full network) ──", args.finetune_epochs)
        for param in model.parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam(model.parameters(), lr=args.finetune_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.finetune_epochs
        )

        best_val_acc = 0.0
        best_state = None

        for epoch in range(1, args.finetune_epochs + 1):
            t = time.monotonic()
            train_loss, train_acc = _train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss,   val_acc   = _val_epoch(model, val_loader, criterion, device)
            scheduler.step()
            elapsed = time.monotonic() - t
            marker = " ← best" if val_acc > best_val_acc else ""
            log.info(
                "  Fine-tune epoch %2d/%d  train_loss=%.4f acc=%.1f%%  "
                "val_loss=%.4f acc=%.1f%%%s  (%.1fs)",
                epoch, args.finetune_epochs,
                train_loss, train_acc * 100,
                val_loss,   val_acc   * 100,
                marker, elapsed,
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)
            log.info("Restored best checkpoint (val_acc=%.1f%%)", best_val_acc * 100)

    # ── Per-class accuracy ────────────────────────────────────────────────────
    log.info("── Per-class accuracy (validation set) ──")
    per_class = _per_class_accuracy(model, val_loader, class_names, device)
    width = max(len(c) for c in class_names) + 2
    for cls, acc in sorted(per_class.items(), key=lambda x: -x[1]):
        bar = "█" * int(acc * 20)
        log.info("  %-*s %5.1f%%  %s", width, cls, acc * 100, bar)

    # ── Save checkpoint ───────────────────────────────────────────────────────
    torch.save({"model_state": model.state_dict(), "labels": class_names}, output_path)
    log.info("Saved checkpoint → %s", output_path)
    log.info(
        "\nTo use in the pipeline:\n"
        "  CLASSIFIER_BACKEND=efficientnet\n"
        "  EFFICIENTNET_MODEL_PATH=%s",
        output_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune EfficientNet-B0 on your feeder bird crops.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir", required=True, metavar="DIR",
        help="Directory with one sub-folder per species (folder name = label)",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/models/feeder_birds.pt",
        metavar="FILE",
        help="Output checkpoint path (default: data/models/feeder_birds.pt)",
    )
    parser.add_argument(
        "--warmup-epochs", type=int, default=5, metavar="N",
        help="Epochs to train the head only (default: 5)",
    )
    parser.add_argument(
        "--finetune-epochs", type=int, default=15, metavar="N",
        help="Epochs to fine-tune the full network (default: 15)",
    )
    parser.add_argument(
        "--warmup-lr", type=float, default=1e-3, metavar="LR",
        help="Learning rate for warm-up phase (default: 1e-3)",
    )
    parser.add_argument(
        "--finetune-lr", type=float, default=1e-4, metavar="LR",
        help="Learning rate for fine-tune phase (default: 1e-4)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, metavar="N",
        help="Batch size (default: 16 — reduce to 8 if out of memory)",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.2, metavar="F",
        help="Fraction of data held out for validation (default: 0.2)",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
