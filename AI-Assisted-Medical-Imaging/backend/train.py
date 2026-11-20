"""Fine-tune DenseNet121 on a chest X-ray dataset (Kaggle 'Chest X-Ray Images (Pneumonia)').

Expected layout:  <data>/train/{NORMAL,PNEUMONIA}/*.jpeg   and   <data>/test/{NORMAL,PNEUMONIA}/*.jpeg

    python backend/train.py --data data/chest_xray --epochs 5
    python backend/train.py --data data/demo --epochs 3          # synthetic smoke test
    python backend/train.py --data data/chest_xray --limit 400   # fast run on CPU
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from monai.transforms import Compose, RandRotate, RandZoom
from monai.utils import set_determinism
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from common import CLASSES, DenseNetCAM, get_device, load_unnormalised, normalise, to_tensor

ROOT = Path(__file__).resolve().parent.parent
EXT = {".jpg", ".jpeg", ".png"}


def list_samples(split_dir: Path, limit=None, rng=None):
    items = []
    for ci, cls in enumerate(CLASSES):
        files = sorted(p for p in (split_dir / cls).glob("*") if p.suffix.lower() in EXT)
        if limit and rng:
            rng.shuffle(files)
            files = files[:limit]
        items += [(p, ci) for p in files]
    return items


class XrayDataset(Dataset):
    def __init__(self, items, train: bool):
        self.items = items
        self.aug = (
            Compose([RandRotate(range_x=0.26, prob=0.5, padding_mode="zeros"),
                     RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5)])
            if train else None
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        x = load_unnormalised(Image.open(path))
        if self.aug is not None:
            x = to_tensor(self.aug(x))
        return normalise(x), label


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        p = torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy()
        probs += p.tolist()
        labels += y.tolist()
    return np.array(probs), np.array(labels)


def metrics(probs, labels):
    preds = (probs >= 0.5).astype(int)
    out = {"accuracy": float((preds == labels).mean()),
           "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
           "report": classification_report(labels, preds, target_names=CLASSES,
                                           output_dict=True, zero_division=0)}
    out["auc"] = float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "chest_xray"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--limit", type=int, default=None, help="max training images per class")
    ap.add_argument("--out", default=str(ROOT / "models"))
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    set_determinism(seed=a.seed)
    rng = random.Random(a.seed)
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print("device:", device)

    all_train = list_samples(data / "train", a.limit, rng)
    if not all_train:
        raise SystemExit(f"No images found in {data}/train/{{NORMAL,PNEUMONIA}}")
    rng.shuffle(all_train)
    n_val = max(2, int(0.15 * len(all_train)))  # the official Kaggle val set has only 16 images
    val_items, train_items = all_train[:n_val], all_train[n_val:]
    test_dir = data / "test" if (data / "test").exists() else data / "val"
    test_items = list_samples(test_dir)
    print(f"train {len(train_items)} | val {len(val_items)} | test {len(test_items)}")

    mk = lambda items, train: DataLoader(XrayDataset(items, train), batch_size=a.batch,
                                         shuffle=train, num_workers=0)
    train_dl, val_dl, test_dl = mk(train_items, True), mk(val_items, False), mk(test_items, False)

    counts = np.bincount([y for _, y in train_items], minlength=2)
    weights = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    model = DenseNetCAM(num_classes=2, pretrained=True).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)

    best, history = -1.0, []
    for ep in range(1, a.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        vp, vl = predict(model, val_dl, device)
        m = metrics(vp, vl)
        score = m["auc"] if m["auc"] is not None else m["accuracy"]
        history.append({"epoch": ep, "train_loss": total / len(train_items),
                        "val_acc": m["accuracy"], "val_auc": m["auc"]})
        print(f"epoch {ep}/{a.epochs} loss {history[-1]['train_loss']:.4f} "
              f"val_acc {m['accuracy']:.3f} val_auc {m['auc']}")
        if score >= best:
            best = score
            torch.save(model.state_dict(), out / "model.pth")

    model.load_state_dict(torch.load(out / "model.pth", map_location=device))
    tp, tl = predict(model, test_dl, device)
    result = {"test": metrics(tp, tl), "history": history, "classes": CLASSES,
              "train_images": len(train_items), "test_images": len(test_items)}
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    print("TEST accuracy:", result["test"]["accuracy"], "| AUC:", result["test"]["auc"])
    print("Saved:", out / "model.pth", "and", out / "metrics.json")


if __name__ == "__main__":
    main()
