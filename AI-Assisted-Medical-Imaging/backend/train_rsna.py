
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import RandomRotation

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from common import (
    CLASSES,
    DenseNetCAM,
    get_device,
    pil_to_input,
    read_image
)

DATA = Path(r"F:\Medical-Imaging\datasets")
IMAGE_DIR = DATA / "stage_2_train_images"
CSV_FILE = (
    DATA
    / "stage_2_train_labels.csv"
    / "stage_2_train_labels.csv"
)

OUT = ROOT / "models" / "rsna_model.pth"
METRICS = ROOT / "models" / "rsna_metrics.json"


class XrayDataset(Dataset):
    def __init__(self, paths, labels, augment=False):
        self.paths = paths
        self.labels = labels

        # Apply random rotation only when augmentation is enabled.
        self.augment = (
            RandomRotation(degrees=5)
            if augment else None
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = Path(self.paths[index])

        image, is_dicom = read_image(
            path.read_bytes(),
            path.name
        )

        if not is_dicom:
            raise ValueError(f"Expected DICOM: {path}")

        # Augment PIL image before model preprocessing.
        if self.augment is not None:
            image = self.augment(image)

        x = pil_to_input(image)

        return x, self.labels[index]


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)

    if not CSV_FILE.is_file():
        raise FileNotFoundError(
            f"Labels CSV not found: {CSV_FILE}"
        )

    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f"DICOM image directory not found: {IMAGE_DIR}"
        )

    # Consolidate repeated annotations into patient-level labels.
    labels = {}

    with open(CSV_FILE, newline="") as f:
        for row in csv.DictReader(f):
            pid = row["patientId"]
            labels[pid] = max(
                labels.get(pid, 0),
                int(row["Target"])
            )

    available = {
        p.stem: p
        for p in IMAGE_DIR.glob("*.dcm")
    }

    records = [
        (available[pid], target)
        for pid, target in labels.items()
        if pid in available
    ]

    rng = random.Random(42)

    by_class = {
        c: [item for item in records if item[1] == c]
        for c in (0, 1)
    }

    count = min(
        200,
        len(by_class[0]),
        len(by_class[1])
    )

    if count < 2:
        raise RuntimeError(
            "Not enough labelled DICOM images."
        )

    selected = []

    for c in (0, 1):
        rng.shuffle(by_class[c])
        selected.extend(by_class[c][:count])

    rng.shuffle(selected)

    paths = [str(p) for p, _ in selected]
    targets = [y for _, y in selected]

    train_p, val_p, train_y, val_y = train_test_split(
        paths,
        targets,
        test_size=0.2,
        random_state=42,
        stratify=targets
    )

    # Augmentation is enabled only for training data.
    train_dl = DataLoader(
        XrayDataset(train_p, train_y, augment=True),
        batch_size=1,
        shuffle=True,
        num_workers=0
    )

    # Validation data remains unaugmented.
    val_dl = DataLoader(
        XrayDataset(val_p, val_y, augment=False),
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    device = get_device()

    print("Device:", device)
    print("Matched DICOMs:", len(records))
    print("Selected patients:", len(selected))
    print(
        "Training:", len(train_p),
        "| Validation:", len(val_p)
    )
    print("Data augmentation: RandomRotation +/- 5 degrees")
    print("Training augmentation: Enabled")
    print("Validation augmentation: Disabled")

    print("Class counts:", {
        "Target 0": count,
        "Target 1": count
    })

    print(
        "Loading DenseNet121 pretrained weights "
        "(if available)..."
    )

    model = DenseNetCAM(
        num_classes=2,
        pretrained=True
    ).to(device)

    # CPU pilot: freeze feature extractor.
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    model.features.eval()

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    model.features.eval()

    total_loss = 0.0

    for step, (x, y) in enumerate(train_dl, start=1):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = loss_fn(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)

        if step % 10 == 0:
            print(
                f"Training batch {step}/{len(train_dl)}"
            )

    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, y in val_dl:
            probs = torch.softmax(
                model(x.to(device)),
                dim=1
            )[:, 1]

            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(y.tolist())

    predictions = (
        np.array(all_probs) >= 0.5
    ).astype(int)

    result = {
        "dataset": "RSNA Pneumonia Detection Challenge",
        "training_type": (
            "400-patient CPU pilot; classifier head only; "
            "1 epoch; RandomRotation augmentation"
        ),
        "augmentation": "RandomRotation +/- 5 degrees (training only)",
        "train_patients": len(train_p),
        "validation_patients": len(val_p),
        "train_loss": total_loss / len(train_p),
        "validation_accuracy": float(
            accuracy_score(all_labels, predictions)
        ),
        "validation_confusion_matrix": confusion_matrix(
            all_labels,
            predictions,
            labels=[0, 1]
        ).tolist(),
        "validation_auc": float(
            roc_auc_score(all_labels, all_probs)
        ),
        "classes": [
            "NO_LUNG_OPACITY",
            "PNEUMONIA"
        ],
        "warning": (
            "Small validation subset; "
            "not clinical performance."
        )
    }

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(model.state_dict(), OUT)

    METRICS.write_text(
        json.dumps(result, indent=2)
    )

    print("\nTRAINING TEST COMPLETED")
    print(json.dumps(result, indent=2))
    print("Model saved:", OUT)
    print("Metrics saved:", METRICS)


if __name__ == "__main__":
    main()
