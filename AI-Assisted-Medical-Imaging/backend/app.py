"""
FastAPI service: upload scan -> DenseNet121 prediction -> Grad-CAM heatmap -> doctor review.

Run: cd backend && uvicorn app:app --reload --port 8000
"""
import os
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import requests
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from common import (
    CLASSES,
    DenseNetCAM,
    get_device,
    grad_cam,
    heatmap_rgba,
    pil_to_input,
    read_image,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "rsna_model.pth"))
STORAGE = Path(os.getenv("STORAGE_DIR", ROOT / "storage"))

ORTHANC_URL = os.getenv("ORTHANC_URL", "").rstrip("/")
ORTHANC_USERNAME = os.getenv("ORTHANC_USERNAME", "")
ORTHANC_PASSWORD = os.getenv("ORTHANC_PASSWORD", "")

DB = STORAGE / "cases.db"
STORAGE.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AI-Assisted Medical Imaging and Diagnosis System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = get_device()
MODEL = DenseNetCAM(num_classes=2, pretrained=False).to(DEVICE)
MODEL_LOADED = False

if MODEL_PATH.exists():
    MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    MODEL_LOADED = True
else:
    print(
        f"[warn] {MODEL_PATH} not found - predictions will come "
        "from an UNTRAINED model. Run training first."
    )

MODEL.eval()


# ------------------------------------------------------------------
# Database
# ------------------------------------------------------------------

def db():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con


with closing(db()) as con:
    con.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            id TEXT PRIMARY KEY,
            filename TEXT,
            is_dicom INTEGER,
            orthanc_id TEXT,
            label TEXT,
            confidence REAL,
            prob_normal REAL,
            prob_pneumonia REAL,
            model_trained INTEGER,
            created_at TEXT,
            status TEXT DEFAULT 'pending',
            decision TEXT,
            final_label TEXT,
            notes TEXT,
            reviewer TEXT,
            reviewed_at TEXT
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT NOT NULL,
            action TEXT NOT NULL,
            decision TEXT,
            final_label TEXT,
            notes TEXT,
            reviewer TEXT,
            timestamp TEXT NOT NULL
        )
    """)

    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_audit_logs_case_id
        ON audit_logs(case_id)
    """)

    con.commit()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_case(case_id: str) -> dict:
    with closing(db()) as con:
        row = con.execute(
            "SELECT * FROM cases WHERE id = ?",
            (case_id,)
        ).fetchone()

    if row is None:
        raise HTTPException(404, "Case not found")

    return dict(row)


# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL_LOADED,
        "device": str(DEVICE),
        "orthanc": bool(ORTHANC_URL),
    }


# ------------------------------------------------------------------
# Prediction and Orthanc DICOM upload
# ------------------------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()

    try:
        img, is_dicom = read_image(data, file.filename or "upload")
    except Exception as e:
        raise HTTPException(400, f"Could not read image: {e}")

    x = pil_to_input(img).unsqueeze(0).to(DEVICE)
    cam, probs, c = grad_cam(MODEL, x)

    case_id = uuid.uuid4().hex[:12]

    img.resize((448, 448)).save(
        STORAGE / f"{case_id}_image.png"
    )

    heatmap_rgba(cam).save(
        STORAGE / f"{case_id}_heatmap.png"
    )

    orthanc_id = None

    if is_dicom and ORTHANC_URL:
        try:
            auth = (
                (ORTHANC_USERNAME, ORTHANC_PASSWORD)
                if ORTHANC_USERNAME
                else None
            )

            response = requests.post(
                f"{ORTHANC_URL}/instances",
                data=data,
                headers={
                    "Content-Type": "application/dicom"
                },
                auth=auth,
                timeout=30,
            )

            response.raise_for_status()

            orthanc_id = response.json().get("ID")

        except requests.RequestException as e:
            print("[warn] Orthanc upload failed:", e)

        except (ValueError, AttributeError) as e:
            print("[warn] Invalid Orthanc response:", e)

    created_at = now()

    with closing(db()) as con:
        con.execute(
            """
            INSERT INTO cases (
                id, filename, is_dicom, orthanc_id,
                label, confidence, prob_normal,
                prob_pneumonia, model_trained, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                case_id,
                file.filename,
                int(is_dicom),
                orthanc_id,
                CLASSES[c],
                float(probs[c]),
                float(probs[0]),
                float(probs[1]),
                int(MODEL_LOADED),
                created_at,
            ),
        )

        con.execute(
            """
            INSERT INTO audit_logs (case_id, action, timestamp)
            VALUES (?, ?, ?)
            """,
            (
                case_id,
                "SCAN_UPLOADED_AND_ANALYZED",
                created_at,
            ),
        )

        con.commit()

    return get_case(case_id)


# ------------------------------------------------------------------
# Case management
# ------------------------------------------------------------------

@app.get("/cases")
def list_cases():
    with closing(db()) as con:
        rows = con.execute(
            """
            SELECT * FROM cases
            ORDER BY created_at DESC
            LIMIT 100
            """
        ).fetchall()

    return [dict(r) for r in rows]


@app.get("/cases/{case_id}")
def case(case_id: str):
    return get_case(case_id)


@app.get("/cases/{case_id}/image")
def case_image(case_id: str):
    get_case(case_id)

    path = STORAGE / f"{case_id}_image.png"

    if not path.exists():
        raise HTTPException(404, "Image not found")

    return FileResponse(path)


@app.get("/cases/{case_id}/heatmap")
def case_heatmap(case_id: str):
    get_case(case_id)

    path = STORAGE / f"{case_id}_heatmap.png"

    if not path.exists():
        raise HTTPException(404, "Heatmap not found")

    return FileResponse(path)


# ------------------------------------------------------------------
# Doctor review
# ------------------------------------------------------------------

class Review(BaseModel):
    decision: Literal["agree", "override"]
    final_label: Literal["NO_LUNG_OPACITY", "PNEUMONIA"]
    notes: str = ""
    reviewer: str = "Doctor"


@app.post("/cases/{case_id}/review")
def review(case_id: str, body: Review):
    get_case(case_id)

    reviewed_at = now()

    with closing(db()) as con:
        cursor = con.execute(
            """
            UPDATE cases
            SET status='reviewed',
                decision=?,
                final_label=?,
                notes=?,
                reviewer=?,
                reviewed_at=?
            WHERE id=?
            """,
            (
                body.decision,
                body.final_label,
                body.notes,
                body.reviewer,
                reviewed_at,
                case_id,
            ),
        )

        if cursor.rowcount == 0:
            raise HTTPException(404, "Case not found")

        con.execute(
            """
            INSERT INTO audit_logs (
                case_id, action, decision, final_label,
                notes, reviewer, timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                case_id,
                "DOCTOR_REVIEW",
                body.decision,
                body.final_label,
                body.notes,
                body.reviewer,
                reviewed_at,
            ),
        )

        con.commit()

    return get_case(case_id)


# ------------------------------------------------------------------
# Audit history
# ------------------------------------------------------------------

@app.get("/cases/{case_id}/audit")
def get_case_audit(case_id: str):
    get_case(case_id)

    with closing(db()) as con:
        rows = con.execute(
            """
            SELECT * FROM audit_logs
            WHERE case_id=?
            ORDER BY id ASC
            """,
            (case_id,),
        ).fetchall()

    return [dict(row) for row in rows]

# ------------------------------------------------------------------
# RSNA Pneumonia Bounding Box Localization
# ------------------------------------------------------------------

@app.get("/cases/{case_id}/localization")
def case_localization(case_id: str):
    import csv
    from PIL import Image, ImageDraw

    record = get_case(case_id)

    if not record.get("is_dicom"):
        raise HTTPException(400, "Localization requires a DICOM scan")

    csv_path = Path(
        os.getenv("RSNA_LABELS_CSV", "/app/datasets/stage_2_train_labels.csv")
    )

    if not csv_path.exists():
        raise HTTPException(
            503,
            "RSNA labels CSV is not configured. Set RSNA_LABELS_CSV."
        )

    patient_id = Path(record["filename"]).stem
    image_path = STORAGE / f"{case_id}_image.png"

    if not image_path.exists():
        raise HTTPException(404, "Scan image not found")

    boxes = []

    with open(csv_path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if (
                row.get("patientId") == patient_id
                and row.get("Target") == "1"
                and all(
                    row.get(k) not in ("", None)
                    for k in ("x", "y", "width", "height")
                )
            ):
                boxes.append(row)

    if not boxes:
        raise HTTPException(
            404,
            "No ground-truth pneumonia bounding boxes found for this scan"
        )

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    scale_x = image.width / 1024
    scale_y = image.height / 1024

    for box in boxes:
        x = float(box["x"]) * scale_x
        y = float(box["y"]) * scale_y
        width = float(box["width"]) * scale_x
        height = float(box["height"]) * scale_y

        draw.rectangle(
            [x, y, x + width, y + height],
            outline="red",
            width=3
        )

    output_path = STORAGE / f"{case_id}_localization.png"
    image.save(output_path)

    return FileResponse(
        output_path,
        media_type="image/png",
        filename=f"{case_id}_localization.png"
    )