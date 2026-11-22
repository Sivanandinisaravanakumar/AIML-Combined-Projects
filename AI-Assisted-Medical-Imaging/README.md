# AI-Assisted Medical Imaging and Diagnosis System

An AI-assisted chest X-ray research prototype that accepts PNG, JPG, and DICOM images, predicts whether a scan shows **Pneumonia** or **No Lung Opacity**, generates a Grad-CAM heatmap, and supports doctor review.

> **Research prototype only. Not a medical device. Predictions and confidence scores must not be used for clinical diagnosis or treatment decisions.**

## 1. Project Overview

The system combines deep learning and medical-image processing to support an experimental chest X-ray review workflow.

Main features:
- Upload chest X-ray images in PNG, JPG, or DICOM format.
- Classify images using a DenseNet121 model.
- Display predicted label and confidence score.
- Generate a Grad-CAM heatmap for model explainability.
- View the original scan and heatmap overlay.
- Adjust heatmap strength.
- Review AI predictions using Agree or Override.
- Store case information and doctor review details in SQLite.
- Maintain an audit log of scan uploads and doctor reviews.

## 2. Technology Stack

| Component | Technology |
|---|---|
| Deep learning | PyTorch, DenseNet121 |
| Medical image preprocessing | MONAI |
| DICOM processing | pydicom |
| Explainability | Grad-CAM |
| Backend API | FastAPI |
| Database | SQLite |
| Frontend | React, Vite |
| Image processing | Pillow, NumPy |

## 3. Dataset

This project uses the **RSNA Pneumonia Detection Challenge** dataset.

Dataset files used:
- `stage_2_train_images`: DICOM training images
- `stage_2_train_labels.csv`: patient labels and bounding-box annotations
- `stage_2_test_images`: DICOM test images

The training CSV contains multiple annotation rows for some patients. Patient-level labels are consolidated using the maximum `Target` value.

Class labels:
- `NO_LUNG_OPACITY` — Target 0
- `PNEUMONIA` — Target 1

The dataset is stored locally and is not included in this repository.

## 4. Model Training and Evaluation

A small CPU-based pilot training run was performed using 400 patients from the RSNA dataset.

| Item | Value |
|---|---:|
| Total sampled patients | 400 |
| Training patients | 320 |
| Validation patients | 80 |
| Training epochs | 1 |
| Training approach | DenseNet121 classifier head |
| Device | CPU |
| Validation accuracy | 67.5% |
| Validation AUC | 0.73125 |

Validation confusion matrix:

| | Predicted No Lung Opacity | Predicted Pneumonia |
|---|---:|---:|
| Actual No Lung Opacity | 30 | 10 |
| Actual Pneumonia | 16 | 24 |

Model artifacts:
- `models/rsna_model.pth` — RSNA pilot model used by the backend.
- `models/rsna_metrics.json` — pilot training and validation metrics.
- `models/model.pth` and `models/metrics.json` — separate synthetic demo artifacts; they are not the RSNA-trained model.

**Important:** These results are from a small pilot experiment and are not clinical performance measurements. The validation sample is limited and further training and independent validation are required.

## 5. Project Structure

```text
AIML-02-Medical-Imaging/
├── backend/
│   ├── app.py
│   ├── common.py
│   ├── train.py
│   ├── train_rsna.py
│   ├── make_demo_data.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   ├── package.json
│   └── ...
├── models/
│   ├── rsna_model.pth
│   ├── rsna_metrics.json
│   ├── model.pth
│   └── metrics.json
├── storage/
├── data/
├── docker-compose.yml
├── dvc.yaml
└── README.md
```

## 6. Setup

Use Python and Node.js on Windows.

Create and activate a virtual environment if one is not already available:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install backend dependencies:

```powershell
cd backend
pip install -r requirements.txt
```

Install frontend dependencies:

```powershell
cd ..\frontend
npm install
```

## 7. Run the Application

Run the backend from the `backend` directory:

```powershell
cd backend
python -m uvicorn app:app --host 127.0.0.1 --port 8010
```

Backend API:
- Health check: `http://127.0.0.1:8010/health`
- Swagger documentation: `http://127.0.0.1:8010/docs`

In a second terminal, run the frontend:

```powershell
cd frontend
npm run dev -- --host 127.0.0.1
```

Frontend:
- `http://127.0.0.1:5173/`

Keep both terminals running while using the application.

## 8. API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check backend and model status |
| POST | `/predict` | Upload and classify a scan |
| GET | `/cases` | List stored cases |
| GET | `/cases/{case_id}` | Get case details |
| GET | `/cases/{case_id}/image` | Retrieve original image |
| GET | `/cases/{case_id}/heatmap` | Retrieve Grad-CAM heatmap |
| POST | `/cases/{case_id}/review` | Save doctor review |
| GET | `/cases/{case_id}/audit` | View case audit history |

## 9. Testing and Verified Workflows

The following workflows were tested locally:

- Backend health endpoint returned a successful response.
- DICOM upload and AI prediction returned successfully.
- Original scan display worked in the frontend.
- Grad-CAM heatmap generation and display worked.
- Heatmap strength adjustment worked.
- Doctor Agree review was saved successfully.
- Doctor Override review was saved successfully.
- Case status was updated after review.
- Audit logs recorded scan upload and doctor review actions.
- Python backend files passed syntax compilation.
- Frontend production build completed successfully.

## 10. Current Limitations

- The model was trained on a small 400-patient pilot subset for one epoch.
- The validation metrics are preliminary and do not establish clinical accuracy.
- The model is intended for experimental chest X-ray classification only.
- Grad-CAM is an explainability visualization; it is not a validated lesion segmentation system.
- CT and MRI diagnosis are not implemented.
- Orthanc was not connected or verified in the tested local workflow.
- Docker deployment has not been verified as part of the tested workflow.
- User authentication and production-grade access control are not implemented.

## 11. Future Enhancements

- Train and validate using a larger dataset.
- Perform independent external validation.
- Improve model calibration and reliability.
- Integrate and test Orthanc for DICOM storage.
- Add user authentication and role-based access.
- Improve audit and security features.
- Explore dedicated models for CT and MRI.

## 12. Disclaimer

This application is an academic research prototype developed for educational purposes. It is not a medical device and must not be used to diagnose, treat, or make decisions about patients. All AI predictions require appropriate professional assessment.

## Project Documentation

- [Architecture and API Documentation](docs/architecture-and-api.md)
- [Experimental Results and Limitations](docs/results-and-limitations.md)
- [Final Presentation Outline](docs/final-presentation-outline.md)
- [Individual Reflection Template](docs/individual-reflection-template.md)

## Safety, validation and implementation status

This project is an educational research prototype, not a clinically validated
medical device. It must not be used for diagnosis, treatment, or emergency
decisions. Predictions and Grad-CAM visualizations can be inaccurate.

### Implemented
- RSNA Pneumonia Detection dataset with DICOM image reading.
- DenseNet121 binary classification training pilot.
- Image preprocessing and training-time augmentation.
- Grad-CAM visualization.
- FastAPI scan upload, prediction and case management.
- Doctor review/override workflow with SQLite audit history.
- React scan review interface and audit history display.
- Orthanc integration configuration in Docker Compose.

### Current model limitations
The current classifier was trained as a limited 400-patient, one-epoch CPU
pilot. Its validation metrics are preliminary and are not clinical performance
claims. A larger, independently validated training and evaluation process is
required before any real-world use.

### Segmentation limitation
The RSNA dataset used here provides 2D chest X-ray images and bounding-box
annotations, not 3D voxel-wise segmentation masks. Therefore, a trained 3D
U-Net segmentation model is not claimed as completed. A separate dataset with
suitable segmentation masks, such as an appropriately licensed medical
segmentation dataset, is required.

### DVC limitation
DVC configuration and remote storage must be checked and configured for the
team's actual storage provider. Do not assume that a local DVC cache is a
remote backup.

### Run with Docker Compose
From the project root:

```powershell
docker compose up -d --build
docker compose ps
```

Frontend: http://localhost:5173
Backend API: http://localhost:8000/docs
Orthanc: http://localhost:8042

The default Orthanc credentials in the development Compose configuration
must be changed before any network or production deployment.
