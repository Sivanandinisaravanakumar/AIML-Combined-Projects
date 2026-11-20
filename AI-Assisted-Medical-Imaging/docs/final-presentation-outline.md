# Final Presentation – 10 Minute Outline

## Slide 1 – Title (30 seconds)
AI-Assisted Medical Imaging and Diagnosis System

- Capstone Project AIML-02
- Chest X-ray analysis research prototype
- Presenter: [Your name]

## Slide 2 – Problem Statement (1 minute)
- Medical image review can be time-consuming.
- Clinicians need tools to organize scans and review AI-generated findings.
- AI explanations can help users inspect model attention.
- The system is an assistive research prototype, not a replacement for doctors.

## Slide 3 – Project Objectives (45 seconds)
- Accept chest X-ray images and DICOM scans.
- Classify scans into pneumonia and no-lung-opacity categories.
- Generate Grad-CAM visual explanations.
- Store cases and support doctor review.
- Integrate with an Orthanc DICOM server.

## Slide 4 – System Architecture (1 minute)
Show the architecture diagram in `architecture-and-api.md`.

Explain:
- React frontend
- FastAPI backend
- DenseNet121 model
- SQLite storage
- Orthanc DICOM server
- Grad-CAM explainability

## Slide 5 – Dataset and Preprocessing (1 minute)
- RSNA Pneumonia Detection Challenge dataset.
- DICOM image handling.
- Resizing and intensity normalization.
- Training and validation split: 320 and 80 patients.
- Pilot experiment used 400 patients.

## Slide 6 – Model and Results (1 minute)
- DenseNet121 transfer learning.
- One training epoch completed.
- Validation accuracy: 68.75%.
- Validation AUC: 0.75375.
- Training loss: 0.6257.
- Show the confusion matrix from `results-and-limitations.md`.

Emphasize that these are preliminary pilot results.

## Slide 7 – Explainability and Viewer (1 minute)
Demonstrate:
- Scan upload
- Prediction and confidence
- Grad-CAM heatmap
- Case history
- Doctor review and audit information

Use actual screenshots from the working application.

## Slide 8 – DICOM Integration and Workflow (1 minute)
- Orthanc server integration.
- DICOM case handling.
- Ground-truth RSNA bounding-box visualization.
- Explain that automatic continuous study polling is not yet verified.

## Slide 9 – Limitations and Future Work (1 minute)
- Small pilot dataset and one epoch.
- 3D U-Net experiment was not completed because of memory constraints.
- Need larger-scale training and external validation.
- Need stronger security and clinical evaluation.
- Educational prototype only.

## Slide 10 – Conclusion (45 seconds)
- Developed an end-to-end medical imaging research prototype.
- Combined classification, explainability, case management and DICOM integration.
- Demonstrated a working software workflow.
- Further research and validation are necessary.

## Questions (remaining time)
Thank the audience and answer questions honestly.
