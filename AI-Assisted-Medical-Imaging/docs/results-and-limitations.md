# Experimental Results and Limitations

## 1. Classification Experiment

| Item | Value |
|---|---|
| Dataset | RSNA Pneumonia Detection Challenge |
| Model | DenseNet121 |
| Training approach | Transfer learning; classifier head training |
| Training sample | 320 patients |
| Validation sample | 80 patients |
| Total pilot sample | 400 patients |
| Epochs | 1 |
| Training loss | 0.6257 |
| Validation accuracy | 68.75% |
| Validation AUC | 0.75375 |

## 2. Confusion Matrix

Class order: NO_LUNG_OPACITY, PNEUMONIA.

| Actual / Predicted | NO_LUNG_OPACITY | PNEUMONIA |
|---|---:|---:|
| NO_LUNG_OPACITY | 25 | 15 |
| PNEUMONIA | 10 | 30 |

The validation set contained 80 samples. These results are preliminary and should not be interpreted as clinical performance.

## 3. Implemented Features

- Chest X-ray image and DICOM input support.
- DenseNet121-based pneumonia classification.
- Grad-CAM heatmap generation.
- Original scan and heatmap display in the React viewer.
- Case storage and audit history.
- Doctor review workflow.
- Orthanc DICOM server integration.
- RSNA ground-truth bounding-box visualization when annotations are available.
- Docker Compose deployment configuration.
- DVC tracking configuration for the RSNA model artifact.

## 4. Incomplete Work

A 3D U-Net segmentation experiment using MONAI was attempted, but training could not be completed because of local memory constraints. Therefore, no completed 3D U-Net segmentation performance metrics are reported.

Automatic continuous polling of Orthanc for new studies has not been verified as complete.

## 5. Limitations

- Small pilot training and validation samples.
- Only one training epoch was completed.
- Results may not generalize to other hospitals, scanners or patient populations.
- No independent external validation was performed.
- Ground-truth localization is available only for matching RSNA annotations.
- No clinical trial or regulatory validation has been performed.

## 6. Future Improvements

- Train and validate with a larger dataset and more epochs.
- Complete 3D U-Net segmentation with adequate GPU memory.
- Perform external validation and bias analysis.
- Implement and test automated Orthanc study ingestion.
- Improve user authentication, access control and deployment security.
- Conduct formal usability and clinical safety evaluations.

## 7. Safety Statement

This project is an educational research prototype only. It is not a medical device. Its predictions, confidence scores and visualizations must not be used for diagnosis, treatment or other clinical decisions.
