# AIML Combined Projects

A collection of two end-to-end AI/ML prototypes covering **financial fraud detection** and **medical imaging diagnosis**, built with production-style tooling such as FastAPI, Kafka, Feast, Docker, DVC, and cloud deployment.

> **Disclaimer:** Both projects are academic and educational prototypes. They are not intended for real banking decisions or clinical diagnosis or treatment.

## Projects at a Glance

| # | Project | Domain | Core Technologies | Live Frontend |
|---|---|---|---|---|
| 1 | [AI-Based Credit Card Fraud Detection System](#project-1-ai-based-credit-card-fraud-detection-system) | Finance | XGBoost, GNN, FastAPI, Feast, Kafka, MLflow, Evidently | [Open](https://aiml-01-real-time-financial-fraud-d.vercel.app/) |
| 2 | [AI-Assisted Medical Imaging and Diagnosis System](#project-2-ai-assisted-medical-imaging-and-diagnosis-system) | Healthcare | DenseNet121, Grad-CAM, FastAPI, React, Orthanc, DVC | [Open](https://aiml-02-ai-assisted-medical-imaging.vercel.app/) |

## Quick Links

| Component | Project 1: Fraud Detection | Project 2: Medical Imaging |
|---|---|---|
| Frontend | [Vercel App](https://aiml-01-real-time-financial-fraud-d.vercel.app/) | [Vercel App](https://aiml-02-ai-assisted-medical-imaging.vercel.app/) |
| Backend API | [Render](https://aiml-combined-projects.onrender.com/) | [Cloudflare Quick Tunnel](https://charter-colours-cutting-episodes.trycloudflare.com) |
| API Docs (Swagger) | [/docs](https://aiml-combined-projects.onrender.com/docs) | [/docs](https://charter-colours-cutting-episodes.trycloudflare.com/docs) |
| Health Check | N/A | [/health](https://charter-colours-cutting-episodes.trycloudflare.com/health) |
| Source Code | [GitHub](https://github.com/Sivanandinisaravanakumar/AIML-Combined-Projects) | [GitHub](https://github.com/Sivanandinisaravanakumar/AIML-Combined-Projects) |

> **Note (Project 1):** The backend is hosted on Render. On a free-tier instance, the first request after inactivity may take some time because the service can go to sleep.
>
> **Note (Project 2):** The backend link uses a temporary Cloudflare Quick Tunnel. It works only while the tunnel is running and may change when restarted. A permanent cloud backend deployment is planned.

## Repository Structure

```text
AIML-Combined-Projects/
│
├── Real-Time-Financial-Fraud-Detection/     # Project 1
│   ├── src/
│   ├── requirements.txt
│   └── ...
│
├── AI-Assisted-Medical-Imaging/             # Project 2
│   ├── backend/
│   ├── frontend/
│   ├── data/rsna/
│   ├── models/
│   ├── storage/
│   ├── docker-compose.yml
│   ├── Dockerfile
│   ├── dvc.yaml
│   └── README.md
│
└── README.md                                # This file
```

---

# Project 1: AI-Based Credit Card Fraud Detection System

An AI-powered fraud detection system that analyzes financial transactions, predicts the probability of fraudulent activity, and supports real-time transaction monitoring using machine learning and streaming technologies.

## 1.1 Overview

Credit card fraud is a major challenge in digital financial transactions. This project uses machine learning to identify potentially fraudulent transactions based on transaction details and card-related behavioral features.

The system combines an XGBoost-based fraud detection model with Graph Neural Network (GNN) experiments and a hybrid scoring approach. It provides a FastAPI backend for predictions, a web frontend for transaction scoring, and Kafka-based event streaming for monitoring recent predictions.

## 1.2 Key Features

- Credit card transaction fraud prediction
- XGBoost-based machine learning model
- Graph Neural Network (GNN) experiments
- Hybrid XGBoost and GNN scoring experiments
- Transaction-level fraud probability
- Configurable fraud classification threshold
- FastAPI-based prediction service
- Feast feature store integration
- Kafka event publishing for scored transactions
- Cloud-based transaction monitoring dashboard
- Streamlit analyst dashboard for local use
- Model experiment tracking with MLflow
- Data drift monitoring using Evidently
- Docker-based development and deployment support
- Local Kubernetes/Minikube deployment experiments

## 1.3 System Architecture

### Current Cloud Deployment

```text
                 User
                  |
                  v
          Vercel Frontend
                  |
                  | HTTP Request
                  v
          Render FastAPI Backend
                  |
                  v
          Feast Feature Store
                  |
                  v
          Fraud Detection Model
          (XGBoost + available
           GNN score integration)
                  |
                  v
          Fraud Prediction
                  |
                  v
          Aiven Cloud Kafka
                  |
                  v
       Transaction Monitoring
          through Backend API
                  |
                  v
          Vercel Dashboard
```

The frontend sends transaction details to the Render-hosted FastAPI backend. The backend retrieves available features, generates a fraud prediction, and publishes the scored transaction as a Kafka event. The frontend retrieves monitoring information through backend dashboard endpoints.

### Local Development and Experimentation

The project also includes local development components such as Docker Compose, Minikube/Kubernetes experiments, a local Kafka consumer, and a Streamlit analyst dashboard. These are separate from the current public Vercel and Render deployment.

## 1.4 Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | XGBoost, Scikit-learn |
| Deep Learning / Graph Learning | PyTorch, Graph Neural Networks |
| Backend | FastAPI |
| Frontend | HTML, CSS, JavaScript |
| Feature Store | Feast |
| Streaming | Apache Kafka |
| Cloud Kafka | Aiven |
| Analyst Dashboard | Streamlit |
| Experiment Tracking | MLflow |
| Drift Monitoring | Evidently |
| Containerization | Docker |
| Orchestration Experiments | Kubernetes, Minikube |
| Frontend Hosting | Vercel |
| Backend Hosting | Render |
| Version Control | Git, GitHub |

## 1.5 Datasets

The project explores two fraud detection datasets:

1. **Credit Card Fraud Detection Dataset (ULB):** anonymized credit card transactions with fraud labels, used for model development and experimentation.
2. **IEEE-CIS Fraud Detection Dataset:** transaction and identity-related information, used for feature engineering and model experiments with transaction and card-related behavior.

## 1.6 Feature Engineering

The project uses transaction and behavioral features such as:

- Transaction amount
- Transaction timestamp
- Transaction hour and day
- Number of transactions made by a card within one hour
- Number of transactions made by a card within 24 hours
- Average transaction amount for a card
- Transaction amount compared with the card's average
- Device change indicator
- Address change indicator
- Missing device information indicator

The deployed prediction API accepts transaction details and retrieves available card-related features through Feast.

## 1.7 Machine Learning Models

**XGBoost:** the primary transaction-level fraud classification model. It produces a fraud probability for each submitted transaction.

**Graph Neural Network (GNN):** experiments explore relationships between cards and transaction behavior, to investigate whether graph-based information can improve fraud detection.

**Hybrid Model:** a hybrid scoring approach combines XGBoost and GNN scores. It is evaluated experimentally and compared with the standalone XGBoost model.

> Model metrics are based on the recorded validation experiments. They should not be interpreted as guaranteed real-world banking performance or independent external test results.

## 1.8 Model Evaluation

The project evaluates models using Precision, Recall, F1-score, ROC-AUC, PR-AUC, and Confusion Matrix.

Saved XGBoost validation results:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.8810 |
| PR-AUC | 0.3507 |
| Precision at 0.5 threshold | 0.1583 |
| Recall at 0.5 threshold | 0.7443 |
| F1-score at 0.5 threshold | 0.2611 |
| Tuned-threshold Precision | 0.3161 |
| Tuned-threshold Recall | 0.4549 |
| Tuned-threshold F1-score | 0.3730 |

The tuned XGBoost threshold is approximately **0.7657**. The hybrid experiment has a separately selected validation threshold, so its metrics and threshold should be reported separately.

## 1.9 Backend API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Backend root endpoint |
| GET | `/docs` | Interactive Swagger API documentation |
| POST | `/predict` | Predict fraud probability for a transaction |
| GET | `/dashboard/stats` | Retrieve transaction monitoring statistics |
| GET | `/dashboard/transactions` | Retrieve recent transaction records |

### Example Prediction Request

```json
{
  "card_key": "card_123",
  "TransactionDT": 86400,
  "TransactionAmt": 2500.50,
  "card1": 12345,
  "card2": 111,
  "card3": 150,
  "card5": 226,
  "card6": "credit",
  "addr1": 315,
  "addr2": 87,
  "DeviceType": "desktop",
  "DeviceInfo": "Windows"
}
```

The API returns a prediction containing the fraud probability and classification result, along with relevant model and threshold information. For the exact request and response schema, refer to the [live Swagger documentation](https://aiml-combined-projects.onrender.com/docs).

## 1.10 Kafka-Based Transaction Monitoring

The backend publishes scored transaction events to the Aiven Kafka topic `fraud-transactions`.

Each event contains the transaction ID, model name, transaction features, fraud probability, prediction label, and classification threshold. The cloud monitoring dashboard retrieves recent transaction information and statistics through the backend dashboard endpoints.

> **Important:** The current live flow publishes events when transactions are submitted for scoring. It does not automatically replay the entire historical dataset into Kafka.

## 1.11 Feast Feature Store

Feast is integrated to retrieve available card-related behavioral features for prediction:

- `card_txn_count_1h`
- `card_txn_count_24h`
- `card_avg_amount_30d`
- `amount_vs_card_average`
- `device_changed`
- `address_changed`
- `device_missing`

The API uses the available online feature values associated with the submitted card key.

## 1.12 Dashboards and Monitoring

- **Web Dashboard:** the Vercel frontend provides transaction scoring and a cloud-based monitoring view, communicating with the Render backend over HTTP.
- **Streamlit Analyst Dashboard:** a separate dashboard for local analyst-oriented visualization and experimentation.
- **MLflow:** tracks model experiments and evaluation results.
- **Evidently:** data quality and drift monitoring experiments.

## 1.13 Running Project 1 Locally

```bash
# 1. Clone the repository
git clone https://github.com/Sivanandinisaravanakumar/AIML-Combined-Projects.git

# 2. Navigate to the project
cd AIML-Combined-Projects/Real-Time-Financial-Fraud-Detection

# 3. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows

# 4. Install dependencies
pip install -r requirements.txt
```

**5. Configure environment variables** for the backend and any optional integrations you want to run locally. Do not commit API keys, Kafka passwords, certificates, or other secrets to GitHub.

```bash
# 6. Start the FastAPI backend (IEEE-CIS API module)
uvicorn src.ieee_api:app --reload

# 7. Open API docs at http://127.0.0.1:8000/docs

# 8. Run the Streamlit dashboard
python -m streamlit run src/dashboard.py --server.port 8503
# Then open http://localhost:8503
```

If your local setup requires additional environment variables or a different working directory, follow the configuration used by the project.

## 1.14 Deployment

| Layer | Platform |
|---|---|
| Frontend | Vercel |
| Backend | Render |
| Streaming | Aiven Cloud Kafka |

The cloud deployment does not require the developer's laptop to remain switched on. Local Docker, Minikube, and Streamlit components are used for development and experimentation, not as the backend path for the current public deployment.

## 1.15 Limitations

- Fraud predictions are model-based estimates and should not be treated as final banking decisions.
- Model performance depends on the dataset, feature quality, and classification threshold.
- The available Feast features may not represent every real-time behavioral change.
- The live Kafka stream contains submitted scoring events rather than an automatic replay of all historical transactions.
- GNN and hybrid experiments require further evaluation before making claims of consistent improvement.
- Independent evaluation on a separate untouched labelled test dataset is required before claiming production-level performance.
- The system is an academic prototype and requires additional security, scalability, and operational validation before real financial deployment.

## 1.16 Future Enhancements

- Integrate additional real-time transaction sources
- Improve online feature updates
- Add more robust card and device behavior tracking
- Evaluate models on an untouched independent test dataset
- Improve GNN and hybrid model performance
- Add user authentication and role-based access control
- Improve alerting and analyst review workflows
- Add automated model monitoring and retraining pipelines
- Improve deployment scalability and fault tolerance

---

# Project 2: AI-Assisted Medical Imaging and Diagnosis System

An end-to-end AI-powered medical imaging prototype that assists in detecting pneumonia from chest X-ray images using deep learning. The system combines medical image processing, DenseNet121 classification, Grad-CAM explainability, doctor review workflows, audit logging, and containerized deployment.

> **Disclaimer:** This project is developed for educational and research purposes only. It is not intended for clinical diagnosis or treatment decisions.

## 2.1 Overview

The system demonstrates how artificial intelligence can support medical image analysis. It uses the RSNA Pneumonia Detection dataset to classify chest X-ray images into **Pneumonia** or **Normal**, and provides Grad-CAM heatmaps to visualize the image regions that influenced the model's prediction.

A doctor review workflow allows reviewers to agree with or override predictions, add notes, and maintain an audit history.

## 2.2 Key Features

- Chest X-ray image upload and processing
- DICOM, JPG, and PNG image support
- Pneumonia classification using DenseNet121
- Prediction confidence display
- Grad-CAM heatmap visualization
- RSNA ground-truth localization visualization where annotations are available
- Case queue and case management
- Doctor review and prediction override
- Reviewer notes and timestamps
- SQLite-based audit history
- FastAPI backend
- React-based interactive frontend
- Orthanc DICOM server integration
- Docker Compose deployment
- DVC-based model artifact versioning

## 2.3 System Architecture

```text
             User / Doctor
                   |
                   v
             React Frontend
                 (Vite)
                   |
                   v
             FastAPI Backend
                   |
        +----------+----------+
        |          |          |
        v          v          v
   Image Input   ML Model   Case Management
        |       DenseNet121       |
        |          |              |
        v          v              v
   Preprocessing Prediction    SQLite Audit
                   |
             +-----+------+
             |            |
             v            v
          Grad-CAM    Doctor Review
             |
             v
       Visualization

       Orthanc DICOM Server
```

## 2.4 Technology Stack

| Component | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Model Architecture | DenseNet121 |
| Explainable AI | Grad-CAM |
| Backend | FastAPI |
| Frontend | React, Vite |
| Medical Image Processing | pydicom |
| Database | SQLite |
| DICOM Server | Orthanc |
| Containerization | Docker, Docker Compose |
| Model Versioning | DVC |
| Frontend Deployment | Vercel |
| Backend Development Exposure | Cloudflare Quick Tunnel |

## 2.5 Dataset

**RSNA Pneumonia Detection Challenge Dataset**

| Description | Count |
|---|---:|
| Total DICOM Images | 26,684 |
| Total Label Rows | 30,227 |
| Unique Patients | 26,684 |
| Pneumonia Positive | 6,012 |
| Normal / Negative | 20,672 |

The dataset includes bounding-box annotations for pneumonia-positive images.

Dataset source: [RSNA Pneumonia Detection Challenge – Kaggle](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)

The dataset is not included in this Git repository. Download it separately and follow its usage terms.

## 2.6 Machine Learning Model

The system uses **DenseNet121**, a convolutional neural network based on dense connectivity. The model is initialized using transfer learning, with a classification layer adapted for binary pneumonia classification.

### Model Workflow

1. Load the chest X-ray image.
2. Preprocess and normalize the image.
3. Pass the image through DenseNet121.
4. Generate pneumonia classification probabilities.
5. Display the predicted class and confidence.
6. Generate a Grad-CAM heatmap for visual explanation.

### Preliminary Model Evaluation

A small balanced pilot experiment was conducted using 400 patients:

- Training samples: 320
- Validation samples: 80
- Training duration: 1 epoch
- Training approach: classifier-head-only fine-tuning

| Metric | Result |
|---|---:|
| Validation Accuracy | 68.75% |
| ROC-AUC | 0.75375 |

Confusion Matrix:

```text
                 Predicted
                 Normal  Pneumonia

Actual Normal       25       15
Actual Pneumonia    10       30
```

> These are preliminary results from a small pilot experiment, not a clinically validated model evaluation.

## 2.7 Explainable AI: Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) visualizes image regions that contribute to the model's prediction. The system generates a heatmap over the original chest X-ray image to help users understand which regions influenced the classification.

Grad-CAM is an explanatory visualization, not proof of a correct diagnosis.

## 2.8 Doctor Review and Audit History

The application includes a human review workflow. Reviewers can:

- View AI-generated predictions
- Agree with the prediction
- Override the prediction
- Add reviewer notes
- Record review timestamps
- View previous review and audit records

## 2.9 Project Structure

```text
AI-Assisted-Medical-Imaging/
│
├── backend/
│   ├── app/
│   ├── main.py
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── vite.config.js
│
├── data/
│   └── rsna/
│
├── models/
├── storage/
│
├── docker-compose.yml
├── Dockerfile
├── dvc.yaml
├── dvc.lock
├── .dvc/
├── .gitignore
└── README.md
```

## 2.10 Installation and Setup

### Prerequisites

- Python 3.10 or later
- Node.js and npm
- Git
- Docker Desktop
- Docker Compose

### Clone and Run with Docker Compose

```bash
git clone https://github.com/Sivanandinisaravanakumar/AIML-Combined-Projects.git
cd AIML-Combined-Projects/AI-Assisted-Medical-Imaging

# Build and start the services
docker compose up -d --build

# Check running containers
docker compose ps

# Stop the services
docker compose down
```

### Local Service URLs

| Service | URL |
|---|---|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| API Documentation | http://localhost:8000/docs |
| Health Check | http://localhost:8000/health |
| Orthanc | http://localhost:8042 |

## 2.11 Backend API

The FastAPI backend provides endpoints for image processing, prediction, explainability, case management, and health monitoring.

| Endpoint | Purpose |
|---|---|
| `/health` | Backend and model health status |
| `/docs` | Interactive API documentation |
| Image upload endpoint | Upload medical images |
| Prediction endpoint | Generate classification results |
| Grad-CAM endpoint | Generate explanation heatmaps |
| Case endpoints | Manage cases and review history |
| Localization endpoint | Retrieve available RSNA ground-truth localization |

Refer to the live [Swagger API Documentation](https://charter-colours-cutting-episodes.trycloudflare.com/docs) for the exact endpoint paths and request formats.

## 2.12 DVC: Data and Model Versioning

Data Version Control (DVC) is used to manage machine learning artifacts and support reproducibility.

```bash
dvc status
dvc push
dvc pull
```

The RSNA dataset is not committed directly to GitHub.

## 2.13 Orthanc DICOM Integration

Orthanc is used as a DICOM server to demonstrate medical imaging storage and integration. The project includes an Orthanc service in Docker Compose and backend connectivity checks.

A complete automated workflow from a newly received Orthanc study through image retrieval, inference, and result storage remains future integration work until fully verified.

## 2.14 Deployment

**Frontend:** the React frontend is deployed using Vercel at https://aiml-02-ai-assisted-medical-imaging.vercel.app/

**Backend:** the FastAPI backend has been tested using Docker and exposed through a Cloudflare Quick Tunnel during development at https://charter-colours-cutting-episodes.trycloudflare.com

The current tunnel URL is temporary. A stable cloud backend deployment is planned.

The frontend uses the following Vite environment variable to connect to the backend:

```env
VITE_API_URL=https://your-backend-url
```

After changing this variable in Vercel, rebuild and redeploy the frontend.

## 2.15 Current Limitations and Future Enhancements

- Train and evaluate the model on a larger, representative dataset
- Improve model generalization and evaluate sensitivity, specificity, precision, recall, and calibration
- Implement and train a MONAI-based 3D U-Net segmentation model
- Complete automated Orthanc study retrieval and inference integration
- Move audit history to persistent cloud database storage
- Deploy the backend and DICOM services using a suitable cloud architecture
- Add authentication and role-based access control
- Improve monitoring, logging, and security
- Conduct appropriate clinical validation before any real-world medical use

---

## Educational Purpose

These projects demonstrate the application of machine learning, deep learning, explainable AI, graph learning, streaming systems, feature stores, medical image processing, backend development, frontend integration, containerization, model versioning, and human-in-the-loop review.

They are educational prototypes and must not be used as a substitute for professional financial or medical decision-making.

## License

These projects are developed for academic and educational purposes. Add an appropriate open-source license if you intend to distribute or reuse the code publicly.

## Author

**Sivanandini Saravanakumar**

GitHub: [Sivanandinisaravanakumar](https://github.com/Sivanandinisaravanakumar)
