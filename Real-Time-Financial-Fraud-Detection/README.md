# AI-Based Credit Card Fraud Detection System

An end-to-end machine learning capstone project for credit card fraud detection using traditional ML, real-time streaming, feature stores, model monitoring, and containerized deployment.

> **Educational prototype only.** This project is not intended for real financial decisions or production banking use.

## 1. Project Overview

This project explores two fraud detection datasets and two complementary machine learning approaches:

- **ULB Credit Card Fraud Dataset:** An anonymized dataset with `Time`, `Amount`, and PCA-transformed transaction features (`V1`–`V28`).
- **IEEE-CIS Fraud Detection Dataset:** A larger transaction and identity dataset used for feature engineering, XGBoost training, and additional graph-based experiments.

The system supports transaction scoring through FastAPI and publishes prediction events to Apache Kafka.

## 2. Architecture

```text
                 Transaction Request
                         |
                         v
                    FastAPI API
                         |
             +-----------+-----------+
             |                       |
             v                       v
        Feature Store            ML Model
          (Feast)               (XGBoost)
             |                       |
             +-----------+-----------+
                         |
                         v
                 Fraud Prediction
                         |
                         v
                  Kafka Producer
                         |
                         v
               fraud-transactions
                         |
                         v
                  Kafka Consumer
                         |
                         v
                Transaction Alerts
```

Additional project components:

- Streamlit dashboard for analyst case review and monitoring.
- Evidently AI for data drift analysis.
- Graph Neural Network experiments for transaction relationship analysis.
- Docker and Kubernetes deployment for the IEEE-CIS API.

## 3. Technology Stack

| Area | Technologies |
|---|---|
| Language | Python |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn, XGBoost |
| Graph learning | PyTorch, PyTorch Geometric |
| API | FastAPI, Uvicorn |
| Feature store | Feast |
| Streaming | Apache Kafka |
| Dashboard | Streamlit |
| Monitoring | Evidently AI |
| Experiment tracking | MLflow |
| Containerization | Docker, Docker Compose |
| Orchestration | Kubernetes using Minikube with Docker driver |

### MLflow Tracking

MLflow is used to track IEEE-CIS XGBoost training experiments, parameters, and evaluation metrics.

Start the MLflow UI from the project root:

```powershell
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open the UI at:

```text
http://127.0.0.1:5000
```

### Deployment Environment

- Docker Compose runs the Kafka service.
- Minikube with the Docker driver hosts the IEEE-CIS FastAPI deployment.
- Kubernetes exposes the API through a NodePort service.

## 4. Model Performance

### Original ULB Credit Card Fraud Model

The following metrics are reported in the original ULB project README. They are reported separately from the IEEE-CIS experiments.

| Metric | Score |
|---|---:|
| ROC-AUC | 0.9805 |
| PR-AUC | 0.8382 |
| Precision | 0.4398 |
| Recall | 0.8571 |

### IEEE-CIS XGBoost Model

Validation results at the default classification threshold of 0.5:

| Metric | Score |
|---|---:|
| ROC-AUC | 0.8810 |
| PR-AUC | 0.3507 |
| Precision | 0.1583 |
| Recall | 0.7443 |
| F1-score | 0.2611 |

A separate threshold-tuning experiment used a threshold of approximately **0.7657** and reported:

| Metric | Score |
|---|---:|
| Precision | 0.3161 |
| Recall | 0.4549 |
| F1-score | 0.3730 |

Changing the classification threshold affects precision, recall, and F1-score. ROC-AUC and PR-AUC remain threshold-independent ranking metrics.

### Graph Neural Network Experiment

The second GNN experiment (GCN v2) achieved the following validation results at the card-node level:

| Metric | Score |
|---|---:|
| ROC-AUC | 0.8503 |
| Average Precision | 0.5749 |
| Precision | 0.3514 |
| Recall | 0.6575 |
| F1-score | 0.4580 |

The GNN validation results are based on 2,979 card nodes and should not be directly compared with transaction-level XGBoost metrics.

### Hybrid XGBoost + GNN Experiment

The XGBoost and GNN scores were combined experimentally using an 80% XGBoost and 20% GNN weighting.

| Metric | Threshold 0.5 | Tuned threshold 0.7968 |
|---|---:|---:|
| Precision | 0.1105 | 0.3045 |
| Recall | 0.8362 | 0.4546 |
| F1-score | 0.1952 | 0.3647 |
| PR-AUC | 0.3327 | 0.3327 |
| ROC-AUC | 0.8749 | 0.8749 |

The hybrid experiment did not outperform the tuned XGBoost validation result in this evaluation. The GNN is therefore treated as an additional experimental signal rather than a proven performance improvement.

### Evaluation Notes

- ULB and IEEE-CIS metrics come from different datasets and evaluation setups.
- XGBoost and hybrid metrics are transaction-level results; GNN metrics are card-node-level results.
- Thresholds were selected using validation data, so these results should not be treated as unbiased estimates of future performance.
- The deployed IEEE-CIS API uses a configured threshold of approximately **0.7968** and its tested response reported the model as `IEEE-CIS-XGBoost+GNN`.
- The Kaggle IEEE-CIS test dataset does not provide public fraud labels for direct final test evaluation.

These results are experimental and intended for educational demonstration, not real-world banking decisions.

## 5. Project Structure

```text
AIML-01-Fraud-Detection/
│
├── src/
│   ├── api.py
│   ├── ieee_api.py
│   ├── ieee_feature_engineering.py
│   ├── train_ieee_model.py
│   ├── ieee_kafka_consumer.py
│   ├── feast_api.py
│   ├── evaluate_hybrid.py
│   ├── drift_monitor.py
│   └── dashboard.py
│
├── models/
│   └── ieee_cis/
│
├── feature_repo/
│   └── feature_repo/
│       ├── feature_store.yaml
│       ├── feature_definitions.py
│       └── data/
│
├── data/
│   └── ieee_cis/
│
├── processed/
├── reports/
├── k8s/
│   └── ieee-deployment.yaml
│
├── Dockerfile
├── Dockerfile.ieee
├── Dockerfile.ieee.dockerignore
├── requirements-ieee.txt
└── README.md
```

Large source datasets and local runtime files are excluded from the repository where appropriate. Feast runtime files are included in the deployment build because the API uses the local feature store.

## 6. Dataset

### ULB Credit Card Fraud Dataset

Contains anonymized credit card transactions with:
- `Time`
- `Amount`
- `V1` to `V28`
- Fraud label

### IEEE-CIS Fraud Detection Dataset

The IEEE-CIS dataset contains transaction and identity information. Feature engineering includes:
- Transaction hour and day
- Transaction velocity features
- Previous transaction counts over time windows
- Average transaction amount features
- Device and address change indicators
- Device missing-value indicators

Dataset source: [IEEE-CIS Fraud Detection – Kaggle](https://www.kaggle.com/c/ieee-fraud-detection)

Download the dataset through Kaggle and place the extracted files in the expected local data directory before running the feature engineering pipeline.

## 7. API Usage

The project provides two FastAPI-based fraud prediction APIs for the ULB and IEEE-CIS datasets.

### Original ULB API

The ULB API accepts transaction features including `Time`, `V1` to `V28`, and `Amount`.

Endpoint:

```text
POST /predict
```

Run the API:

```powershell
docker compose up -d kafka
.\.venv\Scripts\python.exe -m uvicorn src.ieee_api:app --host 127.0.0.1 --port 8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

### IEEE-CIS API

The IEEE-CIS API uses FastAPI, XGBoost, Feast online features, and the configured GNN scoring integration.

Run the API locally:

```powershell
$env:KAFKA_DISABLED="1"
.\.venv\Scripts\python.exe -m uvicorn src.ieee_api:app --host 127.0.0.1 --port 8003
```

Swagger documentation:

```text
http://127.0.0.1:8003/docs
```

#### Available endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API health/status |
| POST | `/predict` | Score a transaction and publish its prediction event when Kafka is enabled |

#### Prediction request example

```json
{
  "card_key": "10000_111.0_150.0_117.0_debit",
  "TransactionDT": 86400,
  "TransactionAmt": 100.0
}
```

Required fields:
- `card_key`
- `TransactionDT`
- `TransactionAmt`

Additional card, address, and device fields are optional.

#### Prediction response

The response includes:
- Transaction ID
- Fraud probability
- Decision threshold
- Prediction and classification label
- Model identifier
- Feature source
- XGBoost probability
- GNN usage indicator
- Kafka publishing status

A successful Kubernetes API test returned a fraud probability of approximately `0.0550`, a `NORMAL` prediction, Feast Online Store as the feature source, and `MESSAGE_SENT` as the Kafka status.

For Kubernetes deployment, access the API through the Minikube service URL generated by:

```powershell
minikube service ieee-fraud-api --url
```

Append `/docs` to the generated URL to open Swagger documentation.

## 8. Kafka Streaming

The application publishes transaction predictions to the Kafka topic:

```text
fraud-transactions
```

Kafka listener configuration:

| Client | Bootstrap server |
|---|---|
| Windows host consumer | `localhost:9092` |
| Docker/Kubernetes API | `host.docker.internal:9093` |

Start Kafka:

```powershell
docker compose up -d kafka
```

Run the original ULB consumer:

```powershell
python -m src.kafka_consumer
```

Run the IEEE-CIS consumer:

```powershell
python -m src.ieee_kafka_consumer
```

The consumer receives transaction messages and displays prediction details.

## 9. Feast Feature Store

Feast is used as the online feature store for the IEEE-CIS fraud prediction API. It provides consistent transaction features during model inference.

Feature repository:

```text
feature_repo/feature_repo/
```

Important files:
- `feature_store.yaml` – Feast configuration.
- `feature_definitions.py` – Entity, data source, feature view, and feature service definitions.
- `data/fraud_features.parquet` – Offline feature data.
- Local registry and SQLite online store – Feature metadata and online feature values.

### Registered features

The `card_fraud_features` feature view contains:

| Feature | Description |
|---|---|
| `card_txn_count_1h` | Card transaction count over one hour |
| `card_txn_count_24h` | Card transaction count over 24 hours |
| `card_avg_amount_30d` | Historical average transaction amount |
| `amount_vs_card_average` | Current amount compared with historical card average |
| `device_changed` | Device change indicator |
| `address_changed` | Address change indicator |
| `device_missing` | Missing device information indicator |

The feature view uses `card` as its entity and `card_key` as its join key. The target fraud label is excluded from online inference features to avoid label leakage.

### Apply and materialize features

Run from the feature repository directory:

```powershell
cd .\feature_repo\feature_repo
..\..\.venv\Scripts\feast.exe apply
```

Materialize available feature data into the online store:

```powershell
..\..\.venv\Scripts\python.exe -c "from feast import FeatureStore; from datetime import datetime, timezone; s=FeatureStore(repo_path='.'); s.materialize_incremental(end_date=datetime.now(timezone.utc)); print('Materialization complete')"
```

The project verified:
- Feature view registration.
- Feature service registration (`fraud_detection_v1`).
- Online feature materialization.
- Online feature retrieval for a card key.
- Feature Service retrieval.

The IEEE-CIS API retrieves these features from the Feast online store during prediction. Historical records outside the materialization window may not be available in the local online store.

## 10. Streamlit Analyst Dashboard

The Streamlit dashboard provides a visual interface for reviewing fraud alerts, analyst cases, feature drift, and transaction network information.

Run the dashboard from the project root:

```powershell id="tqv8x5"
$env:OPENBLAS_NUM_THREADS="1"
.\.venv\Scripts\python.exe -m streamlit run .\src\dashboard.py --server.port 8503
```

Open:

```text id="2wjf7p"
http://localhost:8503
```

### Dashboard modules

- **Overview:** Displays model performance, drift summary, and open analyst cases.
- **Analyst Case Queue:** Shows historical fraud cases and allows analysts to update case statuses.
- **Data Drift:** Displays offline feature drift results and monitoring summaries.
- **Live Kafka Fraud Alerts:** Displays transactions consumed from the `fraud-transactions` topic.
- **Fraud Network:** Displays card-device relationships and shared-device investigation leads.

### Dashboard data

The analyst case queue uses historical labelled IEEE-CIS examples. Kafka alerts are collected separately from the streaming topic.

Case status changes are stored locally in:

```text id="by7vzf"
reports/analyst_cases.csv
```

Kafka alert history is stored in:

```text id="sc5g3p"
reports/kafka_alerts.csv
```

The dashboard is intended for educational analyst workflow demonstration. Shared devices and network connections are investigation signals, not proof of fraud.

## 11. Data Drift Monitoring

Evidently AI is used to monitor changes in transaction feature distributions.

The project includes two monitoring workflows:

### Offline data drift

`src/drift_monitor.py` compares training and test feature distributions.

Generated reports include:

```text id="qg0n3m"
reports/ieee_data_drift_report.html
```

The report contains feature-level drift information and an aggregate drift summary.

### Live Kafka data drift

`src/live_drift_monitor.py` compares reference transaction features with a sample of transactions collected from the Kafka stream.

Generated outputs include:

```text id="p7rvx3"
reports/ieee_live_drift_report.html
reports/live_retraining_trigger.json
```

The live monitoring workflow records

## 12. Docker Deployment

Docker is used to package the IEEE-CIS FastAPI application and its runtime dependencies.

Build the IEEE-CIS API image inside the Minikube Docker environment:

```powershell id="p85qhw"
minikube image build -t fraud-detection-ieee-api:latest -f Dockerfile.ieee .
```

Verify the image:

```powershell id="x6cs2a"
minikube image ls | Select-String "fraud-detection-ieee-api"
```

The `Dockerfile.ieee` image includes:
- IEEE-CIS FastAPI application code.
- XGBoost model artifacts.
- Feast feature definitions and local feature-store data.
- Python runtime dependencies.

The image exposes port `8000` and starts the API using Uvicorn.

The Kubernetes manifest uses `imagePullPolicy: Never`, so the image must be available in Minikube before the deployment is started.

Kafka runs separately through Docker Compose:

```powershell id="p5s2v0"
docker compose up -d kafka
```

This is a local demonstration setup. The container image and services are not deployed to a public cloud environment.

## 13. Kubernetes Deployment (Minikube)

The IEEE-CIS Fraud Detection API was successfully deployed and tested on a local Minikube cluster using the Docker driver.

Build the image inside Minikube:

```powershell
minikube image build -t fraud-detection-ieee-api:latest -f Dockerfile.ieee .
```

Apply the Kubernetes deployment and service:

```powershell
kubectl apply -f k8s/ieee-deployment.yaml
kubectl rollout status deployment/ieee-fraud-api
kubectl get pods
kubectl get services
```

Expose the service locally:

```powershell
minikube service ieee-fraud-api --url
```

Keep the terminal running while using the generated local service URL. Open `/docs` to access the FastAPI Swagger interface.

The deployment uses `imagePullPolicy: Never`, so the image must already exist in Minikube's local image environment. The Kubernetes Service exposes container port `8000` through NodePort `30082`. The API connects to Kafka at `host.docker.internal:9093`.

The Kubernetes API prediction endpoint was tested successfully. A test transaction was scored using Feast online features and the XGBoost + GNN scoring integration, and its corresponding event was verified in the Kafka `fraud-transactions` topic.

This is a local demonstration deployment, not a public cloud deployment.

## 14. Important Limitations

- Metrics in `models/ieee_cis/metrics.json` are measured at threshold 0.5 (PR-AUC 0.351, ROC-AUC 0.881, recall 74.4%, precision 15.8%). The API uses a tuned threshold of approximately 0.7968.
- At the tuned XGBoost validation threshold, precision is approximately 30.4%, recall 45.5%, and F1-score 0.365 for the hybrid validation experiment. The standalone XGBoost threshold result is precision 31.6%, recall 45.5%, and F1-score 0.373.
- The Live Kafka Alerts dashboard displays messages consumed from the Kafka topic. The demo feed may include replayed and manually generated transactions; it should not be interpreted as a real-world fraud stream.
- Live drift results depend on the streamed sample, including manual test and replay records. They demonstrate monitoring and trigger behavior, not confirmed production drift.
- Retraining is gated on confirmed labelled data. The orchestrator was verified in `NO_ACTION` and `WAITING_FOR_LABELS` states. A separate historical-data candidate backtest completed and passed its configured quality gate; this is a pipeline test, not live-labelled retraining or independent validation. Production promotion was not performed.
- The API's XGBoost + GNN scoring integration was exercised successfully, and the response reported `gnn_used: true`. The GNN experiment also has separately saved model artifacts. The trained GNN weights are not packaged as a standalone model in the API image; the API integration uses the available card-level GNN score data. The hybrid approach did not outperform tuned XGBoost in the validation experiment.
- GNN metrics (v2: Average Precision 0.575, ROC-AUC 0.850) are card-node-level results on 2,979 validation cards. They are not directly comparable to transaction-level XGBoost metrics.
- The XGBoost training run is tracked in MLflow by `src/train_ieee_model.py`. Additional GNN and hybrid experiment results are available in saved metrics and MLflow logging outputs.
- Kubernetes deployment was verified using Minikube with the Docker driver, not Docker Desktop Kubernetes.
- The local Feast SQLite configuration is intended for this educational project, not a production multi-instance feature-store architecture.
- Kafka events from both project APIs share the `fraud-transactions` topic. Consumers should identify the dataset and model fields in each message.
- Dataset anonymization limits the interpretation of some features.
- Validation metrics do not guarantee performance on future transactions. Thresholds should be selected according to operational requirements.
- Drift detection does not automatically retrain or redeploy the production model.
- This project is an educational prototype, not a production banking system.

## 15. Future Improvements

- Evaluate models on a strictly independent test set.
- Improve temporal validation and leakage prevention.
- Explore model calibration and cost-sensitive threshold selection.
- Improve graph construction and GNN evaluation.
- Add automated model retraining with approval controls.
- Add authentication, monitoring, and production-grade feature-store infrastructure.
- Separate Kafka topics or consumer groups by dataset and application.

## 16. Disclaimer

This project was developed for educational and capstone demonstration purposes. Predictions must not be used as the sole basis for approving, declining, or investigating real financial transactions.


## 17. Experiment Comparison (MLflow, validation split)

| Run | Threshold | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|
| XGBoost baseline | 0.50 | 0.158 | 0.744 | 0.261 | 0.351 | 0.881 |
| XGBoost tuned threshold | 0.766 | 0.316 | 0.455 | 0.373 | 0.351 | 0.881 |
| Hybrid XGBoost + GNN | 0.50 | 0.111 | 0.836 | 0.195 | 0.333 | 0.875 |
| Hybrid XGBoost + GNN | 0.797 | 0.304 | 0.455 | 0.365 | 0.333 | 0.875 |
| GCN v2 (card nodes) | - | 0.351 | 0.658 | 0.458 | 0.575 | 0.850 |
| GCN v1 (card nodes) | - | 0.000 | 0.000 | 0.000 | 0.123 | 0.500 |

- Moving the XGBoost threshold from 0.5 to 0.766 roughly doubled precision (0.158 to 0.316), lowered recall (0.744 to 0.455) and raised F1 (0.261 to 0.373).
- The hybrid blend did not beat tuned XGBoost (F1 0.365 vs 0.373), so only XGBoost is served by the API.
- GCN rows use 2,979 card nodes. XGBoost and hybrid use 118,108 validation transactions, so GCN numbers are not directly comparable.
- GCN v1 collapsed to predicting no fraud.
- Thresholds were selected on the validation set, so the numbers are optimistic. The Kaggle test file has no labels.

Log the saved outputs into MLflow and open the UI:

```powershell
.\.venv\Scripts\python.exe scripts\log_mlflow_runs.py
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow_ieee.db --host 127.0.0.1 --port 5001
```

Open http://127.0.0.1:5001, switch to Model training, open the `IEEE-CIS-Fraud-Detection` experiment and compare runs. Running the logging script twice adds duplicate runs.

## 18. Running the Demo

Run commands from the project root in separate PowerShell terminals.

### Start Kafka

```powershell
docker compose up -d kafka
```

### Check Kubernetes deployment

```powershell
kubectl get pods
kubectl get services
```

Expose the API locally:

```powershell
minikube service ieee-fraud-api --url
```

Keep this terminal open while using the generated URL.

API Swagger documentation:

```text
<generated-minikube-service-url>/docs
```

### Start the Kafka consumer

In a separate terminal:

```powershell
.\.venv\Scripts\python.exe -m src.ieee_kafka_consumer
```

### Start the Streamlit dashboard

In another terminal:

```powershell
$env:OPENBLAS_NUM_THREADS="1"
.\.venv\Scripts\python.exe -m streamlit run .\src\dashboard.py --server.port 8503
```

Dashboard URL:

```text
http://localhost:8503
```

### Test a transaction

Open the API Swagger page, select `POST /predict`, and submit a valid IEEE-CIS transaction request.

Verify that:
- The API returns a fraud probability and prediction.
- The response identifies the feature source as Feast Online Store.
- The Kafka message is published to `fraud-transactions`.
- The Kafka consumer receives the transaction event.

The demo uses local Docker, Minikube, and Kafka services. Keep the required services running during the demonstration.

## 19. Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Seven tests pass.

## 20. Monitoring and Retraining Verification

The project includes offline drift monitoring, live Kafka drift monitoring, and a retraining orchestration workflow.

### Verification results

| Component | Observed result |
|---|---|
| Offline Evidently report | 3 of 20 features drifted (15%) |
| Offline drift decision | `NO_ACTION` |
| Live drift trigger | 50% drift share |
| Live drift decision | Retraining review requested |
| Retraining orchestrator | `WAITING_FOR_LABELS` |
| Labelled incoming data | Not available for candidate training |
| Production model promotion | Not performed |

### Run live drift monitoring

From the project root:

```powershell
.\.venv\Scripts\python.exe .\src\live_drift_monitor.py
```

### Run retraining orchestration

```powershell
.\.venv\Scripts\python.exe .\src\retraining_orchestrator.py
```

### Retraining data requirement

The orchestrator checks:

```text
data/ieee_cis/incoming/labelled_transactions.csv
```

At least 1,000 valid labelled transactions containing both classes are required before candidate training can begin.

The current live workflow safely waits for confirmed labels instead of automatically retraining on unlabelled streaming data. The latest orchestrator result was `WAITING_FOR_LABELS`: drift share 50%, threshold 50%, no incoming labelled file, and no production model change.

A separate historical-data backtest of the candidate trainer completed using 472,432 training rows and 118,108 validation rows. Its quality gate passed, but candidate and baseline metrics were identical (PR-AUC 0.350733, ROC-AUC 0.881013, precision 0.316073, recall 0.454875, F1 0.372979). This verifies that the candidate-training pipeline can run; it does not demonstrate model improvement or independent generalisation because the backtest uses the existing labelled dataset and its established split. The candidate was not promoted.

Candidate ID: `20261118T155321Z`  
MLflow run ID: `af62825f93b144d194eb1f9668055fbf`

The production model was not changed. Current model SHA-256 reference:
`980AB4A90F9AAC66BD812D0E9979144EFC8673D9916CA6AC5E00CCBBA0CD58D5`

The offline drift report and live drift trigger use different inputs and monitoring runs; their results should not be interpreted as contradictory.

## 21. Architecture Diagram

See [docs/architecture.md](docs/architecture.md).

## 22. Additional Limitations

- The GNN and hybrid are offline experiments and are not served by the API.
- The Kaggle test file has no labels, so final metrics need an independent labelled split.

## Live Demo (Frontend)

Frontend: https://aiml-01-real-time-financial-fraud-d.vercel.app/

The frontend is a static page hosted on Vercel. It calls the FastAPI backend running on a local Docker Desktop Kubernetes cluster, exposed through a temporary Cloudflare tunnel, so it works only while the demo machine is on. The permanent code is in the GitHub repository.
