
from pathlib import Path
import json
import os
import joblib
import pandas as pd
import numpy as np
import uuid
import threading
import subprocess

from kafka import KafkaProducer, KafkaConsumer
from collections import deque
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from feast import FeatureStore
from xgboost import XGBClassifier


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "ieee_cis"
FEAST_REPO = ROOT / "feature_repo" / "feature_repo"
FEAST_CONFIG = FEAST_REPO / "feature_store.yaml"
FEAST_REGISTRY = FEAST_REPO / "data" / "registry.db"


# --------------------------------------------------
# LOAD MODEL CONFIGURATION
# --------------------------------------------------

with open(MODEL_DIR / "feature_columns.json") as f:
    FEATURE_COLUMNS = json.load(f)

with open(MODEL_DIR / "threshold.json") as f:
    THRESHOLD = json.load(f)["best_threshold"]

NUMERIC_COLS = FEATURE_COLUMNS["numeric"]
CATEGORICAL_COLS = FEATURE_COLUMNS["categorical"]
ALL_COLS = FEATURE_COLUMNS["all"]

encoder = joblib.load(MODEL_DIR / "ordinal_encoder.joblib")
medians = joblib.load(MODEL_DIR / "numeric_medians.joblib")

model = XGBClassifier()
model.load_model(str(MODEL_DIR / "ieee_xgboost.json"))


# --------------------------------------------------
# FEAST INITIALIZATION
# --------------------------------------------------

def initialize_feast():

    if not FEAST_CONFIG.exists():
        raise FileNotFoundError(
            f"Feast configuration not found: {FEAST_CONFIG}"
        )

    if not FEAST_REPO.exists():
        raise FileNotFoundError(
            f"Feast repository not found: {FEAST_REPO}"
        )

    print("Initializing Feast...")
    print("Feast repository:", FEAST_REPO)

    try:
        result = subprocess.run(
            ["feast", "apply"],
            cwd=str(FEAST_REPO),
            capture_output=True,
            text=True,
            check=True,
            timeout=180
        )

        print("Feast apply completed.")

        if result.stdout:
            print("Feast apply output:", result.stdout)

        if result.stderr:
            print("Feast apply messages:", result.stderr)

    except subprocess.CalledProcessError as exc:
        print("Feast apply failed.")
        print("STDOUT:", exc.stdout)
        print("STDERR:", exc.stderr)
        raise RuntimeError(
            "Unable to apply Feast feature definitions."
        ) from exc

    except Exception as exc:
        print("Feast initialization error:", repr(exc))
        raise

    if not FEAST_REGISTRY.exists():
        raise FileNotFoundError(
            f"Feast registry was not created: {FEAST_REGISTRY}"
        )

    feature_store = FeatureStore(
        repo_path=str(FEAST_REPO)
    )

    feature_views = feature_store.list_feature_views()

    view_names = [
        view.name for view in feature_views
    ]

    print("Registered Feast feature views:", view_names)

    if "card_fraud_features" not in view_names:
        raise RuntimeError(
            "card_fraud_features was not found "
            "after running feast apply."
        )

    print("Feast initialization successful.")

    return feature_store


store = initialize_feast()


# --------------------------------------------------
# KAFKA CONFIGURATION - AIVEN CLOUD
# --------------------------------------------------

KAFKA_TOPIC = "fraud-transactions"

# --------------------------------------------------
# KAFKA LIVE DASHBOARD CONSUMER
# --------------------------------------------------

dashboard_consumer = None
dashboard_consumer_lock = threading.Lock()
dashboard_poll_lock = threading.Lock()
dashboard_transactions = deque(maxlen=500)


def get_dashboard_consumer():
    global dashboard_consumer

    if os.getenv("KAFKA_DISABLED", "0") == "1":
        return None

    if dashboard_consumer is not None:
        return dashboard_consumer

    with dashboard_consumer_lock:
        if dashboard_consumer is not None:
            return dashboard_consumer

        try:
            bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
            username = os.getenv("KAFKA_SASL_USERNAME")
            password = os.getenv("KAFKA_SASL_PASSWORD")

            if not all([bootstrap_servers, username, password]):
                raise ValueError("Missing Kafka environment variables")

            config = {
                "bootstrap_servers": [
                    server.strip()
                    for server in bootstrap_servers.split(",")
                    if server.strip()
                ],
                "security_protocol": os.getenv(
                    "KAFKA_SECURITY_PROTOCOL", "SASL_SSL"
                ),
                "sasl_mechanism": os.getenv(
                    "KAFKA_SASL_MECHANISM", "SCRAM-SHA-256"
                ),
                "sasl_plain_username": username,
                "sasl_plain_password": password,
                "group_id": "fraud-vercel-dashboard",
                "auto_offset_reset": "earliest",
                "enable_auto_commit": True,
                "request_timeout_ms": 15000,
            }

            ca_file = os.getenv("KAFKA_SSL_CAFILE")
            if ca_file:
                if not os.path.isfile(ca_file):
                    raise FileNotFoundError("Kafka CA certificate file not found")
                config["ssl_cafile"] = ca_file

            dashboard_consumer = KafkaConsumer(
                KAFKA_TOPIC,
                **config
            )

            print("Dashboard Kafka consumer initialized")
            return dashboard_consumer

        except Exception as exc:
            dashboard_consumer = None
            print("Dashboard Kafka consumer error:", repr(exc))
            return None


def collect_dashboard_messages():
    consumer = get_dashboard_consumer()

    if consumer is None:
        return False

    try:
        with dashboard_poll_lock:
            records = consumer.poll(timeout_ms=200, max_records=100)

            for messages in records.values():
                for record in messages:
                    try:
                        message = record.value

                        if isinstance(message, bytes):
                            message = json.loads(message.decode("utf-8"))
                        elif isinstance(message, str):
                            message = json.loads(message)

                        message["received_at"] = datetime.now(
                            timezone.utc
                        ).isoformat()

                        dashboard_transactions.append(message)

                    except Exception as exc:
                        print("Invalid Kafka dashboard message:", repr(exc))

        return True

    except Exception as exc:
        print("Kafka dashboard polling error:", repr(exc))
        return False


producer = None
producer_lock = threading.Lock()


def get_kafka_producer():

    global producer

    if os.getenv("KAFKA_DISABLED", "0") == "1":
        return None

    if producer is not None:
        return producer

    with producer_lock:

        if producer is not None:
            return producer

        try:

            bootstrap_servers = os.getenv(
                "KAFKA_BOOTSTRAP_SERVERS"
            )

            username = os.getenv("KAFKA_SASL_USERNAME")
            password = os.getenv("KAFKA_SASL_PASSWORD")

            if not all([
                bootstrap_servers,
                username,
                password
            ]):
                raise ValueError(
                    "Missing Kafka environment variables"
                )

            kafka_config = {
                "bootstrap_servers": [
                    server.strip()
                    for server in bootstrap_servers.split(",")
                    if server.strip()
                ],
                "security_protocol": os.getenv(
                    "KAFKA_SECURITY_PROTOCOL",
                    "SASL_SSL"
                ),
                "sasl_mechanism": os.getenv(
                    "KAFKA_SASL_MECHANISM",
                    "SCRAM-SHA-256"
                ),
                "sasl_plain_username": username,
                "sasl_plain_password": password,
                "request_timeout_ms": 15000,
                "max_block_ms": 15000,
                "retries": 3,
                "acks": "all"
            }

            ca_file = os.getenv("KAFKA_SSL_CAFILE")

            if ca_file:

                if not os.path.isfile(ca_file):
                    raise FileNotFoundError(
                        "Kafka CA certificate file not found"
                    )

                kafka_config["ssl_cafile"] = ca_file

            producer = KafkaProducer(**kafka_config)

            print(
                "Aiven Kafka producer initialized:",
                bootstrap_servers
            )

            return producer

        except Exception as exc:

            producer = None

            print(
                "Kafka producer initialization failed:",
                repr(exc)
            )

            return None


def json_serializer_default(value):

    if isinstance(value, np.generic):
        return value.item()

    if pd.isna(value):
        return None

    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not JSON serializable"
    )


def publish_to_kafka(message):

    global producer

    if os.getenv("KAFKA_DISABLED", "0") == "1":
        return "DISABLED"

    try:

        kafka_producer = get_kafka_producer()

        if kafka_producer is None:
            return "ERROR"

        message_bytes = json.dumps(
            message,
            default=json_serializer_default
        ).encode("utf-8")

        future = kafka_producer.send(
            KAFKA_TOPIC,
            value=message_bytes
        )

        metadata = future.get(timeout=15)

        kafka_producer.flush(timeout=10)

        print(
            f"Kafka message delivered successfully: "
            f"topic={metadata.topic}, "
            f"partition={metadata.partition}, "
            f"offset={metadata.offset}"
        )

        return "MESSAGE_SENT"

    except Exception as exc:

        print(
            "Kafka message publishing failed:",
            repr(exc)
        )

        with producer_lock:

            if producer is not None:
                try:
                    producer.close(timeout=2)
                except Exception:
                    pass

            producer = None

        return "ERROR"


# --------------------------------------------------
# OPTIONAL GNN CARD-LEVEL SCORES
# --------------------------------------------------

GNN_SCORES = {}
GNN_WEIGHT = 0.2
XGB_WEIGHT = 0.8
HYBRID_THRESHOLD = THRESHOLD

_gnn_path = MODEL_DIR / "gnn_card_scores.json"

if _gnn_path.exists():

    with open(_gnn_path) as f:
        GNN_SCORES = json.load(f)

    _hm = MODEL_DIR / "hybrid_metrics.json"

    if _hm.exists():

        with open(_hm) as f:
            HYBRID_THRESHOLD = json.load(f).get(
                "hybrid_best_validation_threshold",
                THRESHOLD
            )


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="IEEE-CIS Fraud Detection API",
    version="1.0.0",
    description=(
        "IEEE-CIS XGBoost prediction with "
        "Feast online features and optional GNN scores"
    )
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# DASHBOARD API
# --------------------------------------------------

@app.get("/dashboard/stats")
def dashboard_stats():
    kafka_connected = collect_dashboard_messages()
    transactions = list(dashboard_transactions)

    fraud_count = sum(
        1 for item in transactions
        if str(item.get("label", "")).upper() == "FRAUD"
        or item.get("prediction") == 1
    )

    normal_count = sum(
        1 for item in transactions
        if str(item.get("label", "")).upper() == "NORMAL"
        or item.get("prediction") == 0
    )

    return {
        "kafka_connected": kafka_connected,
        "total_transactions": len(transactions),
        "fraud_transactions": fraud_count,
        "normal_transactions": normal_count,
        "recent_transactions": list(reversed(transactions))[:20]
    }


@app.get("/dashboard/transactions")
def dashboard_transactions_api():
    kafka_connected = collect_dashboard_messages()

    return {
        "kafka_connected": kafka_connected,
        "transactions": list(reversed(dashboard_transactions))[:50]
    }


# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------

class TransactionRequest(BaseModel):

    card_key: str
    TransactionDT: float
    TransactionAmt: float

    card1: str | None = None
    card2: str | None = None
    card3: str | None = None
    card5: str | None = None
    card6: str | None = None

    addr1: str | None = None
    addr2: str | None = None

    DeviceType: str | None = None
    DeviceInfo: str | None = None


# --------------------------------------------------
# HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "IEEE-CIS Fraud Detection API is running"
    }


# --------------------------------------------------
# FEAST DIAGNOSTIC ENDPOINT
# --------------------------------------------------

@app.get("/debug/feast")
def debug_feast():

    try:

        feature_views = store.list_feature_views()

        view_names = [
            view.name for view in feature_views
        ]

        return {
            "feature_repo_exists": FEAST_REPO.exists(),
            "feature_store_yaml_exists": FEAST_CONFIG.exists(),
            "registry_exists": FEAST_REGISTRY.exists(),
            "feature_views": view_names,
            "card_fraud_features_found": (
                "card_fraud_features" in view_names
            )
        }

    except Exception as exc:

        return {
            "feast_error": str(exc)
        }


# --------------------------------------------------
# FRAUD PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
def predict(request: TransactionRequest):

    try:

        # INPUT VALIDATION

        txn_dt = float(request.TransactionDT)
        transaction_amount = float(request.TransactionAmt)

        if not np.isfinite(txn_dt) or txn_dt < 0:

            raise HTTPException(
                status_code=400,
                detail=(
                    "TransactionDT must be a "
                    "finite non-negative number"
                )
            )

        if not np.isfinite(transaction_amount):

            raise HTTPException(
                status_code=400,
                detail="TransactionAmt must be finite"
            )

        # FETCH FEAST ONLINE FEATURES

        online = store.get_online_features(
            features=[
                "card_fraud_features:card_txn_count_1h",
                "card_fraud_features:card_txn_count_24h",
                "card_fraud_features:card_avg_amount_30d",
                "card_fraud_features:amount_vs_card_average",
                "card_fraud_features:device_changed",
                "card_fraud_features:address_changed",
                "card_fraud_features:device_missing",
            ],
            entity_rows=[
                {"card_key": request.card_key}
            ],
        ).to_dict()

        # FEATURE PREPARATION

        row = {
            "TransactionDT": txn_dt,
            "TransactionAmt": transaction_amount,
            "transaction_hour": (txn_dt // 3600) % 24,
            "transaction_day": txn_dt // 86400,

            "card_txn_count_1h":
                online["card_txn_count_1h"][0],

            "card_txn_count_24h":
                online["card_txn_count_24h"][0],

            "card_avg_amount_30d":
                online["card_avg_amount_30d"][0],

            "amount_vs_card_average":
                online["amount_vs_card_average"][0],

            "device_changed":
                online["device_changed"][0],

            "address_changed":
                online["address_changed"][0],

            "device_missing":
                online["device_missing"][0],
        }

        for col in CATEGORICAL_COLS:
            row[col] = getattr(request, col)

        X = pd.DataFrame(
            [row],
            columns=ALL_COLS
        )

        # NUMERIC PREPROCESSING

        for col in NUMERIC_COLS:

            X[col] = pd.to_numeric(
                X[col],
                errors="coerce"
            )

        X[NUMERIC_COLS] = X[NUMERIC_COLS].fillna(
            medians
        )

        # CATEGORICAL PREPROCESSING

        for col in CATEGORICAL_COLS:

            X[col] = (
                X[col]
                .astype("string")
                .fillna("__MISSING__")
                .astype(str)
            )

        encoded = encoder.transform(
            X[CATEGORICAL_COLS]
        )

        encoded_df = pd.DataFrame(
            encoded,
            columns=CATEGORICAL_COLS,
            index=X.index
        )

        # FINAL MODEL INPUT

        X_final = pd.concat(
            [
                X[NUMERIC_COLS].astype("float64"),
                encoded_df
            ],
            axis=1
        )[ALL_COLS].astype("float64")

        # XGBOOST PREDICTION

        xgb_probability = float(
            model.predict_proba(X_final)[0, 1]
        )

        # OPTIONAL GNN HYBRID PREDICTION

        gnn_score = GNN_SCORES.get(
            request.card_key
        )

        if gnn_score is not None:

            probability = (
                XGB_WEIGHT * xgb_probability
                + GNN_WEIGHT * float(gnn_score)
            )

            active_threshold = HYBRID_THRESHOLD

            model_name = "IEEE-CIS-XGBoost+GNN"

        else:

            probability = xgb_probability

            active_threshold = THRESHOLD

            model_name = "IEEE-CIS-XGBoost"

        prediction = int(
            probability >= active_threshold
        )

        label = (
            "FRAUD" if prediction else "NORMAL"
        )

        # TRANSACTION ID

        transaction_id = str(uuid.uuid4())

        # KAFKA MESSAGE

        kafka_message = {

            "model": model_name,

            "transaction_id": transaction_id,

            "features": X_final.iloc[0].to_dict(),

            "raw_features": (
                request.model_dump()
                if hasattr(request, "model_dump")
                else request.dict()
            ),

            "fraud_probability": probability,

            "prediction": prediction,

            "label": label,

            "threshold": active_threshold
        }

        # PUBLISH TO KAFKA

        kafka_status = publish_to_kafka(
            kafka_message
        )

        # API RESPONSE

        return {

            "transaction_id": transaction_id,

            "fraud_probability": probability,

            "threshold": active_threshold,

            "prediction": prediction,

            "label": label,

            "model": model_name,

            "feature_source": "Feast Online Store",

            "xgb_probability": xgb_probability,

            "gnn_used": gnn_score is not None,

            "kafka_status": kafka_status
        }

    except HTTPException:
        raise

    except Exception as error:

        print(
            "Prediction error:",
            repr(error)
        )

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
