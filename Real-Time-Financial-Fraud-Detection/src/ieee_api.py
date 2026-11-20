
from pathlib import Path
import json
import os
import joblib
import pandas as pd
import numpy as np
import uuid
import threading

from kafka import KafkaProducer
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

store = FeatureStore(repo_path=str(FEAST_REPO))


# --------------------------------------------------
# KAFKA CONFIGURATION
# --------------------------------------------------

KAFKA_TOPIC = "fraud-transactions"

KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "localhost:9092"
)

producer = None
producer_lock = threading.Lock()


def get_kafka_producer():
    """
    Create Kafka producer when needed.
    Retry automatically if the initial connection failed.
    """

    global producer

    if os.getenv("KAFKA_DISABLED", "0") == "1":
        return None

    if producer is not None:
        return producer

    with producer_lock:

        # Check again after acquiring the lock.
        if producer is not None:
            return producer

        try:
            producer = KafkaProducer(
                bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
                request_timeout_ms=10000,
                max_block_ms=10000,
                retries=2,
                acks="all"
            )

            print(
                "Kafka producer initialized successfully:",
                KAFKA_BOOTSTRAP_SERVERS
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
    """
    Convert NumPy and Pandas values into JSON-compatible values.
    """

    if isinstance(value, np.generic):
        return value.item()

    if pd.isna(value):
        return None

    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def publish_to_kafka(message):
    """
    Publish a transaction and wait for broker acknowledgement.
    """

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

        # Wait until Kafka acknowledges the message.
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

        # Reset producer so the next request can reconnect.
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
# FRAUD PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
def predict(request: TransactionRequest):

    try:

        # ------------------------------------------
        # INPUT VALIDATION
        # ------------------------------------------

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

        # ------------------------------------------
        # FETCH FEAST ONLINE FEATURES
        # ------------------------------------------

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

        # ------------------------------------------
        # FEATURE PREPARATION
        # ------------------------------------------

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

        # ------------------------------------------
        # NUMERIC PREPROCESSING
        # ------------------------------------------

        for col in NUMERIC_COLS:

            X[col] = pd.to_numeric(
                X[col],
                errors="coerce"
            )

        X[NUMERIC_COLS] = X[NUMERIC_COLS].fillna(
            medians
        )

        # ------------------------------------------
        # CATEGORICAL PREPROCESSING
        # ------------------------------------------

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

        # ------------------------------------------
        # FINAL MODEL INPUT
        # ------------------------------------------

        X_final = pd.concat(
            [
                X[NUMERIC_COLS].astype("float64"),
                encoded_df
            ],
            axis=1
        )[ALL_COLS].astype("float64")

        # ------------------------------------------
        # XGBOOST PREDICTION
        # ------------------------------------------

        xgb_probability = float(
            model.predict_proba(X_final)[0, 1]
        )

        # ------------------------------------------
        # OPTIONAL GNN HYBRID PREDICTION
        # ------------------------------------------

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

        # ------------------------------------------
        # TRANSACTION ID
        # ------------------------------------------

        transaction_id = str(uuid.uuid4())

        # ------------------------------------------
        # KAFKA MESSAGE
        # ------------------------------------------

        kafka_message = {

            "model": model_name,

            "transaction_id": transaction_id,

            # Exact features passed to the model.
            "features": X_final.iloc[0].to_dict(),

            # Original request for auditing.
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

        # ------------------------------------------
        # PUBLISH TO KAFKA
        # ------------------------------------------

        kafka_status = publish_to_kafka(
            kafka_message
        )

        # ------------------------------------------
        # API RESPONSE
        # ------------------------------------------

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
