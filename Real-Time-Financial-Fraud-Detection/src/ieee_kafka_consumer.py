
import json
from pathlib import Path
from datetime import datetime, timezone

from kafka import KafkaConsumer


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "data" / "ieee_cis" / "incoming"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "streamed_transactions.jsonl"


consumer = KafkaConsumer(
    "fraud-transactions",
    bootstrap_servers=["localhost:9092"],
    group_id="ieee-cis-feature-monitor",
    auto_offset_reset="latest",
    enable_auto_commit=True,
    value_deserializer=lambda value: json.loads(
        value.decode("utf-8")
    ),
)

print("IEEE-CIS Kafka Consumer Started...")
print("Saving model input features to:", OUTPUT_FILE)

try:
    for message in consumer:
        transaction = message.value

        # Ignore messages from other models sharing this topic.
        if transaction.get("model") not in ("IEEE-CIS-XGBoost", "IEEE-CIS-XGBoost+GNN"):
            continue

        features = transaction.get("features")

        if not isinstance(features, dict) or not features:
            print("Skipping message: missing model features.")
            continue

        record = {
            "model": transaction.get("model"),
            "transaction_id": transaction.get("transaction_id"),
            "observed_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "features": features,
            "fraud_probability": transaction.get(
                "fraud_probability"
            ),
            "prediction": transaction.get("prediction"),
            "threshold": transaction.get("threshold"),
            "actual_isFraud": None,
        }

        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

        print("--------------------------------")
        print("IEEE-CIS TRANSACTION SAVED")
        print("Transaction ID:", record["transaction_id"])
        print("Prediction:", record["prediction"])
        print("Fraud probability:", record["fraud_probability"])
        print("--------------------------------\n")

except KeyboardInterrupt:
    print("\nIEEE-CIS Consumer stopped.")

finally:
    consumer.close()
