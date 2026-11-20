import json
from pathlib import Path
import pandas as pd
from kafka import KafkaProducer

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "creditcard.csv"

producer = KafkaProducer(
    bootstrap_servers=["localhost:9092"],
    value_serializer=lambda value: json.dumps(value).encode("utf-8")
)

def send_transaction(transaction):
    producer.send("fraud-transactions", value=transaction)
    producer.flush()
    print("Transaction sent:", transaction["transaction_id"])

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    features = df.drop(columns=["Class"]).iloc[0].to_dict()

    transaction = {
        "transaction_id": "TX1001",
        "features": features
    }

    send_transaction(transaction)
    producer.close()
