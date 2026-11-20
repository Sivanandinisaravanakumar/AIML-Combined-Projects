import time
import json
from pathlib import Path

import pandas as pd
import requests

CSV_PATH = Path("data/ieee_cis/processed/test_features.csv")
API_URL = "http://localhost:8000/predict"
OUTPUT_PATH = Path("data/ieee_cis/incoming/replay_results.jsonl")

BATCH_SIZE = 1000
DELAY_SECONDS = 0.10

columns = [
    "TransactionID", "TransactionDT", "TransactionAmt",
    "card1", "card2", "card3", "card5", "card6",
    "addr1", "addr2", "DeviceType", "DeviceInfo"
]

df = pd.read_csv(CSV_PATH, usecols=lambda c: c in columns, nrows=BATCH_SIZE)

def card_key(row):
    fields = ["card1", "card2", "card3", "card5", "card6"]
    values = [
        "missing" if pd.isna(row.get(c)) else str(row.get(c))
        for c in fields
    ]
    if pd.isna(row.get("card1")):
        return "missing_" + str(int(row["TransactionID"]))
    return "_".join(values)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

success = 0
failed = 0

with OUTPUT_PATH.open("w", encoding="utf-8") as output:
    for index, row in df.iterrows():
        payload = {
            "card_key": card_key(row),
            "TransactionDT": float(row["TransactionDT"]),
            "TransactionAmt": float(row["TransactionAmt"]),
        }

        for col in columns[3:]:
            value = row.get(col)
            if pd.notna(value):
                payload[col] = str(value)

        try:
            response = requests.post(API_URL, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            output.write(json.dumps(result) + "\n")
            output.flush()
            success += 1
        except Exception as exc:
            failed += 1
            print(f"FAILED row {index}: {exc}")

        if (success + failed) % 100 == 0:
            print(f"Processed: {success + failed}/1000 | Success: {success} | Failed: {failed}")

        time.sleep(DELAY_SECONDS)

print("\nReplay completed")
print("Success:", success)
print("Failed:", failed)
print("Results:", OUTPUT_PATH)
