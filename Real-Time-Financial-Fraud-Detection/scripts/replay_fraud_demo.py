import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

CSV_PATH = Path("data/ieee_cis/processed/train_features.csv")
API_URL = "http://localhost:30082/predict"
OUTPUT_PATH = Path("data/ieee_cis/incoming/replay_fraud_demo_results.jsonl")

COLUMNS = [
    "TransactionID", "TransactionDT", "TransactionAmt",
    "card1", "card2", "card3", "card5", "card6",
    "addr1", "addr2", "DeviceType", "DeviceInfo",
]


def card_key(row):
    if pd.isna(row.get("card1")):
        return "missing_" + str(int(row["TransactionID"]))
    fields = ["card1", "card2", "card3", "card5", "card6"]
    return "_".join(
        "missing" if pd.isna(row.get(c)) else str(row.get(c)) for c in fields
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fraud", type=int, default=10)
    parser.add_argument("--legit", type=int, default=40)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(CSV_PATH, usecols=lambda c: c in COLUMNS or c == "isFraud")
    df = df.sort_values('TransactionDT').tail(118108)
    fraud = df[df["isFraud"] == 1].sample(
        n=min(args.fraud, int((df["isFraud"] == 1).sum())), random_state=args.seed
    )
    legit = df[df["isFraud"] == 0].sample(n=args.legit, random_state=args.seed)
    batch = pd.concat([fraud, legit]).sample(frac=1, random_state=args.seed)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ok = failed = flagged = 0

    with OUTPUT_PATH.open("w", encoding="utf-8") as out:
        for i, (_, row) in enumerate(batch.iterrows(), start=1):
            payload = {
                "card_key": card_key(row),
                "TransactionDT": float(row["TransactionDT"]),
                "TransactionAmt": float(row["TransactionAmt"]),
            }
            for col in COLUMNS[3:]:
                if pd.notna(row.get(col)):
                    payload[col] = str(row.get(col))
            try:
                resp = requests.post(API_URL, json=payload, timeout=30)
                resp.raise_for_status()
                result = resp.json()
                result["true_label"] = int(row["isFraud"])
                out.write(json.dumps(result) + "\n")
                out.flush()
                ok += 1
                if result.get("prediction") == 1:
                    flagged += 1
                    print(f"[{i}/{len(batch)}] FLAGGED true_label={int(row['isFraud'])} {result}")
            except Exception as exc:
                failed += 1
                print(f"[{i}/{len(batch)}] FAILED: {exc}")
            time.sleep(args.delay)

    print(f"Done. Sent: {ok} | Failed: {failed} | Flagged as fraud: {flagged}")
    print("Results:", OUTPUT_PATH)


if __name__ == "__main__":
    main()

