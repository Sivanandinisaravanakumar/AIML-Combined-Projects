import json, time
from pathlib import Path
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "ieee_cis" / "processed" / "train_features.csv"
API_URL = "http://localhost:30082/predict"
OUT = ROOT / "data" / "ieee_cis" / "incoming" / "replay_validation_results.jsonl"
N_FRAUD, N_NORMAL, DELAY = 15, 85, 1.0

df = pd.read_csv(CSV)
y = df["isFraud"].astype(int)
# same split as src/train_ieee_model.py (test_size=0.2, random_state=42, stratify)
_, valid_idx = train_test_split(df.index, test_size=0.2, random_state=42, stratify=y)
valid = df.loc[valid_idx]

sample = pd.concat([
    valid[valid["isFraud"] == 1].sample(N_FRAUD, random_state=1),
    valid[valid["isFraud"] == 0].sample(N_NORMAL, random_state=1),
]).sample(frac=1, random_state=2)

cols = ["card1", "card2", "card3", "card5", "card6", "addr1", "addr2", "DeviceType", "DeviceInfo"]

def card_key(row, idx):
    if pd.isna(row.get("card1")):
        return f"missing_{idx}"
    return "_".join("missing" if pd.isna(row.get(c)) else str(row.get(c))
                    for c in ["card1", "card2", "card3", "card5", "card6"])

tp = fp = fn = tn = 0
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as f:
    for i, (idx, row) in enumerate(sample.iterrows(), 1):
        payload = {
            "card_key": card_key(row, idx),
            "TransactionDT": float(row["TransactionDT"]),
            "TransactionAmt": float(row["TransactionAmt"]),
        }
        for c in cols:
            if pd.notna(row.get(c)):
                payload[c] = str(row[c])
        try:
            r = requests.post(API_URL, json=payload, timeout=30)
            r.raise_for_status()
            res = r.json()
        except Exception as e:
            print("FAILED", idx, e)
            continue
        actual = int(row["isFraud"])
        res["actual_isFraud"] = actual
        f.write(json.dumps(res) + "\n")
        pred = res["prediction"]
        tp += pred == 1 and actual == 1
        fp += pred == 1 and actual == 0
        fn += pred == 0 and actual == 1
        tn += pred == 0 and actual == 0
        print(f"{i:3d} actual={actual} pred={pred} p={res['fraud_probability']:.3f} {res['label']}")
        time.sleep(DELAY)

print(f"\nTP={tp} FP={fp} FN={fn} TN={tn}")
print("Results:", OUT)
