
from pathlib import Path
import json

import joblib
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = (
    ROOT / "data" / "ieee_cis" / "processed"
    / "train_features.csv"
)

STREAM_FILE = (
    ROOT / "data" / "ieee_cis" / "incoming"
    / "streamed_transactions.jsonl"
)

MODEL_DIR = ROOT / "models" / "ieee_cis"
OUTPUT_DIR = ROOT / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_LIVE_ROWS = 1000
DRIFT_THRESHOLD = 0.50

print("Loading reference training data...")

if not TRAIN_FILE.exists():
    raise FileNotFoundError(TRAIN_FILE)

if not STREAM_FILE.exists():
    raise FileNotFoundError(STREAM_FILE)

feature_config = json.loads(
    (MODEL_DIR / "feature_columns.json").read_text()
)

numeric_cols = feature_config["numeric"]
categorical_cols = feature_config["categorical"]
all_cols = feature_config["all"]

encoder = joblib.load(MODEL_DIR / "ordinal_encoder.joblib")
medians = joblib.load(MODEL_DIR / "numeric_medians.joblib")

reference_raw = pd.read_csv(TRAIN_FILE)

reference = reference_raw[all_cols].copy()

for col in numeric_cols:
    reference[col] = pd.to_numeric(
        reference[col], errors="coerce"
    )

reference.loc[:, numeric_cols] = (
    reference[numeric_cols].fillna(medians)
)

for col in categorical_cols:
    reference[col] = (
        reference[col]
        .astype("string")
        .fillna("__MISSING__")
        .astype(str)
    )

reference_encoded = pd.DataFrame(
    encoder.transform(reference[categorical_cols]),
    columns=categorical_cols,
    index=reference.index,
)

reference = pd.concat(
    [reference[numeric_cols].astype("float64"),
     reference_encoded],
    axis=1,
)[all_cols]

print("Loading streamed transactions...")

records_by_id = {}

with open(STREAM_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        record = json.loads(line)

        transaction_id = record.get("transaction_id")
        features = record.get("features")

        if (
            transaction_id
            and isinstance(features, dict)
            and features
        ):
            records_by_id[transaction_id] = features

records = list(records_by_id.values())
records = records[-MIN_LIVE_ROWS:]

print("Unique transactions:", len(records))

if len(records) < MIN_LIVE_ROWS:
    print(
        f"WAITING_FOR_LIVE_DATA: "
        f"{len(records)} / {MIN_LIVE_ROWS} transactions"
    )
    raise SystemExit(0)

current = pd.DataFrame(records)

missing = set(all_cols) - set(current.columns)

if missing:
    raise ValueError(
        f"Stream is missing model features: {sorted(missing)}"
    )

current = current[all_cols].copy()

# API already sends encoded model-input values.
for col in all_cols:
    current[col] = pd.to_numeric(
        current[col], errors="coerce"
    )

if current.isna().any().any():
    raise ValueError(
        "Live feature data contains missing or invalid values."
    )

# Use fixed-size samples for a manageable report.
reference_sample = reference.sample(
    n=min(10000, len(reference)),
    random_state=42,
)

current_sample = current.sample(
    n=min(10000, len(current)),
    random_state=42,
)
EXCLUDE = ["TransactionDT", "transaction_day", "card1", "card2", "card3", "card5", "addr1", "addr2"]
reference_sample = reference_sample.drop(columns=EXCLUDE, errors="ignore")
current_sample = current_sample.drop(columns=EXCLUDE, errors="ignore")
print("Reference rows:", len(reference_sample))
print("Live rows:", len(current_sample))
print("Features:", len(all_cols))

print("Generating live drift report...")

report = Report(metrics=[DataDriftPreset()])

snapshot = report.run(
    reference_data=reference_sample,
    current_data=current_sample,
)

snapshot.save_html(
    str(OUTPUT_DIR / "ieee_live_drift_report.html")
)

snapshot.save_json(
    str(OUTPUT_DIR / "ieee_live_drift_snapshot.json")
)

metrics_data = snapshot.dict()

drift_metric = next(
    (
        metric for metric in metrics_data["metrics"]
        if metric["metric_name"].startswith(
            "DriftedColumnsCount"
        )
    ),
    None,
)

if drift_metric is None:
    raise RuntimeError(
        "Could not find drift summary in Evidently report."
    )

drifted_count = int(drift_metric["value"]["count"])
drift_share = float(drift_metric["value"]["share"])

retraining_required = (
    drift_share >= DRIFT_THRESHOLD
)

trigger = {
    "dataset": "IEEE-CIS",
    "monitoring_source": "Kafka streamed transactions",
    "reference_rows": len(reference_sample),
    "live_rows": len(current_sample),
    "total_features": len(reference_sample.columns),
    "drifted_features": drifted_count,
    "drift_share": drift_share,
    "drift_threshold": DRIFT_THRESHOLD,
    "retraining_required": retraining_required,
    "action": (
        "RETRAINING_REQUESTED"
        if retraining_required
        else "NO_ACTION"
    ),
}

with open(
    OUTPUT_DIR / "live_retraining_trigger.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(trigger, f, indent=4)

print("\nLIVE DRIFT RESULT")
print(json.dumps(trigger, indent=4))

