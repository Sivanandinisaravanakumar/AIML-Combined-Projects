
from pathlib import Path
import json
import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from kafka import KafkaConsumer


# ==================================================
# PATHS
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    ROOT / "feature_repo" / "feature_repo"
    / "data" / "fraud_features.parquet"
)

MODEL_DIR = ROOT / "models" / "ieee_cis"
REPORTS_DIR = ROOT / "reports"

CASE_FILE = REPORTS_DIR / "analyst_cases.csv"
DRIFT_FILE = REPORTS_DIR / "retraining_trigger.json"
METRICS_FILE = MODEL_DIR / "metrics.json"
ALERT_FILE = REPORTS_DIR / "kafka_alerts.csv"

STREAM_FILE = (
    ROOT / "data" / "ieee_cis" / "incoming"
    / "streamed_transactions.jsonl"
)

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# STREAMLIT CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Financial Fraud Analyst Dashboard",
    page_icon="ðŸ›¡ï¸",
    layout="wide",
)

st.title("Financial Fraud Detection")
st.caption(
    "AIML-01 | IEEE-CIS | Fraud Analyst Monitoring Dashboard"
)


# ==================================================
# HELPER FUNCTIONS
# ==================================================

@st.cache_data
def load_feature_data():
    """Load the available Feast transaction feature dataset."""

    if not FEATURE_FILE.exists():
        return pd.DataFrame()

    return pd.read_parquet(FEATURE_FILE)


def load_json(path):
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return {}


def safe_number(value):
    """Convert values to numbers safely."""

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None

        return float(value)

    except (TypeError, ValueError):
        return None


def create_case_file():
    """Create an initial analyst queue from labelled fraud records."""

    df = load_feature_data()

    if df.empty:
        st.error(
            "Transaction feature dataset not found. "
            "Please check the fraud_features.parquet file."
        )
        st.stop()

    required_columns = [
        "TransactionID",
        "card_key",
        "event_timestamp",
        "isFraud",
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        st.error(f"Missing dataset columns: {missing}")
        st.stop()

    fraud_cases = df[
        pd.to_numeric(df["isFraud"], errors="coerce") == 1
    ][required_columns].head(100).copy()

    fraud_cases["case_status"] = "Open"
    fraud_cases["analyst_feedback"] = "Pending"

    fraud_cases.to_csv(CASE_FILE, index=False)

    return fraud_cases


def load_cases():
    """Load analyst cases or create the initial case queue."""

    if not CASE_FILE.exists():
        return create_case_file()

    try:
        cases = pd.read_csv(CASE_FILE)

    except (
        OSError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ):
        return create_case_file()

    expected_columns = [
        "TransactionID",
        "card_key",
        "event_timestamp",
        "isFraud",
        "case_status",
        "analyst_feedback",
    ]

    if not all(
        column in cases.columns
        for column in expected_columns
    ):
        return create_case_file()

    cases["case_status"] = (
        cases["case_status"].fillna("Open")
    )

    cases["analyst_feedback"] = (
        cases["analyst_feedback"].fillna("Pending")
    )

    return cases[expected_columns]


def save_cases(edited_cases):
    """Save analyst changes."""

    temp_file = CASE_FILE.with_suffix(".tmp")

    edited_cases.to_csv(temp_file, index=False)
    temp_file.replace(CASE_FILE)


def normalize_alert(transaction, source="Unknown"):
    """Convert incoming alert records to a common format."""

    prediction = safe_number(
        transaction.get(
            "Prediction",
            transaction.get("prediction"),
        )
    )

    if prediction is not None:
        prediction = int(prediction)

    label = transaction.get(
        "Label",
        transaction.get("label"),
    )

    if label is None:
        if prediction == 1:
            label = "FRAUD"
        elif prediction == 0:
            label = "NORMAL"
        else:
            label = "UNKNOWN"

    model = transaction.get(
        "Model",
        transaction.get("model", source),
    )

    received_at = (
        transaction.get("Received At")
        or transaction.get("observed_at_utc")
    )

    return {
        "Received At": received_at,
        "Transaction ID": (
            transaction.get("Transaction ID")
            or transaction.get("transaction_id")
        ),
        "Fraud Probability": safe_number(
            transaction.get(
                "Fraud Probability",
                transaction.get("fraud_probability"),
            )
        ),
        "Threshold": safe_number(
            transaction.get(
                "Threshold",
                transaction.get("threshold"),
            )
        ),
        "Prediction": prediction,
        "Label": label,
        "Model": model,
        "Actual Fraud": transaction.get(
            "Actual Fraud",
            transaction.get("actual_isFraud"),
        ),
    }


# ==================================================
# AIVEN KAFKA CONFIGURATION
# ==================================================

@st.cache_resource
def get_dashboard_consumer():
    """
    Connect Streamlit dashboard to Aiven Kafka
    using SASL_SSL and SCRAM-SHA-256.
    """

    kafka_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS")

    if not kafka_servers:
        return None

    required_variables = [
        "KAFKA_SASL_USERNAME",
        "KAFKA_SASL_PASSWORD",
        "KAFKA_SECURITY_PROTOCOL",
        "KAFKA_SASL_MECHANISM",
    ]

    missing_variables = [
        name for name in required_variables
        if not os.getenv(name)
    ]

    if missing_variables:
        raise RuntimeError(
            "Missing Kafka environment variables: "
            + ", ".join(missing_variables)
        )

    security_protocol = os.getenv(
        "KAFKA_SECURITY_PROTOCOL",
        "SASL_SSL",
    )

    sasl_mechanism = os.getenv(
        "KAFKA_SASL_MECHANISM",
        "SCRAM-SHA-256",
    )

    kafka_config = {
        "bootstrap_servers": [
            server.strip()
            for server in kafka_servers.split(",")
            if server.strip()
        ],
        "security_protocol": security_protocol,
        "sasl_mechanism": sasl_mechanism,
        "sasl_plain_username": os.getenv(
            "KAFKA_SASL_USERNAME"
        ),
        "sasl_plain_password": os.getenv(
            "KAFKA_SASL_PASSWORD"
        ),
        "group_id": "fraud-transactions-consumer",
        "auto_offset_reset": "earliest",
        "enable_auto_commit": True,
        "value_deserializer": lambda value: json.loads(
            value.decode("utf-8")
        ),
        "request_timeout_ms": 15000,
    }

    # Configure the Aiven CA certificate when provided.
    ca_file = os.getenv("KAFKA_SSL_CAFILE")

    if ca_file:
        if not Path(ca_file).is_file():
            raise FileNotFoundError(
                f"Kafka CA certificate not found: {ca_file}"
            )

        kafka_config["ssl_cafile"] = ca_file

    if security_protocol != "SASL_SSL":
        raise ValueError(
            "Aiven Kafka requires SASL_SSL for this configuration."
        )

    if sasl_mechanism != "SCRAM-SHA-256":
        raise ValueError(
            "Expected SASL mechanism SCRAM-SHA-256."
        )

    consumer = KafkaConsumer(
        "fraud-transactions",
        **kafka_config,
    )

    return consumer


# ==================================================
# READ LIVE ALERTS
# ==================================================

def read_live_alerts():
    """Read historical records and new Kafka messages."""

    all_alerts = []
    new_alerts = []

    # ----------------------------------------------
    # 1. Read JSONL transaction history
    # ----------------------------------------------

    if STREAM_FILE.exists():

        try:
            with STREAM_FILE.open(
                "r",
                encoding="utf-8",
            ) as file:

                for line in file:

                    if not line.strip():
                        continue

                    try:
                        transaction = json.loads(line)

                        all_alerts.append(
                            normalize_alert(
                                transaction,
                                source="Historical Stream",
                            )
                        )

                    except (json.JSONDecodeError, TypeError):
                        continue

        except OSError as error:
            st.warning(
                f"Could not read historical stream: {error}"
            )

    # ----------------------------------------------
    # 2. Read previously saved alert CSV
    # ----------------------------------------------

    if ALERT_FILE.exists():

        try:
            csv_df = pd.read_csv(ALERT_FILE)

            for transaction in csv_df.to_dict("records"):

                all_alerts.append(
                    normalize_alert(
                        transaction,
                        source="Saved Alert History",
                    )
                )

        except (
            OSError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
        ):
            pass

    # ----------------------------------------------
    # 3. Read new Kafka messages
    # ----------------------------------------------

    try:

        consumer = get_dashboard_consumer()

        if consumer is not None:

            messages = consumer.poll(
                timeout_ms=500,
                max_records=50,
            )

            for records in messages.values():

                for message in records:

                    alert = normalize_alert(
                        message.value,
                        source="Kafka",
                    )

                    if not alert["Received At"]:
                        alert["Received At"] = (
                            datetime.now(timezone.utc).isoformat()
                        )

                    new_alerts.append(alert)

        else:
            st.info(
                "Kafka is not configured. "
                "Historical transaction records are shown when available."
            )

    except Exception as error:
        st.warning(
            f"Kafka is currently unavailable: {error}"
        )

    # ----------------------------------------------
    # 4. Save new Kafka messages
    # ----------------------------------------------

    if new_alerts:

        csv_records = [
            {
                "Received At": alert["Received At"],
                "Transaction ID": alert["Transaction ID"],
                "Fraud Probability": alert["Fraud Probability"],
                "Threshold": alert["Threshold"],
                "Prediction": alert["Prediction"],
                "Label": alert["Label"],
                "Model": alert["Model"],
                "Actual Fraud": alert["Actual Fraud"],
            }
            for alert in new_alerts
        ]

        new_df = pd.DataFrame(csv_records)

        if ALERT_FILE.exists():

            new_df.to_csv(
                ALERT_FILE,
                mode="a",
                header=False,
                index=False,
            )

        else:

            new_df.to_csv(
                ALERT_FILE,
                index=False,
            )

        all_alerts.extend(new_alerts)

    # ----------------------------------------------
    # 5. Remove duplicate transaction IDs
    # ----------------------------------------------

    combined_df = pd.DataFrame(all_alerts)

    if not combined_df.empty:

        has_id = combined_df["Transaction ID"].notna()

        identified = combined_df[has_id].drop_duplicates(
            subset=["Transaction ID"],
            keep="last",
        )

        unidentified = combined_df[~has_id]

        combined_df = pd.concat(
            [identified, unidentified],
            ignore_index=True,
        )

    st.session_state.live_alerts = (
        combined_df.to_dict("records")
        if not combined_df.empty
        else []
    )


# ==================================================
# LOAD MONITORING DATA
# ==================================================

drift = load_json(DRIFT_FILE)
model_metrics = load_json(METRICS_FILE)
cases = load_cases()


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("Dashboard Menu")
st.sidebar.write("Dataset: IEEE-CIS")

page = st.sidebar.radio(
    "Select View",
    [
        "Overview",
        "Analyst Case Queue",
        "Data Drift",
        "Live Kafka Alerts",
        "Card Behaviour Analysis",
    ],
)


# ==================================================
# OVERVIEW
# ==================================================

if page == "Overview":

    st.header("Monitoring Overview")

    drift_count = drift.get("drifted_features", 0)
    total_features = drift.get("total_features", 0)
    drift_share = drift.get("drift_share", 0)

    retraining_required = drift.get(
        "retraining_required",
        False,
    )

    open_count = int(
        (cases["case_status"] == "Open").sum()
    )

    closed_count = int(
        (cases["case_status"] == "Closed").sum()
    )

    pr_auc = model_metrics.get("PR_AUC")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Drifted Features",
        f"{drift_count}/{total_features}",
    )

    col2.metric(
        "Drift Share",
        f"{drift_share * 100:.1f}%"
        if isinstance(drift_share, (int, float))
        else "N/A",
    )

    col3.metric(
        "Open Cases",
        open_count,
    )

    col4.metric(
        "XGBoost PR-AUC",
        f"{pr_auc:.4f}"
        if isinstance(pr_auc, (int, float))
        else "N/A",
    )

    st.divider()

    st.subheader("System Status")

    if retraining_required:
        st.error(
            "Drift threshold reached. "
            "Model retraining review is required."
        )
    else:
        st.success(
            "Drift is below the configured threshold. "
            "No retraining request."
        )

    st.info(
        f"Closed cases: {closed_count} | "
        f"Total analyst cases: {len(cases)}"
    )

    st.warning(
        "The analyst queue contains historical labelled "
        "IEEE-CIS fraud examples. It is not a live Kafka case queue."
    )


# ==================================================
# ANALYST CASE QUEUE
# ==================================================

elif page == "Analyst Case Queue":

    st.header("Analyst Case Queue")

    st.write(
        "Review historical labelled fraud transactions "
        "and update their investigation status."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Cases",
        len(cases),
    )

    col2.metric(
        "Open Cases",
        int((cases["case_status"] == "Open").sum()),
    )

    col3.metric(
        "Closed Cases",
        int((cases["case_status"] == "Closed").sum()),
    )

    st.divider()

    with st.form("analyst_case_form"):

        edited_cases = st.data_editor(
            cases,
            hide_index=True,
            use_container_width=True,
            key="analyst_cases_editor",
            disabled=[
                "TransactionID",
                "card_key",
                "event_timestamp",
                "isFraud",
            ],
            column_config={
                "TransactionID": st.column_config.NumberColumn(
                    "Transaction ID",
                ),
                "card_key": st.column_config.TextColumn(
                    "Card Key",
                ),
                "event_timestamp": st.column_config.TextColumn(
                    "Event Timestamp",
                ),
                "isFraud": st.column_config.NumberColumn(
                    "Actual Fraud Label",
                ),
                "case_status": st.column_config.SelectboxColumn(
                    "Case Status",
                    options=[
                        "Open",
                        "Investigating",
                        "Closed",
                    ],
                    required=True,
                ),
                "analyst_feedback": st.column_config.SelectboxColumn(
                    "Analyst Feedback",
                    options=[
                        "Pending",
                        "Confirmed Fraud",
                        "False Positive",
                        "Needs Review",
                    ],
                    required=True,
                ),
            },
        )

        save_clicked = st.form_submit_button(
            "Save Analyst Updates",
            type="primary",
            use_container_width=True,
        )

    if save_clicked:

        try:
            save_cases(edited_cases)

            st.success(
                "Analyst updates saved successfully!"
            )

            st.rerun()

        except (OSError, PermissionError, ValueError) as error:
            st.error(
                f"Could not save analyst updates: {error}"
            )

    st.caption(
        "Updates are saved to reports/analyst_cases.csv. "
        "On Streamlit Cloud, local file changes may not persist "
        "after an app restart."
    )


# ==================================================
# DATA DRIFT
# ==================================================

elif page == "Data Drift":

    st.header("Evidently Data Drift Monitoring")

    if not drift:

        st.warning(
            "Drift report data not found. "
            "Run the drift monitoring process to generate it."
        )

    else:

        drift_count = drift.get("drifted_features", 0)
        total_features = drift.get("total_features", 0)
        drift_share = drift.get("drift_share", 0)
        threshold = drift.get("drift_threshold", 0.5)

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Features",
            total_features,
        )

        col2.metric(
            "Drifted Features",
            drift_count,
        )

        col3.metric(
            "Drift Share",
            f"{drift_share * 100:.1f}%"
            if isinstance(drift_share, (int, float))
            else "N/A",
        )

        st.progress(
            min(max(float(drift_share), 0.0), 1.0),
            text="Share of drifted features",
        )

        st.write(
            f"Configured retraining threshold: "
            f"{threshold * 100:.1f}%"
        )

        if drift.get("retraining_required", False):

            st.error(
                "Retraining review required: "
                "drift threshold reached."
            )

        else:

            st.success(
                "No retraining request: "
                "drift is below threshold."
            )

        st.subheader("Drift Report Details")
        st.json(drift)

    report_file = (
        REPORTS_DIR / "ieee_data_drift_report.html"
    )

    if report_file.exists():

        st.caption(
            "Detailed Evidently HTML report is available "
            "in reports/ieee_data_drift_report.html."
        )


# ==================================================
# LIVE KAFKA ALERTS
# ==================================================

elif page == "Live Kafka Alerts":

    st.header("Live Kafka Fraud Alerts")

    st.caption(
        "Historical JSONL and CSV records, together with "
        "new Kafka messages when a Kafka service is configured."
    )

    @st.fragment(run_every="3s")
    def display_live_alerts():

        try:

            read_live_alerts()

            alerts = st.session_state.live_alerts

            fraud_count = sum(
                1 for alert in alerts
                if safe_number(alert.get("Prediction")) == 1
            )

            normal_count = sum(
                1 for alert in alerts
                if safe_number(alert.get("Prediction")) == 0
            )

            unknown_count = (
                len(alerts) - fraud_count - normal_count
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Total Transactions",
                len(alerts),
            )

            col2.metric(
                "Predicted Fraud",
                fraud_count,
            )

            col3.metric(
                "Predicted Normal",
                normal_count,
            )

            col4.metric(
                "Unknown Predictions",
                unknown_count,
            )

            st.divider()

            if alerts:

                alert_df = pd.DataFrame(alerts)

                alert_df["Prediction"] = pd.to_numeric(
                    alert_df["Prediction"],
                    errors="coerce",
                ).astype("Int64")

                alert_df["Actual Fraud"] = pd.to_numeric(
                    alert_df["Actual Fraud"],
                    errors="coerce",
                ).astype("Int64")

                st.subheader("Transaction Records")

                st.dataframe(
                    alert_df.iloc[::-1],
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    "Predicted Fraud represents model predictions. "
                    "Actual Fraud is ground truth and may not be "
                    "available for live transactions."
                )

            else:

                st.info(
                    "No transaction alert records found yet."
                )

        except Exception as error:

            st.error(
                f"Could not load transaction alerts: {error}"
            )

    display_live_alerts()


# ==================================================
# CARD BEHAVIOUR ANALYSIS
# ==================================================

elif page == "Card Behaviour Analysis":

    st.header("Card Behaviour Analysis")

    st.caption(
        "Transaction frequency, amount behaviour, "
        "device changes and address changes."
    )

    df = load_feature_data()

    if df.empty:

        st.error(
            "Feature dataset not found. "
            "Please check fraud_features.parquet."
        )
        st.stop()

    required_columns = [
        "TransactionID",
        "card_key",
        "isFraud",
        "card_txn_count_1h",
        "card_txn_count_24h",
        "card_avg_amount_30d",
        "amount_vs_card_average",
        "device_changed",
        "address_changed",
        "device_missing",
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:

        st.error(
            f"Required feature columns are missing: {missing}"
        )
        st.stop()

    fraud_df = df[
        pd.to_numeric(df["isFraud"], errors="coerce") == 1
    ]

    normal_df = df[
        pd.to_numeric(df["isFraud"], errors="coerce") == 0
    ]

    fraud_count = len(fraud_df)
    normal_count = len(normal_df)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Transactions",
        f"{len(df):,}",
    )

    col2.metric(
        "Fraud Transactions",
        f"{fraud_count:,}",
    )

    col3.metric(
        "Normal Transactions",
        f"{normal_count:,}",
    )

    st.divider()

    # ----------------------------------------------
    # Behaviour indicators
    # ----------------------------------------------

    st.subheader("Transaction Behaviour Indicators")

    indicator_columns = [
        "card_txn_count_1h",
        "card_txn_count_24h",
        "card_avg_amount_30d",
        "amount_vs_card_average",
        "device_changed",
        "address_changed",
        "device_missing",
    ]

    indicator_summary = []

    for column in indicator_columns:

        fraud_values = pd.to_numeric(
            fraud_df[column],
            errors="coerce",
        )

        normal_values = pd.to_numeric(
            normal_df[column],
            errors="coerce",
        )

        indicator_summary.append({
            "Indicator": column,
            "Fraud Average": fraud_values.mean(),
            "Normal Average": normal_values.mean(),
        })

    summary_df = pd.DataFrame(indicator_summary)

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # ----------------------------------------------
    # Card-level summary
    # ----------------------------------------------

    st.subheader("Card-Level Fraud Summary")

    card_summary = (
        df.groupby("card_key")
        .agg(
            transactions=("TransactionID", "count"),
            fraud_transactions=("isFraud", "sum"),
            avg_txn_1h=("card_txn_count_1h", "mean"),
            avg_txn_24h=("card_txn_count_24h", "mean"),
            device_changes=("device_changed", "sum"),
            address_changes=("address_changed", "sum"),
        )
        .reset_index()
    )

    card_summary["fraud_rate"] = (
        card_summary["fraud_transactions"]
        / card_summary["transactions"]
    )

    card_summary = card_summary.sort_values(
        ["fraud_transactions", "transactions"],
        ascending=False,
    )

    st.dataframe(
        card_summary.head(20),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # ----------------------------------------------
    # Fraud vs normal comparison
    # ----------------------------------------------

    st.subheader("Fraud vs Normal Behaviour")

    comparison_columns = [
        "card_txn_count_1h",
        "card_txn_count_24h",
        "device_changed",
        "address_changed",
        "device_missing",
    ]

    comparison_data = []

    for column in comparison_columns:

        comparison_data.append({
            "Indicator": column,
            "Fraud": pd.to_numeric(
                fraud_df[column],
                errors="coerce",
            ).mean(),
            "Normal": pd.to_numeric(
                normal_df[column],
                errors="coerce",
            ).mean(),
        })

    comparison_df = pd.DataFrame(comparison_data)

    st.bar_chart(
        comparison_df.set_index("Indicator")
    )

    st.info(
        "These are historical labelled transaction features. "
        "Behaviour indicators help analysts identify unusual "
        "patterns; they do not independently prove fraud."
    )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "AIML-01 Financial Fraud Detection | "
    "Streamlit Analyst Dashboard"
)
