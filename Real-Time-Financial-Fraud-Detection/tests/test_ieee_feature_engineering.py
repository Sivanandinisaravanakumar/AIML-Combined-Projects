import pandas as pd

from src.ieee_feature_engineering import create_features


def sample_data():
    return pd.DataFrame({
        "TransactionID": [1, 2, 3, 4, 5],
        "TransactionDT": [0, 1800, 7200, 3600, 4000],
        "TransactionAmt": [100.0, 200.0, 300.0, 50.0, 75.0],
        "card1": ["100", "100", "100", None, None],
        "card2": ["20"] * 5,
        "card3": ["150"] * 5,
        "card5": ["226"] * 5,
        "card6": ["credit"] * 5,
        "DeviceInfo": ["A", "B", "B", None, None],
        "DeviceType": ["desktop"] * 5,
        "addr1": ["10", "20", "20", None, None],
        "addr2": ["87"] * 5,
        "isFraud": [0, 0, 1, 0, 0],
        "dataset_type": ["train"] * 5,
    })


def test_transaction_hour_and_day():
    result = create_features(sample_data())
    rows = result.set_index("TransactionID")

    assert rows.loc[1, "transaction_hour"] == 0
    assert rows.loc[3, "transaction_hour"] == 2
    assert rows.loc[1, "transaction_day"] == 0


def test_historical_transaction_counts():
    result = create_features(sample_data())
    rows = result.set_index("TransactionID")

    assert rows.loc[1, "card_txn_count_1h"] == 0
    assert rows.loc[2, "card_txn_count_1h"] == 1
    assert rows.loc[3, "card_txn_count_1h"] == 0

    assert rows.loc[1, "card_txn_count_24h"] == 0
    assert rows.loc[3, "card_txn_count_24h"] == 2


def test_historical_average_excludes_current_transaction():
    result = create_features(sample_data())
    rows = result.set_index("TransactionID")

    assert pd.isna(rows.loc[1, "card_avg_amount_30d"])
    assert rows.loc[2, "card_avg_amount_30d"] == 100.0
    assert rows.loc[3, "card_avg_amount_30d"] == 150.0


def test_amount_vs_historical_average():
    result = create_features(sample_data())
    rows = result.set_index("TransactionID")

    assert pd.isna(rows.loc[1, "amount_vs_card_average"])
    assert rows.loc[2, "amount_vs_card_average"] == 2.0
    assert rows.loc[3, "amount_vs_card_average"] == 2.0


def test_device_and_address_change_flags():
    result = create_features(sample_data())
    rows = result.set_index("TransactionID")

    assert rows.loc[1, "device_changed"] == 0
    assert rows.loc[2, "device_changed"] == 1
    assert rows.loc[3, "device_changed"] == 0

    assert rows.loc[1, "address_changed"] == 0
    assert rows.loc[2, "address_changed"] == 1
    assert rows.loc[3, "address_changed"] == 0


def test_missing_card_identifiers_are_not_grouped_together():
    result = create_features(sample_data())
    rows = result.set_index("TransactionID")

    assert rows.loc[4, "card_txn_count_24h"] == 0
    assert rows.loc[5, "card_txn_count_24h"] == 0
    assert pd.isna(rows.loc[4, "card_avg_amount_30d"])
    assert pd.isna(rows.loc[5, "card_avg_amount_30d"])


def test_helper_columns_are_removed():
    result = create_features(sample_data())

    assert "card_key" not in result.columns
    assert "transaction_time" not in result.columns
