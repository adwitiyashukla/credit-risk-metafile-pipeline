"""Tests for feature engineering and risk banding."""

import pandas as pd
import pytest

from src.features.feature_engineering import (
    assign_risk_band,
    banking_features,
    build_metafile,
    bureau_features,
)


@pytest.fixture
def trades_df():
    return pd.DataFrame(
        {
            "loan_app_id": ["A1", "A1", "A2"],
            "name": ["X", "X", "Y"],
            "dob": ["1990-01-01"] * 3,
            "bureau": ["CIBIL", "CIBIL", "CRIF"],
            "score_v3": [700, 700, 480],
            "trade_ref_num": ["t1", "t2", "t3"],
            "product_type": ["PL", "CC", "PL"],
            "date_opened": ["2020-01-01"] * 3,
            "current_balance": [1000.0, 2000.0, 500.0],
            "sanctioned_amount": [10000.0, 20000.0, 5000.0],
            "dpd": [0, 45, 95],
        }
    )


@pytest.fixture
def txns_df():
    return pd.DataFrame(
        {
            "loan_app_id": ["A1", "A1", "A1", "A2"],
            "bank_name": ["HDFC"] * 3 + ["SBI"],
            "acct_num": ["111"] * 3 + ["222"],
            "txn_date": pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-10", "2024-01-03"]),
            "txn_amount": [50000.0, 8000.0, 2000.0, 1000.0],
            "txn_type": ["CR", "DR", "DR", "DR"],
            "txn_desc": ["Salary", "Loan EMI", "Cash Withdrawal", "UPI Transfer"],
            "bal_after": [55000.0, 47000.0, 45000.0, -2000.0],
        }
    )


def test_bureau_features_aggregates(trades_df):
    out = bureau_features(trades_df).set_index("loan_app_id")

    assert out.loc["A1", "num_trades"] == 2
    assert out.loc["A1", "max_dpd"] == 45
    assert out.loc["A1", "total_sanctioned_amount"] == 30000.0
    assert out.loc["A1", "dpd_30_plus"] == 1
    assert out.loc["A1", "dpd_60_plus"] == 0
    assert out.loc["A2", "dpd_90_plus"] == 1


def test_banking_features_aggregates(txns_df):
    out = banking_features(txns_df).set_index("loan_app_id")

    assert out.loc["A1", "txn_count"] == 3
    assert out.loc["A1", "total_credits"] == 50000.0
    assert out.loc["A1", "total_debits"] == 10000.0
    assert out.loc["A1", "salary_credits_count"] == 1
    assert out.loc["A1", "emi_txn_count"] == 1
    assert out.loc["A1", "cash_withdrawal_count"] == 1
    assert out.loc["A2", "min_balance"] == -2000.0


@pytest.mark.parametrize(
    "score,max_dpd,expected",
    [
        (800, 0, 1),
        (700, 0, 2),
        (600, 0, 3),
        (450, 0, 4),
        (800, 90, 5),  # deep delinquency overrides good score
        (450, 120, 5),
    ],
)
def test_assign_risk_band(score, max_dpd, expected):
    assert assign_risk_band(score, max_dpd) == expected


def test_build_metafile_one_row_per_applicant(trades_df, txns_df):
    meta = build_metafile(trades_df, txns_df)

    assert len(meta) == 2
    assert meta["loan_app_id"].is_unique
    assert "risk_band" in meta.columns
    # A2: score 480 with max_dpd 95 -> band 5
    assert meta.set_index("loan_app_id").loc["A2", "risk_band"] == 5
