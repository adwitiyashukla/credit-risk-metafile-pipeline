"""Tests for bureau and banking parsers."""

import json

import pytest

from src.parsers.banking_parser import BankingParser
from src.parsers.bureau_parser import BureauParser
from src.utils.io import parse_dpd

BUREAU_SAMPLE = {
    "LOAN_APP_ID": "APP000123",
    "NAME": "Test User",
    "DOB": "1990-01-01",
    "BUREAU": "CIBIL",
    "SCORE_V3": 720,
    "TRADES": [
        {
            "TRADE_REF_NUM": "111",
            "PRODUCT_TYPE": "PL",
            "DATE_OPENED": "2020-05-01",
            "CURRENT_BALANCE": 1000.0,
            "SANCTIONED_AMOUNT": 50000.0,
            "DPD": "015",
        },
        {
            "TRADE_REF_NUM": "222",
            "PRODUCT_TYPE": "CC",
            "DATE_OPENED": "2021-06-01",
            "CURRENT_BALANCE": 0.0,
            "SANCTIONED_AMOUNT": 20000.0,
            "DPD": "XXX",
        },
    ],
}

BANKING_SAMPLE = {
    "LOAN_APP_ID": "APP000123",
    "BANK_NAME": "HDFC",
    "ACCT_NUM": "123456789012",
    "TRANSACTIONS": [
        {
            "TXN_DATE": "2024-01-05",
            "TXN_AMOUNT": 50000.0,
            "TXN_TYPE": "CR",
            "TXN_DESC": "Salary",
            "BAL_AFTER": 55000.0,
        },
        {
            "TXN_DATE": "2024-01-10",
            "TXN_AMOUNT": 8000.0,
            "TXN_TYPE": "DR",
            "TXN_DESC": "Loan EMI",
            "BAL_AFTER": 47000.0,
        },
    ],
}


def test_bureau_parser_one_row_per_trade(tmp_path):
    path = tmp_path / "APP000123.json"
    path.write_text(json.dumps(BUREAU_SAMPLE))

    df = BureauParser(path).parse_trades()

    assert len(df) == 2
    assert df["loan_app_id"].unique().tolist() == ["APP000123"]
    assert df["score_v3"].iloc[0] == 720


def test_bureau_parser_handles_non_numeric_dpd(tmp_path):
    path = tmp_path / "APP000123.json"
    path.write_text(json.dumps(BUREAU_SAMPLE))

    df = BureauParser(path).parse_trades()

    assert df["dpd"].tolist() == [15, 0]  # "015" -> 15, "XXX" -> 0


def test_bureau_parser_empty_trades(tmp_path):
    sample = {**BUREAU_SAMPLE, "TRADES": []}
    path = tmp_path / "APP000123.json"
    path.write_text(json.dumps(sample))

    df = BureauParser(path).parse_trades()

    assert df.empty


def test_bureau_parser_missing_app_id(tmp_path):
    sample = {k: v for k, v in BUREAU_SAMPLE.items() if k != "LOAN_APP_ID"}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(sample))

    with pytest.raises(ValueError, match="LOAN_APP_ID"):
        BureauParser(path).parse_trades()


def test_parser_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        BureauParser(tmp_path / "nope.json")


def test_parser_corrupt_json(tmp_path):
    path = tmp_path / "corrupt.json"
    path.write_text("{not valid json")

    with pytest.raises(ValueError, match="Corrupt JSON"):
        BankingParser(path)


def test_banking_parser_transactions(tmp_path):
    path = tmp_path / "APP000123.json"
    path.write_text(json.dumps(BANKING_SAMPLE))

    df = BankingParser(path).parse_transactions()

    assert len(df) == 2
    assert df["txn_amount"].sum() == 58000.0
    assert str(df["txn_date"].dtype).startswith("datetime64")


@pytest.mark.parametrize(
    "value,expected",
    [("000", 0), ("015", 15), ("180", 180), ("XXX", 0), ("STD", 0), (None, 0), (30, 30)],
)
def test_parse_dpd(value, expected):
    assert parse_dpd(value) == expected
