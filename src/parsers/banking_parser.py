"""Parser for raw banking transaction JSON statements."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.utils.io import load_json

logger = logging.getLogger(__name__)

TXN_COLUMNS = [
    "loan_app_id",
    "bank_name",
    "acct_num",
    "txn_date",
    "txn_amount",
    "txn_type",
    "txn_desc",
    "bal_after",
]


class BankingParser:
    """Parses one bank statement JSON into a transactions DataFrame (1 row per txn)."""

    def __init__(self, json_path: Path):
        self.json_path = Path(json_path)
        self.raw = load_json(self.json_path)

    def parse_transactions(self) -> pd.DataFrame:
        base = {
            "loan_app_id": self.raw.get("LOAN_APP_ID"),
            "bank_name": self.raw.get("BANK_NAME"),
            "acct_num": self.raw.get("ACCT_NUM"),
        }
        if base["loan_app_id"] is None:
            raise ValueError(f"Missing LOAN_APP_ID in {self.json_path}")

        records = []
        for txn in self.raw.get("TRANSACTIONS", []):
            record = base.copy()
            record.update(
                {
                    "txn_date": txn.get("TXN_DATE"),
                    "txn_amount": float(txn.get("TXN_AMOUNT", 0.0)),
                    "txn_type": txn.get("TXN_TYPE"),
                    "txn_desc": txn.get("TXN_DESC"),
                    "bal_after": float(txn.get("BAL_AFTER", 0.0)),
                }
            )
            records.append(record)

        if not records:
            logger.warning("No transactions found in %s", self.json_path.name)
            return pd.DataFrame(columns=TXN_COLUMNS)

        df = pd.DataFrame(records, columns=TXN_COLUMNS)
        df["txn_date"] = pd.to_datetime(df["txn_date"], errors="coerce")
        return df
