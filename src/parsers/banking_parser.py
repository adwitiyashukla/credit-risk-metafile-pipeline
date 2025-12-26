import json
import pandas as pd
from pathlib import Path


class BankingParser:
    def __init__(self, json_path: Path):
        self.json_path = json_path
        self.raw = self._load_json()

    def _load_json(self) -> dict:
        with open(self.json_path, "r") as f:
            return json.load(f)

    def parse_transactions(self) -> pd.DataFrame:
        records = []

        base = {
            "loan_app_id": self.raw["LOAN_APP_ID"],
            "bank_name": self.raw["BANK_NAME"],
            "acct_num": self.raw["ACCT_NUM"],
        }

        for txn in self.raw["TRANSACTIONS"]:
            record = base.copy()
            record.update(
                {
                    "txn_date": txn["TXN_DATE"],
                    "txn_amount": float(txn["TXN_AMOUNT"]),
                    "txn_type": txn["TXN_TYPE"],
                    "txn_desc": txn["TXN_DESC"],
                    "bal_after": float(txn["BAL_AFTER"]),
                }
            )
            records.append(record)

        df = pd.DataFrame(records)
        df["txn_date"] = pd.to_datetime(df["txn_date"])
        return df
