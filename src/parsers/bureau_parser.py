import json
import pandas as pd
from pathlib import Path


class BureauParser:
    def __init__(self, json_path: Path):
        self.json_path = json_path
        self.raw = self._load_json()

    def _load_json(self) -> dict:
        with open(self.json_path, "r") as f:
            return json.load(f)

    def parse_trades(self) -> pd.DataFrame:
        records = []

        base = {
            "loan_app_id": self.raw["LOAN_APP_ID"],
            "name": self.raw["NAME"],
            "dob": self.raw["DOB"],
            "bureau": self.raw["BUREAU"],
            "score_v3": self.raw["SCORE_V3"],
        }

        for trade in self.raw["TRADES"]:
            record = base.copy()
            record.update(
                {
                    "trade_ref_num": trade["TRADE_REF_NUM"],
                    "product_type": trade["PRODUCT_TYPE"],
                    "date_opened": trade["DATE_OPENED"],
                    "current_balance": trade["CURRENT_BALANCE"],
                    "sanctioned_amount": trade["SANCTIONED_AMOUNT"],
                    "dpd": int(trade["DPD"]),
                }
            )
            records.append(record)

        return pd.DataFrame(records)
