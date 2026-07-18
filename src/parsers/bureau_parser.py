"""Parser for raw credit bureau JSON reports."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.utils.io import load_json, parse_dpd

logger = logging.getLogger(__name__)

TRADE_COLUMNS = [
    "loan_app_id",
    "name",
    "dob",
    "bureau",
    "score_v3",
    "trade_ref_num",
    "product_type",
    "date_opened",
    "current_balance",
    "sanctioned_amount",
    "dpd",
]


class BureauParser:
    """Parses one bureau report JSON into a trades DataFrame (1 row per trade)."""

    def __init__(self, json_path: Path):
        self.json_path = Path(json_path)
        self.raw = load_json(self.json_path)

    def parse_trades(self) -> pd.DataFrame:
        base = {
            "loan_app_id": self.raw.get("LOAN_APP_ID"),
            "name": self.raw.get("NAME"),
            "dob": self.raw.get("DOB"),
            "bureau": self.raw.get("BUREAU"),
            "score_v3": self.raw.get("SCORE_V3"),
        }
        if base["loan_app_id"] is None:
            raise ValueError(f"Missing LOAN_APP_ID in {self.json_path}")

        records = []
        for trade in self.raw.get("TRADES", []):
            record = base.copy()
            record.update(
                {
                    "trade_ref_num": trade.get("TRADE_REF_NUM"),
                    "product_type": trade.get("PRODUCT_TYPE"),
                    "date_opened": trade.get("DATE_OPENED"),
                    "current_balance": float(trade.get("CURRENT_BALANCE", 0.0)),
                    "sanctioned_amount": float(trade.get("SANCTIONED_AMOUNT", 0.0)),
                    "dpd": parse_dpd(trade.get("DPD")),
                }
            )
            records.append(record)

        if not records:
            logger.warning("No trades found in %s", self.json_path.name)
            return pd.DataFrame(columns=TRADE_COLUMNS)

        return pd.DataFrame(records, columns=TRADE_COLUMNS)
