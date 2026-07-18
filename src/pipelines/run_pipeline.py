"""End-to-end pipeline: raw JSON -> parsed tables -> applicant-level metafile.

Parses every applicant's bureau + banking files, builds features in a single
vectorised pass, joins observed default labels (if present), and writes the
result to ``data/processed/metafile.parquet``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from rich.progress import track

from src.features.feature_engineering import build_metafile
from src.parsers.banking_parser import BankingParser
from src.parsers.bureau_parser import BureauParser
from src.utils.io import setup_logging

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
BUREAU_DIR = ROOT / "data" / "raw" / "bureau"
BANKING_DIR = ROOT / "data" / "raw" / "banking"
LABELS_PATH = ROOT / "data" / "raw" / "labels.csv"
OUT_DIR = ROOT / "data" / "processed"
META_PATH = OUT_DIR / "metafile.parquet"


def main(limit: int | None = None) -> pd.DataFrame:
    """Build the metafile and save it as Parquet. Returns the final DataFrame."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bureau_files = sorted(BUREAU_DIR.glob("APP*.json"))
    if not bureau_files:
        raise FileNotFoundError(
            f"No raw bureau files found in {BUREAU_DIR}. "
            "Run the generator first (python -m src.main generate)."
        )
    if limit:
        bureau_files = bureau_files[:limit]

    trades_frames: list[pd.DataFrame] = []
    txn_frames: list[pd.DataFrame] = []
    skipped = 0

    for bfile in track(bureau_files, description="Parsing raw files..."):
        app_id = bfile.stem
        bank_file = BANKING_DIR / f"{app_id}.json"

        try:
            trades = BureauParser(bfile).parse_trades()
            txns = BankingParser(bank_file).parse_transactions()
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Skipping %s: %s", app_id, exc)
            skipped += 1
            continue

        if trades.empty or txns.empty:
            logger.warning("Skipping %s: empty trades or transactions", app_id)
            skipped += 1
            continue

        trades_frames.append(trades)
        txn_frames.append(txns)

    if not trades_frames:
        raise RuntimeError("No applicants could be parsed; nothing to build.")

    all_trades = pd.concat(trades_frames, ignore_index=True)
    all_txns = pd.concat(txn_frames, ignore_index=True)

    # Single vectorised aggregation across all applicants at once.
    metafile = build_metafile(all_trades, all_txns)

    # Join observed default labels when available (used to train the scorer).
    if LABELS_PATH.exists():
        labels = pd.read_csv(LABELS_PATH)
        metafile = metafile.merge(labels, on="loan_app_id", how="left")
        matched = metafile["default_flag"].notna().sum()
        logger.info("Joined default labels for %d/%d applicants", matched, len(metafile))
    else:
        logger.warning("No labels file at %s; metafile will have no default_flag", LABELS_PATH)

    metafile.to_parquet(META_PATH, index=False)

    logger.info("Metafile saved to %s", META_PATH)
    logger.info(
        "Applicants: %d | Columns: %d | Skipped: %d",
        metafile.shape[0],
        metafile.shape[1],
        skipped,
    )
    return metafile


if __name__ == "__main__":
    setup_logging()
    main()
