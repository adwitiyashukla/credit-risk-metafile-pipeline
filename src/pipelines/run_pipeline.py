from pathlib import Path
import pandas as pd
from rich.progress import track

from src.parsers.bureau_parser import BureauParser
from src.parsers.banking_parser import BankingParser
from src.features.feature_engineering import build_metafile


ROOT = Path(__file__).resolve().parents[2]
BUREAU_DIR = ROOT / "data" / "raw" / "bureau"
BANKING_DIR = ROOT / "data" / "raw" / "banking"
OUT_DIR = ROOT / "data" / "processed"


def main(limit: int | None = None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bureau_files = sorted(BUREAU_DIR.glob("APP*.json"))
    if limit:
        bureau_files = bureau_files[:limit]

    metafile_rows = []

    for bfile in track(bureau_files, description="Building metafile..."):
        app_id = bfile.stem
        bank_file = BANKING_DIR / f"{app_id}.json"

        if not bank_file.exists():
            continue

        trades = BureauParser(bfile).parse_trades()
        tx = BankingParser(bank_file).parse_transactions()

        meta = build_metafile(trades, tx)
        metafile_rows.append(meta)

    final_df = pd.concat(metafile_rows, ignore_index=True)

    out_path = OUT_DIR / "metafile.parquet"
    final_df.to_parquet(out_path, index=False)

    print("\n Metafile saved to:", out_path)
    print("Rows:", final_df.shape[0], "Cols:", final_df.shape[1])
    print(final_df.head())


if __name__ == "__main__":
    main(limit=2000)  
