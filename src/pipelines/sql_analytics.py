import duckdb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
META_PATH = ROOT / "data" / "processed" / "metafile.parquet"

def main():
    con = duckdb.connect(database=":memory:")

    # Load parquet as a table
    con.execute(f"""
        CREATE VIEW metafile AS
        SELECT * FROM read_parquet('{META_PATH.as_posix()}');
    """)

    print("\n1) Risk band distribution")
    print(con.execute("""
        SELECT risk_band, COUNT(*) AS n
        FROM metafile
        GROUP BY 1
        ORDER BY risk_band;
    """).fetchdf())

    print("\n2) Avg score & delinquency by risk band")
    print(con.execute("""
        SELECT
          risk_band,
          ROUND(AVG(score_v3), 2) AS avg_score,
          ROUND(AVG(max_dpd), 2) AS avg_max_dpd,
          ROUND(AVG(total_credits), 2) AS avg_total_credits,
          ROUND(AVG(total_debits), 2) AS avg_total_debits
        FROM metafile
        GROUP BY 1
        ORDER BY risk_band;
    """).fetchdf())

    print("\n3) Top 10 worst max_dpd applicants")
    print(con.execute("""
        SELECT loan_app_id, score_v3, bureau, max_dpd, num_trades, total_sanctioned_amount, emi_txn_count
        FROM metafile
        ORDER BY max_dpd DESC, score_v3 ASC
        LIMIT 10;
    """).fetchdf())

if __name__ == "__main__":
    main()
