"""Risk analytics on the metafile using DuckDB SQL over Parquet.

Prefers the scored metafile (with model PD) when it exists, otherwise falls
back to the plain metafile.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb

from src.utils.io import setup_logging

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
META_PATH = ROOT / "data" / "processed" / "metafile.parquet"
SCORED_PATH = ROOT / "data" / "processed" / "metafile_scored.parquet"


def main() -> None:
    """Run the standard risk analytics query pack and print results."""
    source = SCORED_PATH if SCORED_PATH.exists() else META_PATH
    if not source.exists():
        raise FileNotFoundError(
            f"No metafile found at {source}. Build it first (python -m src.main build)."
        )
    has_labels_and_pd = source == SCORED_PATH
    logger.info("Running analytics on %s", source.name)

    con = duckdb.connect(database=":memory:")
    con.execute(
        f"CREATE VIEW metafile AS SELECT * FROM read_parquet('{source.as_posix()}');"
    )

    print("\n1) Risk band distribution")
    print(
        con.execute(
            """
            SELECT risk_band, COUNT(*) AS applicants,
                   ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
            FROM metafile
            GROUP BY 1
            ORDER BY risk_band;
            """
        ).fetchdf().to_string(index=False)
    )

    print("\n2) Avg score & delinquency by risk band")
    print(
        con.execute(
            """
            SELECT
              risk_band,
              ROUND(AVG(score_v3), 2) AS avg_score,
              ROUND(AVG(max_dpd), 2) AS avg_max_dpd,
              ROUND(AVG(total_credits), 2) AS avg_total_credits,
              ROUND(AVG(min_balance), 2) AS avg_min_balance
            FROM metafile
            GROUP BY 1
            ORDER BY risk_band;
            """
        ).fetchdf().to_string(index=False)
    )

    print("\n3) Top 10 highest-risk applicants by delinquency")
    print(
        con.execute(
            """
            SELECT loan_app_id, score_v3, bureau, max_dpd, num_trades,
                   total_sanctioned_amount, emi_txn_count
            FROM metafile
            ORDER BY max_dpd DESC, score_v3 ASC
            LIMIT 10;
            """
        ).fetchdf().to_string(index=False)
    )

    if has_labels_and_pd:
        print("\n4) Observed default rate by risk band (rule policy validation)")
        print(
            con.execute(
                """
                SELECT risk_band,
                       COUNT(*) AS applicants,
                       ROUND(100.0 * AVG(default_flag), 2) AS default_rate_pct,
                       ROUND(AVG(pd_score), 4) AS avg_model_pd
                FROM metafile
                GROUP BY 1
                ORDER BY risk_band;
                """
            ).fetchdf().to_string(index=False)
        )

        print("\n5) Model PD quintiles vs observed defaults (rank ordering)")
        print(
            con.execute(
                """
                WITH ranked AS (
                  SELECT *, NTILE(5) OVER (ORDER BY pd_score) AS pd_quintile
                  FROM metafile
                )
                SELECT pd_quintile,
                       COUNT(*) AS applicants,
                       ROUND(AVG(pd_score), 4) AS avg_model_pd,
                       ROUND(100.0 * AVG(default_flag), 2) AS default_rate_pct
                FROM ranked
                GROUP BY 1
                ORDER BY pd_quintile;
                """
            ).fetchdf().to_string(index=False)
        )
    else:
        logger.info("No scored metafile yet; run scoring to unlock default-rate analytics.")


if __name__ == "__main__":
    setup_logging()
    main()
