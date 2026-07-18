"""Command-line entrypoint for the credit risk pipeline.

Usage:
    python -m src.main generate [--n 2000] [--seed 42]
    python -m src.main build [--limit N]
    python -m src.main score
    python -m src.main analytics
    python -m src.main all [--n 2000]
"""

from __future__ import annotations

import argparse

from src.pipelines import generate_synthetic_data, run_pipeline, sql_analytics
from src.scoring import score_model
from src.utils.io import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="credit-risk-pipeline",
        description="End-to-end credit risk metafile pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate", help="Generate synthetic raw bureau/banking data")
    p_gen.add_argument("--n", type=int, default=2000, help="Number of applicants")
    p_gen.add_argument("--seed", type=int, default=42, help="Random seed")
    p_gen.add_argument(
        "--start", type=int, default=0, help="Resume generation from this applicant index"
    )

    p_build = sub.add_parser("build", help="Parse raw data and build the metafile")
    p_build.add_argument("--limit", type=int, default=None, help="Max applicants to process")

    sub.add_parser("score", help="Train the PD model and score all applicants")
    sub.add_parser("analytics", help="Run DuckDB SQL risk analytics")

    p_all = sub.add_parser("all", help="Run the full pipeline end to end")
    p_all.add_argument("--n", type=int, default=2000, help="Number of applicants")
    p_all.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    args = build_parser().parse_args(argv)

    if args.command == "generate":
        generate_synthetic_data.main(n=args.n, seed=args.seed, start=args.start)
    elif args.command == "build":
        run_pipeline.main(limit=args.limit)
    elif args.command == "score":
        score_model.main()
    elif args.command == "analytics":
        sql_analytics.main()
    elif args.command == "all":
        generate_synthetic_data.main(n=args.n, seed=args.seed)
        run_pipeline.main()
        score_model.main()
        sql_analytics.main()


if __name__ == "__main__":
    main()
