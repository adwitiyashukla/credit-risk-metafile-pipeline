"""Credit scoring model: predict probability of default (PD) from the metafile.

Trains two candidate models on the applicant-level metafile using the observed
``default_flag`` label:

- Logistic regression (scaled) - the industry-standard interpretable baseline
- Gradient boosting - a stronger non-linear challenger

Both are evaluated on a held-out test set with ROC AUC, Gini and the KS
statistic (the standard credit-risk discrimination metrics). The best model
scores every applicant; results are saved to
``data/processed/metafile_scored.parquet`` and the fitted model to
``models/credit_model.joblib``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.io import setup_logging

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
META_PATH = ROOT / "data" / "processed" / "metafile.parquet"
SCORED_PATH = ROOT / "data" / "processed" / "metafile_scored.parquet"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "credit_model.joblib"

FEATURES = [
    "score_v3",
    "num_trades",
    "max_dpd",
    "avg_dpd",
    "total_current_balance",
    "total_sanctioned_amount",
    "dpd_30_plus",
    "dpd_60_plus",
    "dpd_90_plus",
    "txn_count",
    "total_credits",
    "total_debits",
    "salary_credits_count",
    "emi_txn_count",
    "cash_withdrawal_count",
    "avg_balance",
    "min_balance",
    "max_balance",
]
TARGET = "default_flag"
RANDOM_STATE = 42


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic: max separation between the cumulative
    distributions of predicted PD for defaulters vs non-defaulters."""
    order = np.argsort(y_prob)[::-1]
    y_sorted = np.asarray(y_true)[order]

    cum_bad = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
    cum_good = np.cumsum(1 - y_sorted) / max((1 - y_sorted).sum(), 1)
    return float(np.max(np.abs(cum_bad - cum_good)))


def build_candidates() -> dict[str, Pipeline]:
    """The two candidate model pipelines."""
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                (
                    "clf",
                    GradientBoostingClassifier(
                        n_estimators=200,
                        max_depth=3,
                        learning_rate=0.05,
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
    }


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Score a fitted model on the held-out set."""
    prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob)
    return {
        "roc_auc": round(float(auc), 4),
        "gini": round(float(2 * auc - 1), 4),
        "ks": round(ks_statistic(y_test.to_numpy(), prob), 4),
    }


def pd_decile_table(df: pd.DataFrame) -> pd.DataFrame:
    """Observed default rate by predicted-PD decile (1 = lowest predicted risk)."""
    out = df.copy()
    out["pd_decile"] = pd.qcut(out["pd_score"].rank(method="first"), 10, labels=range(1, 11))
    return (
        out.groupby("pd_decile", observed=True)
        .agg(
            applicants=("loan_app_id", "count"),
            avg_pd=("pd_score", "mean"),
            observed_default_rate=(TARGET, "mean"),
        )
        .round(4)
        .reset_index()
    )


def main(test_size: float = 0.2) -> dict[str, dict[str, float]]:
    """Train, evaluate, select and apply the credit scoring model."""
    if not META_PATH.exists():
        raise FileNotFoundError(
            f"Metafile not found at {META_PATH}. Build it first (python -m src.main build)."
        )

    df = pd.read_parquet(META_PATH)
    if TARGET not in df.columns or df[TARGET].isna().all():
        raise ValueError(
            "Metafile has no default_flag labels; regenerate data with the current "
            "generator (python -m src.main generate) and rebuild."
        )

    labelled = df.dropna(subset=[TARGET]).copy()
    labelled[TARGET] = labelled[TARGET].astype(int)
    features = [c for c in FEATURES if c in labelled.columns]

    X = labelled[features]
    y = labelled[TARGET]
    logger.info(
        "Training on %d applicants, %d features, default rate %.1f%%",
        len(labelled),
        len(features),
        100 * y.mean(),
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )

    results: dict[str, dict[str, float]] = {}
    fitted: dict[str, Pipeline] = {}
    for name, model in build_candidates().items():
        model.fit(X_train, y_train)
        fitted[name] = model
        results[name] = evaluate(model, X_test, y_test)
        logger.info("%s: %s", name, results[name])

    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best = fitted[best_name]
    logger.info("Selected model: %s", best_name)

    # Score every labelled applicant with the winning model.
    labelled["pd_score"] = best.predict_proba(labelled[features])[:, 1]
    labelled.to_parquet(SCORED_PATH, index=False)
    logger.info("Scored metafile saved to %s", SCORED_PATH)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best, "features": features, "name": best_name}, MODEL_PATH)
    logger.info("Model artifact saved to %s", MODEL_PATH)

    print("\nModel comparison (held-out test set):")
    print(pd.DataFrame(results).T.to_string())
    print("\nObserved default rate by predicted-PD decile:")
    print(pd_decile_table(labelled).to_string(index=False))

    return results


if __name__ == "__main__":
    setup_logging()
    main()
