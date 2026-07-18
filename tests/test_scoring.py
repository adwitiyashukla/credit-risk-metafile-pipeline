"""Tests for the credit scoring model utilities."""

import numpy as np
import pandas as pd

from src.scoring.score_model import build_candidates, ks_statistic, pd_decile_table


def test_ks_perfect_separation():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert ks_statistic(y_true, y_prob) == 1.0


def test_ks_no_separation():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.5, 0.5, 0.5, 0.5])
    assert ks_statistic(y_true, y_prob) <= 0.5


def test_candidates_fit_and_rank_order():
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(size=n)
    y = (x + rng.normal(scale=0.5, size=n) > 0.8).astype(int)
    X = pd.DataFrame({"f1": x, "f2": rng.normal(size=n)})

    for name, model in build_candidates().items():
        model.fit(X, y)
        prob = model.predict_proba(X)[:, 1]
        # Signal feature should produce reasonable in-sample discrimination.
        assert ks_statistic(y, prob) > 0.5, name


def test_pd_decile_table():
    n = 100
    df = pd.DataFrame(
        {
            "loan_app_id": [f"A{i}" for i in range(n)],
            "pd_score": np.linspace(0.01, 0.99, n),
            "default_flag": [0] * 50 + [1] * 50,
        }
    )
    table = pd_decile_table(df)

    assert len(table) == 10
    assert table["applicants"].sum() == n
    # Highest decile should have a higher observed default rate than the lowest.
    assert (
        table["observed_default_rate"].iloc[-1] > table["observed_default_rate"].iloc[0]
    )
