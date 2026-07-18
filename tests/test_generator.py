"""Tests for the synthetic data generator."""

import json

import pytest

import src.pipelines.generate_synthetic_data as gen


@pytest.fixture
def patched_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(gen, "BUREAU_DIR", tmp_path / "bureau")
    monkeypatch.setattr(gen, "BANKING_DIR", tmp_path / "banking")
    monkeypatch.setattr(gen, "LABELS_PATH", tmp_path / "labels.csv")
    return tmp_path


def test_generator_creates_expected_files(patched_dirs):
    gen.main(n=5, seed=1)

    bureau_files = sorted((patched_dirs / "bureau").glob("*.json"))
    banking_files = sorted((patched_dirs / "banking").glob("*.json"))
    assert len(bureau_files) == 5
    assert len(banking_files) == 5

    labels = (patched_dirs / "labels.csv").read_text().strip().splitlines()
    assert labels[0] == "loan_app_id,default_flag"
    assert len(labels) == 6  # header + 5 rows


def test_generated_bureau_schema(patched_dirs):
    gen.main(n=3, seed=2)

    data = json.loads(next((patched_dirs / "bureau").glob("*.json")).read_text())
    assert {"LOAN_APP_ID", "NAME", "DOB", "BUREAU", "SCORE_V3", "TRADES"} <= set(data)
    assert 300 <= data["SCORE_V3"] <= 900
    for trade in data["TRADES"]:
        assert {"TRADE_REF_NUM", "PRODUCT_TYPE", "DPD"} <= set(trade)


def test_generated_banking_is_chronological(patched_dirs):
    gen.main(n=3, seed=3)

    data = json.loads(next((patched_dirs / "banking").glob("*.json")).read_text())
    dates = [t["TXN_DATE"] for t in data["TRANSACTIONS"]]
    assert dates == sorted(dates)


def test_generator_is_deterministic(patched_dirs):
    gen.main(n=2, seed=42)
    first = next((patched_dirs / "bureau").glob("*.json")).read_text()

    gen.main(n=2, seed=42)
    second = next((patched_dirs / "bureau").glob("*.json")).read_text()

    assert first == second


def test_resumed_generation_matches_single_run(patched_dirs, tmp_path_factory, monkeypatch):
    # Full run in one pass.
    gen.main(n=4, seed=7)
    single = {p.name: p.read_text() for p in (patched_dirs / "bureau").glob("*.json")}
    single_labels = (patched_dirs / "labels.csv").read_text()

    # Same run, resumed in two chunks, in a fresh directory.
    other = tmp_path_factory.mktemp("resumed")
    monkeypatch.setattr(gen, "BUREAU_DIR", other / "bureau")
    monkeypatch.setattr(gen, "BANKING_DIR", other / "banking")
    monkeypatch.setattr(gen, "LABELS_PATH", other / "labels.csv")
    gen.main(n=2, seed=7)
    gen.main(n=4, seed=7, start=2)
    chunked = {p.name: p.read_text() for p in (other / "bureau").glob("*.json")}
    chunked_labels = (other / "labels.csv").read_text()

    assert single == chunked
    assert single_labels == chunked_labels
