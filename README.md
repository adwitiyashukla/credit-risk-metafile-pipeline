# Credit Risk Metafile Pipeline

![CI](https://github.com/adwitiyashukla/credit-risk-metafile-pipeline/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

An end-to-end **credit risk data pipeline** inspired by how fintech/NBFC underwriting teams build decisioning datasets ("**metafiles**") from **credit bureau** and **bank statement** data — including a trained **probability-of-default (PD) scoring model**.

```text
raw bureau JSON ─┐
                 ├─> parsers ─> feature engineering ─> metafile.parquet ─> PD model ─> SQL analytics
raw banking JSON ┘                                          ▲
                                labels.csv (observed defaults, joined for training)
```

**Why this matters:** real NBFC and fintech underwriting teams consolidate fragmented bureau and banking data into a single analytics-ready dataset, then train scoring models on observed loan performance. This project reproduces that workflow end to end on realistic synthetic data.

---

## Pipeline stages

### 1) Synthetic data generation (data-lake style)
Generates raw JSON per applicant, driven by a **latent risk factor** so that scores, delinquency, cash-flow behaviour and the observed default outcome are all *consistently correlated* (i.e., the data is learnable, not noise):

- `data/raw/bureau/APPxxxxxx.json` — bureau report (score, trades, DPD; includes messy real-world values like `"DPD": "XXX"`)
- `data/raw/banking/APPxxxxxx.json` — bank statement (monthly salary credits, EMIs, spending, coherent running balance)
- `data/raw/labels.csv` — observed 12-month default flag per applicant (the ML target)

Generation is **deterministic per applicant** (seeded), so runs are reproducible and interrupted generations can be resumed with `--start`.

### 2) Parsing
Raw JSON → clean tables with defensive handling of corrupt files, missing fields and non-numeric DPD codes. Bureau: 1 row per trade. Banking: 1 row per transaction.

### 3) Feature engineering (metafile)
One vectorised aggregation pass builds applicant-level features:

- **Bureau:** `num_trades`, `max_dpd`, `avg_dpd`, `total_sanctioned_amount`, delinquency flags (`dpd_30_plus`, `dpd_60_plus`, `dpd_90_plus`)
- **Banking:** `total_credits`, `total_debits`, `salary_credits_count`, `emi_txn_count`, `cash_withdrawal_count`, `avg_balance`, `min_balance`, `max_balance`
- **Policy:** rule-based `risk_band` (1 = lowest risk, 5 = highest)

Output: `data/processed/metafile.parquet` (1 row per applicant, labels joined).

### 4) PD scoring model
Trains and compares two models on the metafile with a held-out test set:

| Model | ROC AUC | Gini | KS |
|---|---|---|---|
| **Logistic regression (selected)** | **0.74** | **0.48** | **0.39** |
| Gradient boosting | 0.70 | 0.40 | 0.38 |

These are realistic discrimination levels for bureau + banking scorecards. The winning model scores every applicant (`data/processed/metafile_scored.parquet`) and is saved to `models/credit_model.joblib`.

Observed default rate by predicted-PD decile (2,000 applicants, ~18% base rate) rises monotonically from **6%** (decile 1) to **54%** (decile 10) — the model rank-orders risk correctly.

### 5) SQL risk analytics (DuckDB on Parquet)
A query pack over the scored metafile:

- risk band distribution and profile
- **observed default rate by risk band** (validates the rule policy: 6.4% → 34.6%)
- **model PD quintiles vs observed defaults** (rank-ordering check)
- top delinquent applicants

---

## Quickstart

```bash
git clone https://github.com/adwitiyashukla/credit-risk-metafile-pipeline.git
cd credit-risk-metafile-pipeline
pip install -r requirements.txt

# full pipeline: generate -> build -> score -> analytics
python -m src.main all
```

Or run stages individually:

```bash
python -m src.main generate --n 2000 --seed 42   # synthetic raw data
python -m src.main build                          # parse + build metafile
python -m src.main score                          # train + apply PD model
python -m src.main analytics                      # DuckDB SQL analytics
```

### Development

```bash
pip install -r requirements-dev.txt
python -m pytest        # 32 unit tests
ruff check src tests    # lint
```

CI runs lint, tests and a 50-applicant end-to-end smoke run on Python 3.11–3.13 (see `.github/workflows/ci.yml`).

---

## Tech stack

**Python 3.11+**, **Pandas** (feature engineering), **scikit-learn** (PD model), **DuckDB** (SQL analytics on Parquet), **PyArrow** (Parquet), **Rich** (progress), **pytest + ruff + GitHub Actions** (quality).

---

## Project structure

```text
credit-risk-pipeline/
  data/
    raw/
      bureau/            # raw bureau JSON per applicant (generated)
      banking/           # raw banking JSON per applicant (generated)
      labels.csv         # observed default labels (generated)
    processed/
      metafile.parquet          # applicant-level dataset
      metafile_scored.parquet   # + model PD scores
  models/
    credit_model.joblib  # fitted PD model (generated)
  src/
    main.py              # CLI entrypoint
    parsers/             # bureau_parser.py, banking_parser.py
    features/            # feature_engineering.py
    scoring/             # score_model.py (PD model)
    pipelines/           # generate_synthetic_data.py, run_pipeline.py, sql_analytics.py
    utils/               # io.py (IO + logging helpers)
  tests/                 # pytest suite
  .github/workflows/     # CI
```

---

## Sample output

Risk analytics on 2,000 scored applicants (`python -m src.main analytics`):

```text
4) Observed default rate by risk band (rule policy validation)
 risk_band  applicants  default_rate_pct  avg_model_pd
         1         824              6.43        0.0621
         2         483             13.66        0.1393
         3         156             30.77        0.2990
         4           3             33.33        0.5258
         5         534             34.64        0.3411

5) Model PD quintiles vs observed defaults (rank ordering)
 pd_quintile  applicants  avg_model_pd  default_rate_pct
           1         400        0.0402              5.75
           2         400        0.0730              5.50
           3         400        0.1211             13.00
           4         400        0.2055             19.75
           5         400        0.4324             44.25
```

Both the rule-based bands and the model PD rank-order observed defaults monotonically.

---

## Disclaimer

All data is synthetic and all policies/models are illustrative — this is a data engineering portfolio project, not a real credit policy.
