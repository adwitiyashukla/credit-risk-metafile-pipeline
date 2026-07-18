"""Generate synthetic raw data for the credit risk pipeline.

Each applicant gets a latent risk factor ``u`` in [0, 1] that drives every
signal consistently:

- bureau score (higher risk -> lower score)
- delinquency (higher risk -> more and deeper DPD)
- banking behaviour (higher risk -> deeper overdrafts, more cash withdrawals)
- the observed 12-month default outcome (the ML target)

This makes the generated data *learnable*: a scoring model trained on the
metafile can recover real signal instead of fitting noise.

Outputs:
    data/raw/bureau/APPxxxxxx.json   - one bureau report per applicant
    data/raw/banking/APPxxxxxx.json  - one bank statement per applicant
    data/raw/labels.csv              - observed default flag per applicant
"""

from __future__ import annotations

import csv
import json
import logging
import math
import random
import zlib
from datetime import date, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
BUREAU_DIR = ROOT / "data" / "raw" / "bureau"
BANKING_DIR = ROOT / "data" / "raw" / "banking"
LABELS_PATH = ROOT / "data" / "raw" / "labels.csv"

FIRST_NAMES = ["Amit", "Rajesh", "Neha", "Priya", "Vikram", "Ananya", "Sanjay", "Pooja"]
LAST_NAMES = ["Sharma", "Gupta", "Singh", "Verma", "Patel", "Yadav", "Khan", "Iyer"]
BANKS = ["SBI", "HDFC", "ICICI", "Axis", "Kotak", "PNB"]
PRODUCT_TYPES = ["PL", "CC", "BL", "HL", "AL"]
DEBIT_DESCRIPTIONS = [
    "UPI Transfer",
    "Cash Withdrawal",
    "POS Purchase",
    "Credit Card Payment",
    "Rent",
    "Utilities",
]

TXN_WINDOW_START = date(2024, 1, 1)
TXN_WINDOW_END = date(2025, 12, 1)


def _rand_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def make_bureau_json(app_id: str, u: float) -> dict:
    """Build a bureau report whose score and DPD reflect latent risk ``u``."""
    name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    dob = _rand_date(date(1975, 1, 1), date(2004, 12, 31)).isoformat()

    score = int(round(880 - 480 * u + random.gauss(0, 35)))
    score = max(300, min(900, score))

    num_trades = random.randint(1, 6)
    trades = []
    for _ in range(num_trades):
        opened = _rand_date(date(2012, 1, 1), date(2024, 12, 31)).isoformat()

        # Higher latent risk -> higher chance of a delinquent trade, deeper DPD.
        if random.random() < 0.04 + 0.60 * u:
            dpd_value = random.choice([5, 15, 29, 30, 45, 60, 90, 120, 150, 180][int(u * 4):])
        else:
            dpd_value = 0

        # Bureaus occasionally report non-numeric DPD codes (e.g. "XXX").
        dpd = "XXX" if random.random() < 0.03 else str(dpd_value).zfill(3)

        trades.append(
            {
                "TRADE_REF_NUM": str(random.randint(10**17, 10**18 - 1)),
                "PRODUCT_TYPE": random.choice(PRODUCT_TYPES),
                "DATE_OPENED": opened,
                "CURRENT_BALANCE": round(random.uniform(0, 500000), 2),
                "SANCTIONED_AMOUNT": round(random.uniform(50000, 800000), 2),
                "DPD": dpd,
            }
        )

    return {
        "LOAN_APP_ID": app_id,
        "NAME": name,
        "DOB": dob,
        "BUREAU": random.choice(["CIBIL", "Experian", "Equifax", "CRIF"]),
        "SCORE_V3": score,
        "TRADES": trades,
    }


def make_banking_json(app_id: str, u: float) -> dict:
    """Build a bank statement whose cash-flow patterns reflect latent risk ``u``."""
    bank = random.choice(BANKS)
    acct = str(random.randint(10**11, 10**12 - 1))

    is_salaried = random.random() < 0.90
    monthly_salary = random.uniform(30000, 120000)

    # Riskier applicants lean harder on overdraft.
    overdraft_limit = 5000 + 60000 * u

    events: list[tuple[date, float, str, str]] = []  # (date, amount, type, desc)

    # Monthly salary credits (day 1-5 of each month in the window).
    if is_salaried:
        d = TXN_WINDOW_START
        while d <= TXN_WINDOW_END:
            pay_day = d.replace(day=random.randint(1, 5))
            amount = monthly_salary * random.uniform(0.97, 1.03)
            events.append((pay_day, amount, "CR", "Salary"))
            d = (d.replace(day=28) + timedelta(days=4)).replace(day=1)

    # Monthly EMI debits.
    emi_amount = random.uniform(3000, 25000)
    d = TXN_WINDOW_START
    while d <= TXN_WINDOW_END:
        emi_day = d.replace(day=random.randint(5, 10))
        events.append((emi_day, emi_amount * random.uniform(0.99, 1.01), "DR", "Loan EMI"))
        d = (d.replace(day=28) + timedelta(days=4)).replace(day=1)

    # Everyday spending; riskier applicants withdraw cash more often.
    for _ in range(random.randint(60, 180)):
        txn_date = _rand_date(TXN_WINDOW_START, TXN_WINDOW_END)
        amount = random.uniform(50, 0.25 * monthly_salary)
        if random.random() < 0.10 + 0.25 * u:
            desc = "Cash Withdrawal"
        else:
            desc = random.choice(DEBIT_DESCRIPTIONS)
        typ = "CR" if random.random() < 0.08 else "DR"
        events.append((txn_date, amount, typ, desc if typ == "DR" else "UPI Transfer"))

    # Chronological order, then compute a coherent running balance.
    events.sort(key=lambda e: e[0])

    balance = random.uniform(5000, 60000)
    txns = []
    for txn_date, amount, typ, desc in events:
        amount = round(amount, 2)
        if typ == "DR":
            # Clamp debits at the overdraft limit so balances stay plausible.
            if balance - amount < -overdraft_limit:
                amount = round(max(balance + overdraft_limit, 0.0), 2)
                if amount <= 0:
                    continue
            balance -= amount
        else:
            balance += amount

        txns.append(
            {
                "TXN_DATE": txn_date.isoformat(),
                "TXN_AMOUNT": amount,
                "TXN_TYPE": typ,
                "TXN_DESC": desc,
                "BAL_AFTER": round(balance, 2),
            }
        )

    return {
        "LOAN_APP_ID": app_id,
        "BANK_NAME": bank,
        "ACCT_NUM": acct,
        "TRANSACTIONS": txns,
    }


def draw_default_flag(u: float, bureau: dict) -> int:
    """Observed 12-month default outcome, driven by latent risk + realized DPD."""
    dpds = []
    for trade in bureau["TRADES"]:
        try:
            dpds.append(int(trade["DPD"]))
        except (TypeError, ValueError):
            continue
    max_dpd = max(dpds, default=0)

    logit = -3.6 + 4.6 * u + 0.9 * (max_dpd >= 90) + 0.4 * (max_dpd >= 30)
    return int(random.random() < _sigmoid(logit))


def _applicant_seed(seed: int, app_id: str) -> int:
    """Stable per-applicant seed, independent of generation order."""
    return zlib.crc32(f"{seed}:{app_id}".encode())


def generate_applicant(app_id: str, seed: int) -> tuple[dict, dict, int]:
    """Deterministically generate one applicant's bureau, banking and label.

    Each applicant gets its own RNG stream, so results are identical whether
    the dataset is generated in one pass or resumed in chunks.
    """
    random.seed(_applicant_seed(seed, app_id))
    u = random.betavariate(2, 5)  # right-skewed: most applicants low risk

    bureau = make_bureau_json(app_id, u)
    banking = make_banking_json(app_id, u)
    default_flag = draw_default_flag(u, bureau)
    return bureau, banking, default_flag


def main(n: int = 2000, seed: int = 42, start: int = 0) -> None:
    """Generate applicants ``start``..``n-1`` worth of raw JSON plus labels.

    ``start`` supports resuming an interrupted generation; output is identical
    to a single full run thanks to per-applicant seeding.
    """
    BUREAU_DIR.mkdir(parents=True, exist_ok=True)
    BANKING_DIR.mkdir(parents=True, exist_ok=True)

    labels: list[tuple[str, int]] = []

    for i in range(start, n):
        app_id = f"APP{str(i).zfill(6)}"
        bureau, banking, default_flag = generate_applicant(app_id, seed)
        labels.append((app_id, default_flag))

        (BUREAU_DIR / f"{app_id}.json").write_text(json.dumps(bureau, indent=2))
        (BANKING_DIR / f"{app_id}.json").write_text(json.dumps(banking, indent=2))

    with open(LABELS_PATH, "a" if start else "w", newline="") as f:
        writer = csv.writer(f)
        if start == 0:
            writer.writerow(["loan_app_id", "default_flag"])
        writer.writerows(labels)

    default_rate = sum(flag for _, flag in labels) / max(len(labels), 1)
    logger.info("Generated applicants %d-%d in %s", start, n - 1, BUREAU_DIR.parent)
    logger.info(
        "Labels written to %s (default rate this run: %.1f%%)", LABELS_PATH, 100 * default_rate
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
