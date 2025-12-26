import json
import random
from datetime import date, timedelta
from pathlib import Path

random.seed(42)

ROOT = Path(__file__).resolve().parents[2]
BUREAU_DIR = ROOT / "data" / "raw" / "bureau"
BANKING_DIR = ROOT / "data" / "raw" / "banking"


FIRST_NAMES = ["Amit", "Rajesh", "Neha", "Priya", "Vikram", "Ananya", "Sanjay", "Pooja"]
LAST_NAMES = ["Sharma", "Gupta", "Singh", "Verma", "Patel", "Yadav", "Khan", "Iyer"]
BANKS = ["SBI", "HDFC", "ICICI", "Axis", "Kotak", "PNB"]


def rand_date(start: date, end: date) -> str:
    delta = (end - start).days
    return (start + timedelta(days=random.randint(0, delta))).isoformat()


def make_bureau_json(app_id: str) -> dict:
    name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    dob = rand_date(date(1975, 1, 1), date(2004, 12, 31))

    score = random.randint(300, 900)
    num_trades = random.randint(1, 6)

    trades = []
    for _ in range(num_trades):
        opened = rand_date(date(2012, 1, 1), date(2024, 12, 31))
        dpd = str(random.choice([0, 0, 0, 5, 15, 29, 30, 45, 60, 90, 120, 150])).zfill(3)

        trades.append(
            {
                "TRADE_REF_NUM": str(random.randint(10**17, 10**18 - 1)),
                "PRODUCT_TYPE": random.choice(["PL", "CC", "BL", "HL", "AL"]),
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


def make_banking_json(app_id: str) -> dict:
    bank = random.choice(BANKS)
    acct = str(random.randint(10**11, 10**12 - 1))
    start = date(2024, 1, 1)
    end = date(2025, 12, 1)

    txns = []
    balance = random.uniform(2000, 40000)

    for _ in range(random.randint(80, 220)):
        txn_date = rand_date(start, end)

        # salary credits occasionally
        if random.random() < 0.08:
            amt = random.uniform(30000, 120000)
            typ = "CR"
            desc = "Salary"
        else:
            # expenses/debits
            amt = random.uniform(50, 20000)
            typ = random.choice(["DR", "DR", "DR", "CR"])  # mostly debits
            desc = random.choice(
                ["UPI Transfer", "Cash Withdrawal", "POS Purchase", "Loan EMI", "Credit Card Payment", "Rent", "Utilities"]
            )

        amt = round(amt, 2)
        if typ == "DR":
            balance -= amt
        else:
            balance += amt

        txns.append(
            {
                "TXN_DATE": txn_date,
                "TXN_AMOUNT": amt,
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


def main(n: int = 2000) -> None:
    BUREAU_DIR.mkdir(parents=True, exist_ok=True)
    BANKING_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        app_id = f"APP{str(i).zfill(6)}"

        bureau = make_bureau_json(app_id)
        banking = make_banking_json(app_id)

        (BUREAU_DIR / f"{app_id}.json").write_text(json.dumps(bureau, indent=2))
        (BANKING_DIR / f"{app_id}.json").write_text(json.dumps(banking, indent=2))

    print(f"Generated {n} bureau JSON files in: {BUREAU_DIR}")
    print(f"Generated {n} banking JSON files in: {BANKING_DIR}")


if __name__ == "__main__":
    main()
