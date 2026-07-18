import pandas as pd


def bureau_features(trades_df: pd.DataFrame) -> pd.DataFrame:
    # trades_df: multiple rows per loan_app_id
    g = trades_df.groupby("loan_app_id", as_index=False)

    out = g.agg(
        score_v3=("score_v3", "first"),
        bureau=("bureau", "first"),
        num_trades=("trade_ref_num", "nunique"),
        max_dpd=("dpd", "max"),
        avg_dpd=("dpd", "mean"),
        total_current_balance=("current_balance", "sum"),
        total_sanctioned_amount=("sanctioned_amount", "sum"),
    )

    # simple delinquency flags
    out["dpd_30_plus"] = (out["max_dpd"] >= 30).astype(int)
    out["dpd_60_plus"] = (out["max_dpd"] >= 60).astype(int)
    out["dpd_90_plus"] = (out["max_dpd"] >= 90).astype(int)

    return out


def banking_features(tx_df: pd.DataFrame) -> pd.DataFrame:
    # tx_df: multiple rows per loan_app_id
    tx = tx_df.copy()

    # split credits vs debits
    tx["is_credit"] = (tx["txn_type"] == "CR").astype(int)
    tx["is_debit"] = (tx["txn_type"] == "DR").astype(int)

    tx["is_salary"] = tx["txn_desc"].str.contains("salary", case=False, na=False).astype(int)
    tx["is_emi"] = tx["txn_desc"].str.contains("emi", case=False, na=False).astype(int)
    tx["is_cash_withdrawal"] = (
        tx["txn_desc"].str.contains("cash withdrawal", case=False, na=False).astype(int)
    )

    # amounts
    tx["credit_amt"] = tx["txn_amount"] * tx["is_credit"]
    tx["debit_amt"] = tx["txn_amount"] * tx["is_debit"]

    g = tx.groupby("loan_app_id", as_index=False)

    out = g.agg(
        bank_name=("bank_name", "first"),
        acct_num=("acct_num", "first"),
        txn_count=("txn_amount", "count"),
        total_credits=("credit_amt", "sum"),
        total_debits=("debit_amt", "sum"),
        salary_credits_count=("is_salary", "sum"),
        emi_txn_count=("is_emi", "sum"),
        cash_withdrawal_count=("is_cash_withdrawal", "sum"),
        avg_balance=("bal_after", "mean"),
        min_balance=("bal_after", "min"),
        max_balance=("bal_after", "max"),
    )

    return out


def assign_risk_band(score: float, max_dpd: float) -> int:
    """
    Very simple rule-based risk banding (1=best, 5=worst).
    This is NOT a real credit policy; it's for project demo.
    """
    if max_dpd >= 90:
        return 5
    if score < 500:
        return 4
    if score < 650:
        return 3
    if score < 750:
        return 2
    return 1


def build_metafile(bureau_trades: pd.DataFrame, banking_tx: pd.DataFrame) -> pd.DataFrame:
    bf = bureau_features(bureau_trades)
    kf = banking_features(banking_tx)

    meta = bf.merge(kf, on="loan_app_id", how="inner")

    meta["risk_band"] = meta.apply(lambda r: assign_risk_band(r["score_v3"], r["max_dpd"]), axis=1)

    return meta
