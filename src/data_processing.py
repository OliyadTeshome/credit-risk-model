import pandas as pd
from typing import Optional

def compute_rfm(df: pd.DataFrame, customer_id_col: str, date_col: str, amount_col: str, ref_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Compute Recency, Frequency, Monetary features per customer.
    Args:
        df: DataFrame with transactions
        customer_id_col: column name for customer ID
        date_col: column name for transaction date
        amount_col: column name for transaction amount
        ref_date: reference date for recency calculation (default: max date in df)
    Returns:
        DataFrame with columns: CustomerId, recency_days, frequency, monetary
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    if ref_date is None:
        ref_date = df[date_col].max() + pd.Timedelta(days=1)
    rfm = df.groupby(customer_id_col).agg(
        recency_days=(date_col, lambda x: (ref_date - x.max()).days),
        frequency=(date_col, 'count'),
        monetary=(amount_col, 'sum')
    ).reset_index()
    return rfm 