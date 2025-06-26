import pandas as pd
from src.data_processing import compute_rfm

def test_compute_rfm_basic():
    data = {
        'CustomerId': [1, 1, 2, 2, 2],
        'TransactionDate': [
            '2023-01-01', '2023-01-10',
            '2023-01-05', '2023-01-15', '2023-01-20'
        ],
        'Amount': [100, 200, 50, 60, 70]
    }
    df = pd.DataFrame(data)
    ref_date = pd.Timestamp('2023-01-21')
    rfm = compute_rfm(df, 'CustomerId', 'TransactionDate', 'Amount', ref_date=ref_date)
    # Customer 1: recency=11, frequency=2, monetary=300
    # Customer 2: recency=1, frequency=3, monetary=180
    row1 = rfm[rfm['CustomerId'] == 1].iloc[0]
    row2 = rfm[rfm['CustomerId'] == 2].iloc[0]
    assert row1['recency_days'] == 11
    assert row1['frequency'] == 2
    assert row1['monetary'] == 300
    assert row2['recency_days'] == 1
    assert row2['frequency'] == 3
    assert row2['monetary'] == 180 