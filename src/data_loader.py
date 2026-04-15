import pandas as pd

def load_data(path):
    # ✅ Use dynamic path
    df = pd.read_csv(path)

    # Convert datetime
    df['datetime'] = pd.to_datetime(
        df['year'].astype(str) + "-01-01 " + df['transaction_time'],
        format='%Y-%m-%d %H:%M:%S',
        errors='coerce'
    )

    # Revenue
    if 'revenue' not in df.columns:
        df['revenue'] = df['transaction_qty'] * df['unit_price']

    return df