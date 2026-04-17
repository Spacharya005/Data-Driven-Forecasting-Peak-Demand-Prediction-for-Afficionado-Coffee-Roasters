import pandas as pd

def run_pipeline(df, target='transaction_qty'):

    features = [
        'hour', 'day_of_week',
        'lag_1', 'lag_24', 'lag_168',
        'rolling_mean_3', 'rolling_mean_7'
    ]

    X = df[features]
    y = df[target]

    # Time-based split
    split = int(len(df) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train, y_test

def aggregate_data(df, freq='D'):

    # Ensure datetime exists
    df = df.sort_values(by=['year', 'transaction_time']).reset_index(drop=True)

    df['datetime'] = pd.date_range(
        start='2025-01-01',
        periods=len(df),
        freq='h'   # keep hourly base
    )

    # Frequency mapping
    freq_map = {
        "hourly": "h",
        "daily": "D",
        "h": "h",
        "d": "D"
    }

    freq = freq_map.get(freq.lower(), freq)

    # ✅ CORRECT aggregation
    grouped = df.groupby([
        pd.Grouper(key='datetime', freq=freq),
        'store_id'
    ]).agg({
        'transaction_qty': 'sum'
    }).reset_index()

    grouped['revenue'] = grouped['transaction_qty'] * df['unit_price'].mean()

    return grouped

def split_series(df, target='target', test_size=0.2, return_df=False):

    split_index = int(len(df) * (1 - test_size))

    if return_df:
        train_df = df.iloc[:split_index].copy()
        test_df = df.iloc[split_index:].copy()
        return train_df, test_df

    # Existing behavior (DO NOT BREAK)
    train = df[target].iloc[:split_index]
    test = df[target].iloc[split_index:]

    return train, test

def fill_missing_time(df, freq='h'):
    full_data = []

    for store in df['store_id'].unique():
        store_df = df[df['store_id'] == store].copy()

        idx = pd.date_range(
            start=store_df['datetime'].min(),
            end=store_df['datetime'].max(),
            freq=freq
        )

        store_df = store_df.set_index('datetime').reindex(idx)
        store_df['store_id'] = store

        # Fill missing sales with 0 (IMPORTANT)
        store_df['transaction_qty'] = store_df['transaction_qty'].fillna(0)
        store_df['revenue'] = store_df['revenue'].fillna(0)

        store_df = store_df.reset_index().rename(columns={'index': 'transaction_time'})

        full_data.append(store_df)

    return pd.concat(full_data)

def train_test_split_time(df, test_days=7):
    split_date = df['transaction_time'].max() - pd.Timedelta(days=test_days)

    train = df[df['transaction_time'] <= split_date]
    test = df[df['transaction_time'] > split_date]

    return train, test