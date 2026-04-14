import pandas as pd
from sklearn.model_selection import train_test_split

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

def aggregate_data(df, freq='H'):
    df['transaction_time'] = pd.to_datetime(df['transaction_time'])

    df['revenue'] = df['transaction_qty'] * df['unit_price']

    grouped = df.groupby([
        pd.Grouper(key='datetime', freq=freq),
        'store_id'
    ]).agg({
        'transaction_qty': 'sum',
        'revenue': 'sum'
    }).reset_index()

    return grouped

def split_series(df, target='transaction_qty'):

    split = int(len(df) * 0.8)

    train = df[target][:split]
    test = df[target][split:]

    return train, test

def fill_missing_time(df, freq='H'):
    full_data = []

    for store in df['store_id'].unique():
        store_df = df[df['store_id'] == store].copy()

        idx = pd.date_range(
            start=store_df['transaction_time'].min(),
            end=store_df['transaction_time'].max(),
            freq=freq
        )

        store_df = store_df.set_index('transaction_time').reindex(idx)
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