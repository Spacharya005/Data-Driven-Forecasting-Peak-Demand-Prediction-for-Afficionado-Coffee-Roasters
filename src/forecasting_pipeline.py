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

def aggregate_data(df, freq='D'):

    agg_df = df.groupby([
        pd.Grouper(key='datetime', freq=freq),
        'store_id'
    ]).agg({
        'transaction_qty': 'sum',
        'revenue': 'sum'
    }).reset_index()

    return agg_df

def split_series(df, target='transaction_qty'):

    split = int(len(df) * 0.8)

    train = df[target][:split]
    test = df[target][split:]

    return train, test