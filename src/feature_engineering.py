def create_features(df):

    df = df.sort_values('datetime')

    # Only create if missing
    if 'hour' not in df.columns:
        df['hour'] = df['datetime'].dt.hour

    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df['datetime'].dt.dayofweek

    # Lag features
    df['lag_1'] = df['transaction_qty'].shift(1)
    df['lag_24'] = df['transaction_qty'].shift(24)
    df['lag_168'] = df['transaction_qty'].shift(168)

    # Rolling
    df['rolling_mean_3'] = df['transaction_qty'].rolling(3).mean()
    df['rolling_mean_7'] = df['transaction_qty'].rolling(7).mean()

    df = df.dropna()

    return df