def create_features(df):
    
    # print("Before feature engineering:", df.shape)
    df = df.sort_values('datetime')

    # Only create if missing
    if 'hour' not in df.columns:
        df['hour'] = df['datetime'].dt.hour

    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df['datetime'].dt.dayofweek

    # Lag features
    df['lag_1'] = df['transaction_qty'].shift(1)
    df['lag_24'] = df['transaction_qty'].shift(24)
    if len(df) > 168:
        df['lag_168'] = df['target'].shift(168)

    # Rolling
    df['rolling_mean_3'] = df['transaction_qty'].rolling(3).mean()
    df['rolling_mean_7'] = df['transaction_qty'].rolling(7).mean()
    # Drop only rows where target is missing
    if 'target' in df.columns:
        df = df.dropna(subset=['target'])
    else:
        print("⚠️ target column missing in feature_engineering")
        print(df.columns)

    # Fill lag/rolling NaNs instead of dropping everything
    df = df.fillna(method='bfill').fillna(method='ffill')

    return df

# def create_features(df):
#     df = df.sort_values(['store_id', 'datetime'])

#     # Time features
#     df['hour'] = df['datetime'].dt.hour
#     df['day_of_week'] = df['datetime'].dt.dayofweek

#     # Lag features
#     df['lag_1'] = df.groupby('store_id')['transaction_qty'].shift(1)
#     df['lag_24'] = df.groupby('store_id')['transaction_qty'].shift(24)
#     df['lag_168'] = df.groupby('store_id')['transaction_qty'].shift(168)

#     # Rolling features
#     df['rolling_3'] = df.groupby('store_id')['transaction_qty'].transform(lambda x: x.rolling(3).mean())
#     df['rolling_7'] = df.groupby('store_id')['transaction_qty'].transform(lambda x: x.rolling(7).mean())

#     return df.dropna()