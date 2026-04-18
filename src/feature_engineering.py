def create_features(df):
    df = df.copy()

    if 'target' not in df.columns:
        raise ValueError("❌ target missing before feature engineering")

    n = len(df)

    df['lag_1'] = df['target'].shift(1)
    df['lag_24'] = df['target'].shift(24)
    df['lag_168'] = df['target'].shift(168)

    df['rolling_mean_3'] = df['target'].shift(1).rolling(3).mean()
    df['rolling_mean_7'] = df['target'].shift(1).rolling(7).mean()

    
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek

    df = df.dropna().reset_index(drop=True)

    return df