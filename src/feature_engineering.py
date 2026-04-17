
# def create_features(df):
#     df = df.copy()

#     if 'target' not in df.columns:
#         raise ValueError("❌ target missing before feature engineering")

#     n = len(df)

#     # ✅ ALWAYS SAFE
#     df['lag_1'] = df['target'].shift(1)
#     feat_df = feat_df.dropna()
#     # ✅ CONDITIONAL LAGS
#     if n > 24:
#         df['lag_24'] = df['target'].shift(24)

#     if n > 168:
#         df['lag_168'] = df['target'].shift(168)

#     # ✅ ROLLING
#     if n > 3:
#         df['rolling_mean_3'] = df['target'].rolling(3).mean()

#     if n > 7:
#         df['rolling_mean_7'] = df['target'].rolling(7).mean()

#     # ✅ TIME FEATURES (always safe)
#     df['hour'] = df['datetime'].dt.hour
#     df['day_of_week'] = df['datetime'].dt.dayofweek

#     # 🔥 CRITICAL FIX → DO NOT DROP EVERYTHING
#     df = df.dropna(subset=['lag_1']).reset_index(drop=True)

#     return df



def create_features(df):
    df = df.copy()

    if 'target' not in df.columns:
        raise ValueError("❌ target missing before feature engineering")

    n = len(df)

    # ✅ ALWAYS SAFE
    df['lag_1'] = df['target'].shift(1)
    df['lag_24'] = df['target'].shift(24)
    df['lag_168'] = df['target'].shift(168)

    df['rolling_mean_3'] = df['target'].shift(1).rolling(3).mean()
    df['rolling_mean_7'] = df['target'].shift(1).rolling(7).mean()

    # ✅ TIME FEATURES (always safe)
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek

    # 🔥 CRITICAL FIX → DO NOT DROP EVERYTHING
    df = df.dropna(subset=['lag_1']).reset_index(drop=True)

    return df