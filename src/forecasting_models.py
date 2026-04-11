import numpy as np
import pandas as pd

# ML
from sklearn.ensemble import GradientBoostingRegressor

# Statistical
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Prophet
from prophet import Prophet


# -------------------------------
# BASELINE MODELS
# -------------------------------

def naive_forecast(train, test):
    last_value = train.iloc[-1]
    return np.repeat(last_value, len(test))


def moving_average_forecast(train, test, window=3):
    preds = []
    history = list(train)

    for t in range(len(test)):
        preds.append(np.mean(history[-window:]))
        history.append(test.iloc[t])

    return np.array(preds)

def gradient_boosting_model(X_train, y_train, X_test):

    model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    return preds

# -------------------------------
# STATISTICAL MODELS
# -------------------------------

def arima_forecast(train, test):

    try:
        # ✅ Clean data
        train = train.dropna()

        # ✅ Safety checks
        if len(train) < 10:
            return [train.mean()] * len(test)

        if train.nunique() <= 1:
            return [train.iloc[-1]] * len(test)

        # ✅ Safer ARIMA config
        model = ARIMA(train, order=(1,1,1))
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=len(test))

        return forecast

    except Exception as e:
        print("⚠️ ARIMA failed:", e)

        # ✅ FALLBACK (VERY IMPORTANT)
        return np.repeat(train.mean(), len(test))


def exp_smoothing_forecast(train, test):

    model = ExponentialSmoothing(
        train,
        trend='add',
        seasonal='add',
        seasonal_periods=24   # 🔥 for hourly data
    )

    model_fit = model.fit()
    forecast = model_fit.forecast(len(test))

    return forecast.values


# -------------------------------
# PROPHET
# -------------------------------

def prophet_forecast(df, test_size):

    prophet_df = df[['datetime', 'transaction_qty']]
    prophet_df.columns = ['ds', 'y']

    train_df = prophet_df.iloc[:-test_size]  # use all but last test_size rows for training

    model = Prophet()
    model.fit(train_df)

    future = model.make_future_dataframe(periods=test_size, freq='D')
    forecast = model.predict(future)

    return forecast['yhat'][-test_size:].values


# -------------------------------
# MACHINE LEARNING
# -------------------------------

def moving_average_forecast(train, test, window=3):

    preds = []
    history = list(train)

    for t in range(len(test)):
        pred = np.mean(history[-window:])
        preds.append(pred)
        history.append(pred)   # 🔥 FIX (use prediction, not actual)

    return np.array(preds)


# -------------------------------
# MODEL SELECTOR
# -------------------------------

def run_model(model_name, train, test, X_train=None, X_test=None, df=None, horizon=None):
    if horizon is not None:
        last_value = train.iloc[-1]
        future_preds = np.repeat(last_value, horizon)
        return future_preds
    
    if model_name == "Naive":
        return naive_forecast(train, test)

    elif model_name == "Moving Average":
        return moving_average_forecast(train, test)

    elif model_name == "ARIMA":
        return arima_forecast(train, test)

    elif model_name == "Exponential Smoothing":
        return exp_smoothing_forecast(train, test)

    elif model_name == "Prophet":
        return prophet_forecast(df, len(test))

    elif model_name == "Gradient Boosting":
        return gradient_boosting_model(X_train, train, X_test)

    else:
        raise ValueError("Invalid Model")

 