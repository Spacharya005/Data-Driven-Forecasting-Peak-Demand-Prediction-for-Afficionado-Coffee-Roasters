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


# def moving_average_forecast(train, test, window=3):
#     preds = []
#     history = list(train)

#     for t in range(len(test)):
#         preds.append(np.mean(history[-window:]))
#         history.append(test.iloc[t])

#     return np.array(preds)

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

from statsmodels.tsa.arima.model import ARIMA
import numpy as np

def arima_forecast(train, test):

    # ✅ Always inside try
    try:
        train = train.dropna()

        if len(train) < 10:
            return np.repeat(train.mean(), len(test))

        if train.nunique() <= 1:
            return np.repeat(train.iloc[-1], len(test))

        model = ARIMA(train, order=(1,1,1))
        model_fit = model.fit()   # ✅ INSIDE try

        forecast = model_fit.forecast(steps=len(test))

        return np.array(forecast)

    except Exception as e:
        print("⚠️ ARIMA failed:", e)

        # ✅ fallback
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

def prophet_forecast(df, periods):

    prophet_df = df[['datetime', 'target']].copy()
    prophet_df.columns = ['ds', 'y']

    model = Prophet()
    model.fit(prophet_df)

    # 🔥 Detect frequency automatically
    freq = pd.infer_freq(prophet_df['ds'])

    if freq is None:
        freq = 'D'  # fallback

    future = model.make_future_dataframe(periods=periods, freq=freq)

    forecast = model.predict(future)

    return forecast['yhat'][-periods:].values


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

def run_model(model_name, train, test=None, X_train=None, X_test=None, df=None, horizon=None):

    try:
        # ----------------------
        # BASELINE
        # ----------------------
        if model_name == "Naive":
            return naive_forecast(train, test)

        elif model_name == "Moving Average":
            return moving_average_forecast(train, test)

        # ----------------------
        # STAT MODELS
        # ----------------------
        elif model_name == "ARIMA":
            return arima_forecast(train, test)

        elif model_name == "Exponential Smoothing":
            return exp_smoothing_forecast(train, test)

        # ----------------------
        # PROPHET
        # ----------------------
        elif model_name == "Prophet":
            if df is None:
                raise ValueError("Prophet requires full dataframe (df)")
            return prophet_forecast(df, len(test) if test is not None else horizon)

        # ----------------------
        # ML MODEL
        # ----------------------
        elif model_name == "Gradient Boosting":

            # Future forecast case
            if test is None:
                return np.repeat(train.mean(), horizon)

            if X_train is None or X_test is None:
                raise ValueError("Gradient Boosting needs X_train and X_test")

            return gradient_boosting_model(X_train, train.values, X_test)

        # ----------------------
        # DEFAULT
        # ----------------------
        else:
            return np.repeat(train.mean(), len(test))

    except Exception as e:
        print(f"⚠️ {model_name} failed:", e)

        if test is not None:
            return np.repeat(train.mean(), len(test))
        else:
            return np.repeat(train.mean(), horizon)
 