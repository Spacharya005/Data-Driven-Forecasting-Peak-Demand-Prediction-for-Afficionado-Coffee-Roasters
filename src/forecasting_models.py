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

    if len(train) == 0:
        return np.zeros(len(test))

    last_value = train.iloc[-1]
    return np.repeat(last_value, len(test))


def gradient_boosting_model(X_train, y_train, X_test):

    if X_train.shape[0] != len(y_train):
        raise ValueError("Mismatch between X_train and y_train")

    if X_test.shape[1] != X_train.shape[1]:
        raise ValueError("Feature mismatch between train and test")

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )

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

    model = Prophet( daily_seasonality=True, 
                    weekly_seasonality=True)
    model.fit(prophet_df)

    # 🔥 Detect frequency automatically
    freq = pd.infer_freq(prophet_df['ds'])

    if freq is None:
        freq = 'H'  # fallback

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

            # TRAIN
            model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            )
            model.fit(X_train, train.values)

            # 🔥 CASE 1: Normal test prediction
            if X_test is not None:
                return model.predict(X_test)

            # 🔥 CASE 2: Future forecasting (AUTO REGRESSIVE)
            if horizon is not None:
                preds = []
                history = list(train.values)

                for i in range(horizon):
                    # create simple lag features manually
                    lag_1 = history[-1]
                    lag_24 = history[-24] if len(history) >= 24 else lag_1

                    row = np.array([[lag_1, lag_24]])

                    pred = model.predict(row)[0]
                    preds.append(pred)

                    history.append(pred)

                return np.array(preds)

    except Exception as e:
        print(f"❌ {model_name} FAILED → using fallback")
        print("Error:", e)

        if test is not None:
            return np.repeat(train.mean(), len(test))
        else:
            return np.repeat(train.mean(), horizon)
 