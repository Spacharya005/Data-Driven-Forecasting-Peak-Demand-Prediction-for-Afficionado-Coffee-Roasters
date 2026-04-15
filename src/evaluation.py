import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_all(y_true, predictions_dict):

    results = []

    for model, preds in predictions_dict.items():

        mae = mean_absolute_error(y_true, preds)
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        mape = np.mean(
            np.abs((y_true - preds) / np.where(y_true == 0, 1, y_true))
        ) * 100

        results.append({
            "Model": model,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape
        })

    return pd.DataFrame(results).sort_values("RMSE")

def peak_error_rate(y_true, y_pred, threshold=None):
    if threshold is None:
        threshold = np.percentile(y_true, 90)

    peak_mask = y_true > threshold
    return np.mean(np.abs(y_true[peak_mask] - y_pred[peak_mask]))