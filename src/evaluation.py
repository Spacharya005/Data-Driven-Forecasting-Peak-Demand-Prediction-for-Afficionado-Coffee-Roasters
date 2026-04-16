import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_all(y_true, predictions_dict):

    results = []

    for model, preds in predictions_dict.items():

        y_true_arr = np.array(y_true)
        preds_arr = np.array(preds)

        mae = mean_absolute_error(y_true_arr, preds_arr)
        rmse = np.sqrt(mean_squared_error(y_true_arr, preds_arr))
        mape = np.mean(
            np.abs((y_true_arr - preds_arr) / np.where(y_true_arr == 0, 1, y_true_arr))
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