"""
Evaluation metrics and benchmark comparison routines.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    peak_threshold: float = None,
) -> Dict[str, float]:
    """Calculates MAE, RMSE, MAPE, R2, and Peak-Hour MAE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    valid_mask = y_true > 1e-2
    mape = float(np.mean(np.abs((y_true[valid_mask] - y_pred[valid_mask]) / y_true[valid_mask])) * 100.0)
    r2 = float(r2_score(y_true, y_pred))

    out = {
        "MAE (kWh)": round(mae, 2),
        "RMSE (kWh)": round(rmse, 2),
        "MAPE (%)": round(mape, 2),
        "R2 Score": round(r2, 4),
    }

    if peak_threshold is not None:
        peak_mask = y_true >= peak_threshold
        if np.any(peak_mask):
            p_mae = float(mean_absolute_error(y_true[peak_mask], y_pred[peak_mask]))
            out["Peak-Hour MAE (kWh)"] = round(p_mae, 2)
        else:
            out["Peak-Hour MAE (kWh)"] = np.nan

    return out


def evaluate_models_comparison(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    peak_threshold: float,
    baseline_name: str = "Baseline (Same Hour Previous Day)",
) -> pd.DataFrame:
    """Evaluates all candidate models and baselines on the holdout test set."""
    records = []
    base_mae = None

    for name, model in models.items():
        preds = model.predict(X_test)
        m = calculate_metrics(y_test.values, preds, peak_threshold=peak_threshold)
        m["Model Architecture"] = name
        records.append(m)

        if name == baseline_name:
            base_mae = m["MAE (kWh)"]

    df = pd.DataFrame(records)
    cols = ["Model Architecture", "MAE (kWh)", "RMSE (kWh)", "MAPE (%)", "R2 Score", "Peak-Hour MAE (kWh)"]
    df = df[cols]

    if base_mae is not None and base_mae > 0:
        df["MAE Reduction vs. Baseline (%)"] = (
            (base_mae - df["MAE (kWh)"]) / base_mae * 100.0
        ).round(1)

    return df
