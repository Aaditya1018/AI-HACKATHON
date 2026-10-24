"""
Unit tests for chronological partitioning, leak-free lag generation, and models.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.config import DATA_PATH
from src.data_loader import load_dataset, split_chronological, prepare_data
from src.models import PersistenceBaseline24h, RidgeEnergyForecaster
from src.evaluator import calculate_metrics
from src.advisory import CampusAdvisoryEngine


def test_chronological_ordering_zero_leakage():
    """Validates that train, validation, and test sets are strictly chronological."""
    train_df, val_df, test_df, feature_cols, target_col = prepare_data(DATA_PATH)

    assert train_df["timestamp"].max() < val_df["timestamp"].min()
    assert val_df["timestamp"].max() < test_df["timestamp"].min()


def test_shortest_lag_is_at_least_24_hours():
    """Validates that no lag features < 24h exist in the feature set."""
    train_df, val_df, test_df, feature_cols, target_col = prepare_data(DATA_PATH)

    for forbidden in ["lag_1h", "lag_2h", "lag_4h", "lag_12h"]:
        assert forbidden not in feature_cols, f"Lookahead leak: {forbidden} present!"

    assert "lag_24h" in feature_cols
    assert "lag_48h" in feature_cols
    assert "lag_168h" in feature_cols


def test_previous_day_persistence_exact_match():
    """Validates that PersistenceBaseline24h returns exact lag_24h series."""
    model = PersistenceBaseline24h()
    test_features = pd.DataFrame({"lag_24h": [120.0, 185.5, 230.1]})
    preds = model.predict(test_features)
    np.testing.assert_array_equal(preds, test_features["lag_24h"].values)


def test_error_metrics_calculation():
    """Validates MAE, RMSE, and MAPE calculations on synthetic vectors."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 300.0])

    res = calculate_metrics(y_true, y_pred, peak_threshold=250.0)
    assert res["MAE (kWh)"] == pytest.approx(6.67, 0.05)
    assert res["RMSE (kWh)"] == pytest.approx(8.16, 0.05)
    assert res["MAPE (%)"] == pytest.approx(5.0, 0.05)
    assert res["Peak-Hour MAE (kWh)"] == 0.0


def test_advisory_alert_trigger():
    """Validates that advisory engine issues peak alerts when loads exceed threshold."""
    engine = CampusAdvisoryEngine(peak_threshold_kwh=400.0, critical_threshold_kwh=500.0)
    times = pd.date_range("2025-05-19 00:00", periods=24, freq="h")
    preds = np.full(24, 250.0)
    preds[12:17] = 480.0

    df_24h = pd.DataFrame({
        "timestamp": times,
        "predicted_kwh": preds,
        "temperature_c": np.full(24, 29.0),
        "is_occupied": np.full(24, 1),
        "hour": np.arange(24),
    })

    out = engine.assess_24h_window(df_24h)
    assert out["n_flagged_hours"] == 5
    assert len(out["action_plan"]) > 0
    assert "OPERATIONAL NOTICE" in out["governance_disclaimer"]
