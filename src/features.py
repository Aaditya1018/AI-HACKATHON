"""
Feature Engineering module for 24-hour campus energy forecasting.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd


def extract_features(df: pd.DataFrame, target_col: str = "electricity_consumption_kwh") -> Tuple[pd.DataFrame, List[str]]:
    """
    Builds temporal, weather-thermal, occupancy, and shifted autoregressive lag features.
    
    Zero Lookahead Rule:
    When issuing a 24-hour forecast from origin t:
    Only observations up to origin t are known.
    Therefore, all historical lag features are shifted by >= 24 hours.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week_num"] = df["timestamp"].dt.dayofweek
    df["day_of_year"] = df["timestamp"].dt.dayofyear

    # 1. Cyclic Temporal Transforms (continuous circle)
    df["hour_sin"] = np.sin(2.0 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2.0 * np.pi * df["day_of_week_num"] / 7.0)
    df["dow_cos"] = np.cos(2.0 * np.pi * df["day_of_week_num"] / 7.0)
    df["doy_sin"] = np.sin(2.0 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2.0 * np.pi * df["day_of_year"] / 365.25)

    # 2. Thermal Indicators
    df["cooling_degree_demand"] = np.maximum(df["temperature_c"] - 18.0, 0.0)
    df["heating_degree_demand"] = np.maximum(14.0 - df["temperature_c"], 0.0)
    df["temp_gradient_1h"] = df["temperature_c"].diff(1).fillna(0.0)
    df["temp_roll_6h"] = df["temperature_c"].rolling(window=6, min_periods=1).mean()
    df["temp_roll_24h"] = df["temperature_c"].rolling(window=24, min_periods=1).mean()

    # 3. Occupancy Interactions
    df["occ_cooling_interaction"] = df["is_occupied"] * df["cooling_degree_demand"]
    df["occ_heating_interaction"] = df["is_occupied"] * df["heating_degree_demand"]

    # 4. Autoregressive Lags (>= 24h prior)
    df["lag_24h"] = df[target_col].shift(24)
    df["lag_48h"] = df[target_col].shift(48)
    df["lag_168h"] = df[target_col].shift(168)

    # Shifted rolling statistics
    shifted = df[target_col].shift(24)
    df["roll_mean_24h_lag24"] = shifted.rolling(window=24, min_periods=1).mean()
    df["roll_max_24h_lag24"] = shifted.rolling(window=24, min_periods=1).max()
    df["roll_min_24h_lag24"] = shifted.rolling(window=24, min_periods=1).min()
    df["roll_std_24h_lag24"] = shifted.rolling(window=24, min_periods=2).std().fillna(0.0)

    # Trim 168h burn-in window
    df = df.iloc[168:].reset_index(drop=True)

    feature_cols = [
        "hour",
        "day_of_week_num",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "doy_sin",
        "doy_cos",
        "temperature_c",
        "cooling_degree_demand",
        "heating_degree_demand",
        "temp_gradient_1h",
        "temp_roll_6h",
        "temp_roll_24h",
        "is_occupied",
        "occ_cooling_interaction",
        "occ_heating_interaction",
        "lag_24h",
        "lag_48h",
        "lag_168h",
        "roll_mean_24h_lag24",
        "roll_max_24h_lag24",
        "roll_min_24h_lag24",
        "roll_std_24h_lag24",
    ]

    return df, feature_cols
