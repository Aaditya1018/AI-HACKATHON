"""
Forecasting models: Baselines and Machine Learning forecasters.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
import xgboost as xgb

from src.config import RANDOM_SEED


class PersistenceBaseline24h:
    """Predicts that consumption at hour t equals consumption at same hour yesterday (t - 24)."""

    def __init__(self):
        self.name = "Baseline (Same Hour Previous Day)"

    def fit(self, X: pd.DataFrame, y: pd.Series):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["lag_24h"].values


class PersistenceBaseline168h:
    """Predicts that consumption at hour t equals consumption at same hour last week (t - 168)."""

    def __init__(self):
        self.name = "Baseline (Same Hour Previous Week)"

    def fit(self, X: pd.DataFrame, y: pd.Series):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["lag_168h"].values


class HistoricalMeanBaseline:
    """Stratified mean by (day_of_week_num, hour)."""

    def __init__(self):
        self.name = "Baseline (Historical Day-Hour Mean)"
        self.lookup_ = {}
        self.default_ = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        df = X[["day_of_week_num", "hour"]].copy()
        df["target"] = y.values
        self.lookup_ = df.groupby(["day_of_week_num", "hour"])["target"].mean().to_dict()
        self.default_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([self.lookup_.get((int(r["day_of_week_num"]), int(r["hour"])), self.default_) for _, r in X.iterrows()])


class RidgeEnergyForecaster:
    """L2-regularized linear model with standard feature scaling."""

    def __init__(self, alpha: float = 12.0):
        self.name = "Ridge Linear Regression"
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, random_state=RANDOM_SEED)),
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)


class HistGradientBoostingForecaster:
    """Histogram-based Gradient Boosted Trees."""

    def __init__(self, max_iter: int = 350, learning_rate: float = 0.04, max_depth: int = 6):
        self.name = "HistGradientBoosting Forecaster"
        self.model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=RANDOM_SEED,
            early_stopping=True,
            n_iter_no_change=25,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


class XGBoostEnergyForecaster:
    """Extreme Gradient Boosted Trees with validation early stopping."""

    def __init__(
        self,
        n_estimators: int = 450,
        learning_rate: float = 0.03,
        max_depth: int = 5,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
    ):
        self.name = "XGBoost Energy Forecaster"
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            early_stopping_rounds=30,
            eval_metric="rmse",
        )
        self.feature_names_ = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ):
        self.feature_names_ = list(X_train.columns)
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def get_feature_importances(self) -> pd.DataFrame:
        if self.feature_names_ is None:
            raise ValueError("Model must be fitted before extracting feature importances.")
        return pd.DataFrame({
            "feature": self.feature_names_,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
