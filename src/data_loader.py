"""
Data loading and strict chronological partitioning.
"""

from pathlib import Path
from typing import Tuple, List
import pandas as pd
from src.config import DATA_PATH, TRAIN_RATIO, VAL_RATIO
from src.features import extract_features


def load_dataset(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    """Loads the campus energy dataset."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}.")
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def split_chronological(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partitions the dataset chronologically into Train, Validation, and Test sets."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)

    # Invariant assertion
    assert train_df["timestamp"].max() < val_df["timestamp"].min(), "Leakage: Train overlaps Validation!"
    assert val_df["timestamp"].max() < test_df["timestamp"].min(), "Leakage: Validation overlaps Test!"

    return train_df, val_df, test_df


def prepare_data(csv_path: Path = DATA_PATH):
    """Full preparation pipeline: load dataset, extract features, and chronologically partition."""
    raw_df = load_dataset(csv_path)
    target_col = "electricity_consumption_kwh"
    feat_df, feature_cols = extract_features(raw_df, target_col=target_col)
    train_df, val_df, test_df = split_chronological(feat_df)
    return train_df, val_df, test_df, feature_cols, target_col
