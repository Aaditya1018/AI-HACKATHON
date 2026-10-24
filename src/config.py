"""
Configuration settings for Campus Energy Consumption Forecast.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "campus_energy_dataset.csv"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Chronological Partitioning (Strict temporal order without lookahead)
TRAIN_RATIO = 0.70       # Initial 70% of chronological records (~12 months)
VAL_RATIO = 0.15         # Next 15% for model validation & early stopping (~3 months)
TEST_RATIO = 0.15        # Final 15% out-of-sample holdout test (~3 months)

# Operational Forecast Horizon
FORECAST_HORIZON_HOURS = 24

# Operational Alert Thresholds
PEAK_PERCENTILE = 85.0   # Top 15% demand flagged for peak shaving
CRITICAL_PERCENTILE = 95.0

RANDOM_SEED = 42
