"""
Main Execution Pipeline: Campus Energy Consumption Forecast.

Runs:
1. Ingestion of campus hourly energy dataset
2. Strict zero-lookahead chronological partitioning
3. Fitting baseline heuristics & ML forecasters
4. Holdout test set benchmark evaluation
5. 24-hour advance operational forecast & peak demand advisory analysis
6. Automated generation of publication figures & summary reports
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.config import DATA_PATH, REPORTS_DIR, FIGURES_DIR, PEAK_PERCENTILE, CRITICAL_PERCENTILE
from src.data_loader import prepare_data
from src.models import (
    PersistenceBaseline24h,
    PersistenceBaseline168h,
    HistoricalMeanBaseline,
    RidgeEnergyForecaster,
    HistGradientBoostingForecaster,
    XGBoostEnergyForecaster,
)
from src.evaluator import evaluate_models_comparison
from src.advisory import CampusAdvisoryEngine
from src.visualizer import (
    plot_forecast_24h,
    plot_benchmark_metrics,
    plot_feature_importance,
)


def main():
    print("=" * 82)
    print(" CAMPUS ENERGY CONSUMPTION 24-HOUR FORECASTING & PEAK ADVISORY SYSTEM")
    print("=" * 82)

    # 1. Load dataset & chronological partition
    print("\n[1/5] Ingesting dataset and preparing chronological partitions...")
    train_df, val_df, test_df, feature_cols, target_col = prepare_data(DATA_PATH)

    print(f"  • Training set:   {len(train_df):,} hours ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
    print(f"  • Validation set: {len(val_df):,} hours ({val_df['timestamp'].min()} to {val_df['timestamp'].max()})")
    print(f"  • Holdout Test:   {len(test_df):,} hours ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")
    print(f"  • Features:       {len(feature_cols)} (Cyclic, Weather Degree Days, Lags >= 24h)")

    peak_threshold = float(np.percentile(train_df[target_col], PEAK_PERCENTILE))
    critical_threshold = float(np.percentile(train_df[target_col], CRITICAL_PERCENTILE))
    print(f"\n  • Operational Peak Threshold (85th percentile): {peak_threshold:.1f} kWh")
    print(f"  • Critical Peak Threshold    (95th percentile): {critical_threshold:.1f} kWh")

    # 2. Fit models
    print("\n[2/5] Fitting candidate architectures & persistence baselines...")
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val, y_val = val_df[feature_cols], val_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    models = {
        "Baseline (Same Hour Previous Day)": PersistenceBaseline24h(),
        "Baseline (Same Hour Previous Week)": PersistenceBaseline168h(),
        "Baseline (Historical Day-Hour Mean)": HistoricalMeanBaseline(),
        "Ridge Linear Regression": RidgeEnergyForecaster(alpha=12.0),
        "HistGradientBoosting Forecaster": HistGradientBoostingForecaster(),
        "XGBoost Energy Forecaster": XGBoostEnergyForecaster(n_estimators=450, learning_rate=0.03, max_depth=5),
    }

    models["Baseline (Same Hour Previous Day)"].fit(X_train, y_train)
    models["Baseline (Same Hour Previous Week)"].fit(X_train, y_train)
    models["Baseline (Historical Day-Hour Mean)"].fit(X_train, y_train)
    models["Ridge Linear Regression"].fit(X_train, y_train)
    models["HistGradientBoosting Forecaster"].fit(X_train, y_train)
    models["XGBoost Energy Forecaster"].fit(X_train, y_train, X_val, y_val)
    print("  ✓ All models fitted successfully.")

    # 3. Benchmark evaluation
    print("\n[3/5] Evaluating out-of-sample holdout test partition...")
    df_bench = evaluate_models_comparison(
        models=models,
        X_test=X_test,
        y_test=y_test,
        peak_threshold=peak_threshold,
        baseline_name="Baseline (Same Hour Previous Day)",
    )

    print("\n" + "-" * 82)
    print(df_bench.to_string(index=False))
    print("-" * 82)

    # 4. Extract representative 24h operational day
    print("\n[4/5] Running 24-hour advance forecast and operational advisory checks...")
    peak_date_matches = test_df[test_df["timestamp"] == "2025-05-19 00:00:00"].index
    start_idx = peak_date_matches[0] if len(peak_date_matches) > 0 else 0
    df_24h = test_df.iloc[start_idx : start_idx + 24].copy().reset_index(drop=True)

    df_24h["predicted_kwh"] = models["XGBoost Energy Forecaster"].predict(df_24h[feature_cols])
    df_24h["baseline_24h_kwh"] = models["Baseline (Same Hour Previous Day)"].predict(df_24h[feature_cols])

    engine = CampusAdvisoryEngine(
        peak_threshold_kwh=peak_threshold,
        critical_threshold_kwh=critical_threshold,
    )
    advisory_res = engine.assess_24h_window(df_24h, pred_col="predicted_kwh")

    print(f"  • Forecast Horizon Date:       {df_24h['timestamp'].iloc[0].strftime('%Y-%m-%d')} (Summer Heatwave Window)")
    print(f"  • Projected 24h Total Energy:  {advisory_res['total_24h_kwh']:,} kWh")
    print(f"  • Projected Maximum Demand:    {advisory_res['peak_demand_kwh']} kWh (at {advisory_res['peak_time']})")
    print(f"  • Flagged Peak Demand Hours:   {advisory_res['n_flagged_hours']} hours ({', '.join(advisory_res['flagged_hours'])})")
    print(f"  • Dominant Demand Drivers:     {'; '.join(advisory_res['peak_drivers'])}")
    print("\n  OPERATIONAL FACILITY ACTION PLAN:")
    for i, action in enumerate(advisory_res["action_plan"], 1):
        print(f"    {i}. [{action['priority']}] {action['subsystem']}: {action['recommendation']}")
        print(f"       -> {action['technical_details']}")

    # 5. Export figures and documentation
    print("\n[5/5] Generating publication-grade figures and exporting summaries...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    p1 = plot_forecast_24h(df_24h, peak_threshold_kwh=peak_threshold)
    p2 = plot_benchmark_metrics(df_bench)
    feat_imp = models["XGBoost Energy Forecaster"].get_feature_importances()
    p3 = plot_feature_importance(feat_imp)

    print(f"  ✓ Saved 24h forecast chart: {p1.name}")
    print(f"  ✓ Saved benchmark chart:    {p2.name}")
    print(f"  ✓ Saved feature weights:    {p3.name}")

    df_bench.to_csv(REPORTS_DIR / "model_benchmark_results.csv", index=False)
    feat_imp.to_csv(REPORTS_DIR / "feature_importances.csv", index=False)
    df_24h.to_csv(REPORTS_DIR / "operational_24h_forecast.csv", index=False)

    with open(REPORTS_DIR / "advisory_action_plan.json", "w") as f:
        json.dump(
            {k: v for k, v in advisory_res.items() if k != "annotated_df"},
            f, indent=2
        )

    print("\n" + "=" * 82)
    print(" PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 82)


if __name__ == "__main__":
    main()
