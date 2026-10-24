"""
Visualization module for Campus Energy Forecasting.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.config import FIGURES_DIR

COLOR_ACTUAL = "#1a1a1a"
COLOR_FORECAST = "#0b60b0"
COLOR_BASELINE = "#8c8c8c"
COLOR_ALERT = "#c0392b"
COLOR_TEMP = "#d35400"
COLOR_OCC = "#27ae60"


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#d0d0d0",
        "axes.linewidth": 0.8,
        "grid.color": "#ebebeb",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "figure.autolayout": True,
    })


def plot_forecast_24h(
    df_24h: pd.DataFrame,
    peak_threshold_kwh: float,
    output_path: Optional[Path] = None,
) -> Path:
    """Plots 24-hour metered load vs AI forecast vs persistence baseline."""
    setup_style()
    output_path = output_path or (FIGURES_DIR / "forecast_24h_comparison.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]}
    )
    times = df_24h["timestamp"]

    # Top Panel: Electric Load
    ax1.plot(
        times, df_24h["electricity_consumption_kwh"],
        label="Actual Metered Consumption (kWh)",
        color=COLOR_ACTUAL, linewidth=2.2, marker="o", markersize=4, zorder=4
    )
    ax1.plot(
        times, df_24h["predicted_kwh"],
        label="AI 24h Advance Forecast (XGBoost)",
        color=COLOR_FORECAST, linewidth=2.4, marker="s", markersize=4, zorder=5
    )
    ax1.plot(
        times, df_24h["baseline_24h_kwh"],
        label="Baseline: Same Hour Previous Day (t-24)",
        color=COLOR_BASELINE, linewidth=1.8, linestyle="--", zorder=3
    )

    ax1.axhline(
        peak_threshold_kwh,
        color=COLOR_ALERT, linestyle=":", linewidth=1.8,
        label=f"Operational Peak Alert Threshold ({peak_threshold_kwh:.0f} kWh)",
        zorder=2
    )

    # Shaded alert region
    peak_mask = df_24h["predicted_kwh"] >= peak_threshold_kwh
    if peak_mask.any():
        for i in range(len(df_24h)):
            if peak_mask.iloc[i]:
                ax1.axvspan(
                    times.iloc[max(0, i - 1)], times.iloc[min(len(df_24h) - 1, i + 1)],
                    color=COLOR_ALERT, alpha=0.08, zorder=1
                )

    ax1.set_ylabel("Electricity Demand (kWh)", fontsize=11, fontweight="bold")
    ax1.set_title(
        f"Campus Energy Forecast vs. Actual Metered Load & Persistence Baseline\n"
        f"Forecast Horizon: {times.iloc[0].strftime('%Y-%m-%d %H:%M')} to {times.iloc[-1].strftime('%Y-%m-%d %H:%M')}",
        fontsize=12, fontweight="bold", pad=10
    )
    ax1.grid(True)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#e0e0e0", fontsize=9.5)

    # Bottom Panel: Drivers (Dry-bulb temp & Occupancy)
    ax2_twin = ax2.twinx()
    ax2.plot(times, df_24h["temperature_c"], color=COLOR_TEMP, linewidth=2.0, label="Ambient Temp (°C)")
    ax2.set_ylabel("Ambient Temp (°C)", color=COLOR_TEMP, fontsize=10, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=COLOR_TEMP)

    ax2_twin.fill_between(times, 0, df_24h["is_occupied"], color=COLOR_OCC, alpha=0.25, step="mid")
    ax2_twin.set_ylabel("Occupancy", color=COLOR_OCC, fontsize=10, fontweight="bold")
    ax2_twin.set_ylim(-0.1, 1.3)
    ax2_twin.set_yticks([0, 1])
    ax2_twin.set_yticklabels(["Unoccupied", "Occupied"])
    ax2_twin.tick_params(axis="y", labelcolor=COLOR_OCC)

    ax2.set_xlabel("Time of Day", fontsize=10, fontweight="bold")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%b %d"))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax2.grid(True)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_benchmark_metrics(
    df_bench: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Path:
    """Side-by-side comparative bar chart of model errors."""
    setup_style()
    output_path = output_path or (FIGURES_DIR / "model_benchmark_comparison.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    models = df_bench["Model Architecture"].tolist()
    y_pos = np.arange(len(models))
    width = 0.35

    # Panel 1: MAE & RMSE
    ax1.barh(y_pos - width/2, df_bench["MAE (kWh)"], width, label="MAE (kWh)", color="#2b5c8f")
    ax1.barh(y_pos + width/2, df_bench["RMSE (kWh)"], width, label="RMSE (kWh)", color="#d4a276")

    for i, v in enumerate(df_bench["MAE (kWh)"]):
        ax1.text(v + 1.0, i - width/2, f"{v:.1f}", va="center", fontsize=9, fontweight="bold")
    for i, v in enumerate(df_bench["RMSE (kWh)"]):
        ax1.text(v + 1.0, i + width/2, f"{v:.1f}", va="center", fontsize=9, fontweight="bold")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(models, fontsize=9.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("Error (kWh) — Lower is Better", fontsize=10, fontweight="bold")
    ax1.set_title("Overall Holdout Test Error (MAE & RMSE)", fontsize=11, fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(True, axis="x")

    # Panel 2: MAPE (%) & Peak-Hour Error
    ax2.barh(y_pos - width/2, df_bench["MAPE (%)"], width, label="MAPE (%)", color="#6a4c93")
    ax2.barh(y_pos + width/2, df_bench["Peak-Hour MAE (kWh)"], width, label="Peak-Hour MAE (kWh)", color=COLOR_ALERT)

    for i, v in enumerate(df_bench["MAPE (%)"]):
        ax2.text(v + 0.3, i - width/2, f"{v:.1f}%", va="center", fontsize=9, fontweight="bold")
    for i, v in enumerate(df_bench["Peak-Hour MAE (kWh)"]):
        if not np.isnan(v):
            ax2.text(v + 0.8, i + width/2, f"{v:.1f}", va="center", fontsize=9, fontweight="bold")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlabel("Error Magnitude", fontsize=10, fontweight="bold")
    ax2.set_title("Relative Percentage & Peak-Hour MAE", fontsize=11, fontweight="bold")
    ax2.legend(loc="lower right")
    ax2.grid(True, axis="x")

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_feature_importance(
    df_imp: pd.DataFrame,
    top_n: int = 12,
    output_path: Optional[Path] = None,
) -> Path:
    """Plots top predictive feature importances."""
    setup_style()
    output_path = output_path or (FIGURES_DIR / "feature_importance.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_top = df_imp.head(top_n).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(df_top["feature"], df_top["importance"], color="#1f4e79", edgecolor="#0e2840")

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", fontsize=9)

    ax.set_xlabel("Predictive Importance (Relative Gain)", fontsize=10, fontweight="bold")
    ax.set_title(f"Top {top_n} Determinants of Campus Electricity Consumption", fontsize=12, fontweight="bold")
    ax.grid(True, axis="x")

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path
