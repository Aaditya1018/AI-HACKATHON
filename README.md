# Campus Energy Consumption 24-Hour Forecasting & Peak Advisory System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI prototype designed for campus facility managers to **forecast electricity consumption for the next 24 hours**, **benchmark predictions against traditional persistence baselines**, **flag projected peak demand periods**, and **recommend practical energy-saving actions** without automated equipment control.

---

## 1. Scenario & Challenge Overview

A campus building manager records hourly:
- Electricity consumption ($\text{kWh}$)
- Outdoor ambient temperature ($^\circ\text{C}$)
- Day of the week
- Whether the building is occupied (binary indicator)

The manager wants to identify likely high-consumption periods and plan energy-saving actions.

### Requirements Addressed:
1. **Original Dataset:** Ingests recorded hourly campus energy telemetry (`data/campus_energy_dataset.csv`, 13,128 hourly records).
2. **Chronological Splitting:** Enforces strict forward-in-time train/val/test periods without mixing future data into training.
3. **Baseline Comparison:** Benchmarks the AI model against the building manager's traditional persistence heuristic—the **previous day's same-hour consumption** ($y_t = y_{t-24}$) and weekly persistence ($y_t = y_{t-168}$).
4. **Peak Period Advisory:** Detects hours projected to exceed operational thresholds (85th percentile: $453.4\text{ kWh}$) and issues practical operational checks (pre-cooling, static pressure resets, zone setbacks) without claiming direct equipment control.
5. **Data Limitations:** Documents weather forecast uncertainty, thermal inertia lag, and sensor constraints.

---

## 2. Quantitative Model Benchmark (Holdout Test Set)

Evaluated across **1,944 out-of-sample hours** (April 11, 2025 to June 30, 2025) without random shuffling:

| Model Architecture | MAE (kWh) | RMSE (kWh) | MAPE (%) | $R^2$ Score | Peak-Hour MAE (kWh) | Error Reduction vs. Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline: Same Hour Previous Day (t-24)** | **64.24** | **91.12** | **21.08%** | **0.6855** | **90.40** | *Reference (0.0%)* |
| **Baseline: Same Hour Previous Week (t-168)** | 36.71 | 50.02 | 12.62% | 0.9052 | 50.93 | +42.9% |
| **Baseline: Historical Day-Hour Mean** | 68.69 | 88.90 | 22.80% | 0.7007 | 136.06 | -6.9% |
| **Ridge Linear Regression** | 26.75 | 34.24 | 9.02% | 0.9556 | 42.98 | +58.4% |
| **HistGradientBoosting Forecaster** | 9.98 | 13.59 | 3.27% | 0.9930 | 15.56 | +84.5% |
| **XGBoost Energy Forecaster** | **10.00** | **13.76** | **3.26%** | **0.9928** | **16.44** | **+84.4%** |

### Benchmark Takeaways:
- **84.4% Error Reduction:** XGBoost and HistGBDT reduce forecast error from **64.24 kWh** (21.08% MAPE) down to **10.00 kWh** (3.26% MAPE).
- **Peak-Hour Accuracy:** During top 15% high-demand hours, the AI model achieves an MAE of **16.44 kWh** compared to **90.40 kWh** for the previous-day baseline (+81.8% improvement).
- **Why Previous-Day Persistence Fails:** On Monday mornings, previous-day persistence projects Sunday's quiet unoccupied base load onto an active academic instructional day, causing severe under-forecasting.

---

## 3. Operational Advisory & Peak Action Plan

When forecasted electrical demand exceeds the 85th percentile threshold ($453.4\text{ kWh}$), the advisory engine issues prioritized, non-invasive operational checks:

1. **HVAC Thermal Pre-Cooling (High Priority):**  
   Sub-cool primary lecture halls and high-thermal-mass concrete zones by $1.0^\circ\text{C}$ to $1.5^\circ\text{C}$ 2–3 hours prior to the peak window (e.g. 08:00–11:00). This stores thermal cooling capacity in the building envelope and reduces chiller electrical demand during high-tariff afternoon hours.
2. **AHU Fan Static Pressure Reset (High Priority):**  
   Reset Air Handling Unit (AHU) duct static pressure setpoint by $-0.15\text{ in. w.g.}$ to allow variable frequency drives (VFDs) to ramp down safely without starving terminal VAV boxes.
3. **Zone Deadband Offset (Medium Priority):**  
   Temporarily widen temperature deadbands by $+1.0^\circ\text{C}$ in non-critical spaces (atria, administrative suites, corridors) during flagged peak hours.
4. **Peripheral Lighting & Computer Lab Sweep (Medium Priority):**  
   Trigger remote low-power sleep state on instructional computer labs and sweep unoccupied peripheral corridors.
5. **Research Lab Fume Hood Sashes (Safety First):**  
   Encourage research staff to lower variable air volume (VAV) fume hood sashes when unattended to reduce exhaust makeup air volume while preserving all laboratory safety air change requirements.

> **Operational Boundary Notice:** This prototype is strictly an **Advisory Decision Support System**. It does **not** assert direct digital control or autonomously override Building Automation System (BAS), SCADA, or BACnet field controllers.

---

## 4. Repository Structure

```
campus_energy_forecast/
├── README.md                           # System overview and benchmark guide
├── requirements.txt                    # Pinned dependencies
├── Makefile                            # Reproducible automation targets
├── run_pipeline.py                     # CLI pipeline runner
├── run_pipeline.sh                     # 1-click pipeline shell script
├── run_dashboard.sh                    # 1-click dashboard shell script
├── app.py                              # Interactive Streamlit operations console
├── notebooks/
│   └── campus_energy_forecast_analysis.ipynb # Executable research walkthrough
├── data/
│   └── campus_energy_dataset.csv       # Recorded hourly telemetry dataset (13,128 records)
├── src/
│   ├── __init__.py
│   ├── config.py                       # Paths, hyperparameters, percentile thresholds
│   ├── features.py                     # Cyclic encoding, CDD/HDD, shifted lags
│   ├── data_loader.py                  # Chronological train/val/test partitioning
│   ├── models.py                       # Baselines (24h, 168h, Mean), Ridge, HistGBDT, XGBoost
│   ├── evaluator.py                    # MAE, RMSE, MAPE, R2, Peak-Hour MAE
│   ├── advisory.py                     # Peak detector & recommendation engine
│   └── visualizer.py                   # Publication figure generation
├── reports/
│   ├── engineering_technical_report.md # Comprehensive engineering report
│   ├── model_benchmark_results.csv     # Tabular benchmark metrics
│   ├── feature_importances.csv         # Feature weights export
│   └── figures/                        # Generated publication charts:
│       ├── forecast_24h_comparison.png
│       ├── model_benchmark_comparison.png
│       └── feature_importance.png
└── tests/
    ├── __init__.py
    └── test_pipeline.py                # Unit tests (chronological split, lags, metrics)
```

---

## 5. Quick Start & Execution Guide

### Option 1: Run the End-to-End Pipeline (CLI)
```bash
python3 run_pipeline.py
# Or: ./run_pipeline.sh
```

### Option 2: Run the Unit Tests
```bash
pytest -v
```

### Option 3: Launch the Interactive Operations Console
```bash
streamlit run app.py
# Or: ./run_dashboard.sh
```

### Option 4: Open the Jupyter Notebook Walkthrough
```bash
jupyter notebook notebooks/campus_energy_forecast_analysis.ipynb
```
