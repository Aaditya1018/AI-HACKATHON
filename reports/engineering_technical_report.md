# Technical Engineering Report: Campus Energy Consumption 24-Hour Forecast & Peak Advisory

**System Objective:** 24-Hour Advance Electricity Consumption Forecast & Peak Advisory  
**Target Scenario:** Campus Building Energy Operations  
**Evaluation Benchmark Period:** April 11, 2025 to June 30, 2025 (1,944 Holdout Hours)  

---

## 1. Executive Summary & Challenge Overview

A campus building manager records hourly:
- Electricity consumption ($\text{kWh}$)
- Outdoor ambient dry-bulb temperature ($^\circ\text{C}$)
- Day of the week
- Building occupancy status (binary indicator)

The manager wants to identify likely high-consumption periods and plan energy-saving actions. This prototype addresses the challenge by forecasting electricity use for the upcoming 24 hours, benchmarking against standard persistence heuristics, and flagging projected peak demand windows with actionable checklists.

### Key Deliverables Addressed:
1. **Chronological Splitting:** Enforces strict forward-in-time train/val/test splits without mixing future data into training.
2. **Baseline Benchmarking:** Compares against the building manager's traditional persistence heuristic—the **previous day's same-hour consumption** ($y_t = y_{t-24}$) and weekly persistence ($y_t = y_{t-168}$).
3. **Peak Flagging & Advisory:** Detects hours projected to exceed operational thresholds (85th percentile: $453.4\text{ kWh}$) and issues practical, human-in-the-loop facility checks (pre-cooling, static pressure resets, zone setbacks).
4. **Data Limitations:** Documents weather forecast propagation, occupancy proxy limits, and building thermal mass lag.

---

## 2. Quantitative Model Benchmark (Holdout Test Set)

Evaluated across **1,944 out-of-sample hours** without random shuffling:

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
- **Why Persistence Fails:** Monday morning forecasts based on Sunday readings produce severe under-forecasting because Sunday's quiet base load (~140 kWh) fails to account for active classrooms, arriving students, and mechanical chiller staging.

---

## 3. Operational 24-Hour Horizon & Facility Action Plan

During a high-demand spring/summer weekday (**Monday, May 19, 2025**):
- **Projected Total 24h Consumption:** 7,793.8 kWh
- **Projected Maximum Demand:** 570.9 kWh at 14:00 (exceeding the 453.4 kWh threshold)
- **Flagged Peak Window:** 6 consecutive hours (11:00 to 16:00)
- **Primary Drivers:** Heavy instructional lecture occupancy coupled with 31°C ambient dry-bulb cooling load.

### Recommended Action Plan for Facility Engineers:
1. **HVAC Thermal Pre-Cooling (08:00 – 11:00):**  
   Sub-cool primary lecture halls and high-thermal-mass concrete zones by $1.0^\circ\text{C}$ to $1.5^\circ\text{C}$ between 08:00 and 11:00. This stores thermal cooling capacity in the building envelope and reduces chiller electrical demand during high-tariff afternoon hours.
2. **AHU Fan Static Pressure Reset:**  
   Reset Air Handling Unit (AHU) duct static pressure setpoints by $-0.15\text{ in. w.g.}$ to allow variable frequency drives (VFDs) to ramp down safely without starving terminal variable air volume (VAV) boxes.
3. **Zone Deadband Offset:**  
   Implement a temporary $+1.0^\circ\text{C}$ deadband widening in non-critical spaces (atria, administrative suites, corridors) during flagged peak hours to curb chiller compressor staging.
4. **Peripheral Lighting & Computer Lab Sweep:**  
   Trigger remote low-power sleep state on instructional computer labs and sweep unoccupied peripheral corridors.
5. **Research Lab Fume Hood Sashes:**  
   Encourage research staff to lower VAV fume hood sashes when unattended to reduce exhaust makeup air volume while preserving all laboratory safety air change requirements.

---

## 4. Real-World Data Limitations & Operational Boundaries

1. **Weather Forecast Uncertainty:** Historical model training relies on ground-truth recorded ambient temperatures. In live forward-looking 24-hour deployment, the model ingests Numerical Weather Prediction (NWP) forecasts which carry an uncertainty of $\pm 1.5^\circ\text{C}$ to $\pm 3.0^\circ\text{C}$, propagating non-linearly into cooling load predictions.
2. **Occupancy Proxy Granularity:** Binary occupancy indicates whether the building is active, but lacks spatial distribution (e.g., 200 students in a single lecture hall emit $\sim 24\text{ kW}$ of sensible body heat, heavily loading one specific AHU zone compared to 200 occupants dispersed across single offices).
3. **Thermal Mass & Solar Gain:** Heavy concrete masonry introduces a 1–2 hour thermal lag between peak outdoor temperature and peak indoor cooling load.
4. **Operational Boundary:** This prototype is strictly an **Advisory Decision Support System**. It does **not** assert direct digital control or autonomously override Building Automation System (BAS) or SCADA field controllers. All pre-cooling, static pressure resets, or zone setback actions must be reviewed and initiated by licensed facility engineers.
