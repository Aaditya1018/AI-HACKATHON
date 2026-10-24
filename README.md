# CampusGrid AI — Campus Energy Consumption Forecasting System

> **AI-Powered Energy Intelligence & Operational Decision Support for Campus Facilities**
> *Developed for the AI Hackathon (Question 8: Campus Energy Consumption Forecast Scenario)*

---

## ⚡ Overview

Campus facility managers face challenges identifying high-demand electrical loads in advance to implement effective energy-saving strategies. **CampusGrid AI** is an end-to-end predictive prototype that ingests smart-meter telemetry and provides **24-hour ahead hourly consumption forecasts**, baseline comparisons, and prioritized operational checks without claiming autonomous equipment control.

### 🎯 Key Highlights & Deliverables
1. **Zero-Leakage ML Pipeline**: Strict chronological train/test split preserving temporal integrity.
2. **Gradient Boosting Machine (GBR)**: Trained across multi-lag temporal features (`lag_1h` to `lag_168h`), rolling windows, and proxy occupancy/temperature signals.
3. **Naive Baseline Benchmarking**: Compared against previous day same-hour consumption (`lag_24`), demonstrating an **11.3% reduction in MAE** (0.8981 kWh vs 1.0125 kWh).
4. **Actionable Peak Warnings**: Automated threshold detection (>90th percentile) with practical non-intrusive operational guidance (e.g., pre-cooling, lab equipment deferral).
5. **Interactive Single-Page Dashboard**: Built with Vanilla JS & CSS and Chart.js, featuring responsive tabs, analytics, and transparent data limitation disclosures.

---

## 🚀 Getting Started

### 1. Run Locally via Python HTTP Server
```bash
# Clone the repository
git clone https://github.com/Aaditya1018/AI-HACKATHON.git
cd AI-HACKATHON

# Launch local server
python3 -m http.server 8765
```
Open **[http://localhost:8765](http://localhost:8765)** in your browser.

---

## 📂 Project Structure

```text
├── index.html                     # Semantic dashboard UI with responsive tabs & metrics
├── style.css                      # Modern dark theme styling with glassmorphism & CSS tokens
├── app.js                         # Dynamic state management, Chart.js integrations & live data loader
├── energy_forecast_results.json   # Pre-computed model inference results, metrics, & importances
├── campus_energy_data.json        # Processed time-series records from raw meters
├── README.md                      # Documentation & problem statement overview
└── .gitignore                     # Git ignore rules
```

---

## 📊 Model Performance

| Metric | GBR Model | Baseline (Prev. Day Same-Hour) | Improvement |
| :--- | :--- | :--- | :--- |
| **MAE** | **0.8981 kWh** | 1.0125 kWh | **+11.3%** |
| **RMSE** | **2.0552 kWh** | 2.1264 kWh | **+3.4%** |
| **R² Score** | **0.871** | 0.835 | **+4.3%** |

