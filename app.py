"""
Campus Energy Consumption Forecast & Peak Advisory Console.
Interactive decision-support platform for Campus Facilities & Energy Operations.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.config import DATA_PATH, PEAK_PERCENTILE, CRITICAL_PERCENTILE
from src.data_loader import prepare_data
from src.models import (
    PersistenceBaseline24h,
    PersistenceBaseline168h,
    HistoricalMeanBaseline,
    RidgeEnergyForecaster,
    HistGradientBoostingForecaster,
    XGBoostEnergyForecaster,
)
from src.evaluator import evaluate_models_comparison, calculate_metrics
from src.advisory import CampusAdvisoryEngine

# Streamlit Page Config
st.set_page_config(
    page_title="Campus Energy Operations Console",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling: Hides Streamlit Deploy button & top toolbar completely
st.markdown(
    """
    <style>
    /* Remove Streamlit Deploy button, hamburger menu and header toolbar */
    .stDeployButton { display: none !important; }
    div[data-testid="stToolbar"] { display: none !important; visibility: hidden !important; height: 0px !important; }
    div[data-testid="stDecoration"] { display: none !important; }
    #MainMenu { visibility: hidden !important; display: none !important; }
    header[data-testid="stHeader"] { display: none !important; height: 0px !important; }
    footer { visibility: hidden !important; display: none !important; }

    /* Clean Application Styling */
    .main { background-color: #fafafa; padding-top: 1rem; }
    .kpi-container {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .alert-banner-warning {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 20px;
    }
    .alert-banner-ok {
        background-color: #f0fdf4;
        border-left: 4px solid #22c55e;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_and_initialize_system():
    train_df, val_df, test_df, feature_cols, target_col = prepare_data(DATA_PATH)

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

    peak_threshold = float(np.percentile(train_df[target_col], PEAK_PERCENTILE))
    critical_threshold = float(np.percentile(train_df[target_col], CRITICAL_PERCENTILE))

    df_bench = evaluate_models_comparison(models, X_test, y_test, peak_threshold=peak_threshold)
    feat_imp = models["XGBoost Energy Forecaster"].get_feature_importances()

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "models": models,
        "peak_threshold": peak_threshold,
        "critical_threshold": critical_threshold,
        "df_bench": df_bench,
        "feat_imp": feat_imp,
    }


def compute_telemetry_statistics(df: pd.DataFrame, load_col: str, temp_col: str = None) -> dict:
    """Computes comprehensive statistics for any uploaded building telemetry CSV."""
    series = df[load_col].dropna().astype(float)
    n = len(series)

    mean_val = float(series.mean())
    std_val = float(series.std())
    min_val = float(series.min())
    p25 = float(series.quantile(0.25))
    median_val = float(series.median())
    p75 = float(series.quantile(0.75))
    p85 = float(series.quantile(0.85))
    p95 = float(series.quantile(0.95))
    max_val = float(series.max())
    iqr = p75 - p25
    skew = float(series.skew())
    total_energy_kwh = float(series.sum())

    load_factor = (mean_val / max_val * 100.0) if max_val > 0 else 0.0
    baseload_proxy = float(series.quantile(0.05))

    stats_dict = {
        "Sample Count": n,
        "Total Energy (MWh)": round(total_energy_kwh / 1000.0, 2),
        "Mean Demand (kWh)": round(mean_val, 2),
        "Median Demand (kWh)": round(median_val, 2),
        "Standard Deviation (kWh)": round(std_val, 2),
        "Minimum (kWh)": round(min_val, 2),
        "25th Percentile (kWh)": round(p25, 2),
        "75th Percentile (kWh)": round(p75, 2),
        "IQR (kWh)": round(iqr, 2),
        "85th Percentile Peak Trigger (kWh)": round(p85, 2),
        "95th Percentile Critical Peak (kWh)": round(p95, 2),
        "Maximum Demand Peak (kWh)": round(max_val, 2),
        "Load Factor (%)": round(load_factor, 1),
        "Baseload Estimate (kWh)": round(baseload_proxy, 2),
        "Skewness": round(skew, 2),
    }

    if temp_col and temp_col in df.columns:
        valid_pairs = df[[load_col, temp_col]].dropna()
        if len(valid_pairs) > 1:
            stats_dict["Temperature-Load Correlation (r)"] = round(float(valid_pairs[load_col].corr(valid_pairs[temp_col])), 3)

    return stats_dict


def main():
    ctx = load_and_initialize_system()
    test_df = ctx["test_df"]
    feature_cols = ctx["feature_cols"]
    target_col = ctx["target_col"]
    models = ctx["models"]
    peak_threshold = ctx["peak_threshold"]
    critical_threshold = ctx["critical_threshold"]

    # Header
    col_hdr, col_info = st.columns([3, 1])
    with col_hdr:
        st.title("⚡ Campus Energy Consumption Forecast & Peak Advisory Console")
        st.caption("AI-Powered 24-Hour Horizon Load Forecasting, Peak Shaving Advisory & Baseline Benchmarking")
    with col_info:
        st.markdown(
            f"""
            <div style="text-align: right; font-size: 0.85rem; color: #64748b; padding-top: 10px;">
                <strong>Dataset:</strong> 13,128 Hourly Records<br>
                <strong>Operational Threshold:</strong> {peak_threshold:.1f} kWh<br>
                <strong>Evaluation:</strong> Out-of-Sample Holdout Test
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Sidebar Controls
    st.sidebar.header("🕹️ Horizon & Operations Controls")
    preset_scenario = st.sidebar.selectbox(
        "Operational Inspection Scenario",
        [
            "Peak Summer Heatwave Window (May 19, 2025)",
            "Moderate Spring Instructional Day (Apr 15, 2025)",
            "Weekend Low-Occupancy Day (May 18, 2025)",
            "Late Spring Shoulder Day (Jun 02, 2025)",
            "Custom Date Selection",
        ],
    )

    if "May 19, 2025" in preset_scenario:
        selected_date = pd.to_datetime("2025-05-19").date()
    elif "Apr 15, 2025" in preset_scenario:
        selected_date = pd.to_datetime("2025-04-15").date()
    elif "May 18, 2025" in preset_scenario:
        selected_date = pd.to_datetime("2025-05-18").date()
    elif "Jun 02, 2025" in preset_scenario:
        selected_date = pd.to_datetime("2025-06-02").date()
    else:
        min_d = test_df["timestamp"].min().date()
        max_d = (test_df["timestamp"].max() - pd.Timedelta(days=1)).date()
        selected_date = st.sidebar.date_input("Date to Inspect", min_d, min_value=min_d, max_value=max_d)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Peak Demand Threshold Tuning")
    custom_threshold = st.sidebar.slider(
        "Advisory Alert Trigger (kWh)",
        min_value=350.0,
        max_value=600.0,
        value=float(round(peak_threshold, 0)),
        step=5.0,
        help="Loads projected above this limit trigger priority HVAC pre-cooling and static pressure setback checklists.",
    )

    # Slice 24h
    start_ts = pd.to_datetime(f"{selected_date} 00:00:00")
    end_ts = start_ts + pd.Timedelta(hours=23)
    slice_24h = test_df[(test_df["timestamp"] >= start_ts) & (test_df["timestamp"] <= end_ts)].copy().reset_index(drop=True)

    if len(slice_24h) < 24:
        st.warning(f"Incomplete records for date {selected_date}. Please choose another test period date.")
        return

    # Predictions
    slice_24h["predicted_kwh"] = models["XGBoost Energy Forecaster"].predict(slice_24h[feature_cols])
    slice_24h["baseline_24h_kwh"] = models["Baseline (Same Hour Previous Day)"].predict(slice_24h[feature_cols])
    slice_24h["baseline_168h_kwh"] = models["Baseline (Same Hour Previous Week)"].predict(slice_24h[feature_cols])

    # Run advisory engine
    engine = CampusAdvisoryEngine(peak_threshold_kwh=custom_threshold, critical_threshold_kwh=critical_threshold)
    advisory = engine.assess_24h_window(slice_24h, pred_col="predicted_kwh")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "⚡ 24-Hour Horizon & Peak Operations",
        "📈 Custom CSV Telemetry Statistics",
        "📊 Model Accuracy Benchmarking",
        "🧪 What-If Facility Simulator",
        "📋 Data Limitations & Technical Notes",
    ])

    # TAB 1: 24h Horizon
    with tab1:
        # Top KPI Cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Total 24h Energy", f"{advisory['total_24h_kwh']:,} kWh")
        with kpi2:
            st.metric("Peak Projected Demand", f"{advisory['peak_demand_kwh']} kWh", f"at {slice_24h.loc[slice_24h['predicted_kwh'].idxmax(), 'timestamp'].strftime('%H:%M')}")
        with kpi3:
            p_diff = advisory["peak_demand_kwh"] - custom_threshold
            st.metric("Peak Alert Hours", f"{advisory['n_flagged_hours']} hrs", f"{'+' if p_diff > 0 else ''}{p_diff:.1f} kWh vs Threshold")
        with kpi4:
            mae_24h = calculate_metrics(slice_24h[target_col], slice_24h["predicted_kwh"])["MAE (kWh)"]
            base_mae = calculate_metrics(slice_24h[target_col], slice_24h["baseline_24h_kwh"])["MAE (kWh)"]
            st.metric("Forecast Accuracy (MAE)", f"{mae_24h:.1f} kWh", f"-{(base_mae - mae_24h):.1f} kWh vs Prev Day")

        # Plotly Dual-Panel Chart
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.08, row_heights=[0.72, 0.28],
            specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
        )

        times = slice_24h["timestamp"]
        fig.add_trace(go.Scatter(x=times, y=slice_24h[target_col], mode="lines+markers", name="Actual Meter Reading (kWh)", line=dict(color="#0f172a", width=2.5), marker=dict(size=5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=times, y=slice_24h["predicted_kwh"], mode="lines+markers", name="XGBoost 24h Advance Forecast", line=dict(color="#2563eb", width=3.0), marker=dict(size=5, symbol="square")), row=1, col=1)
        fig.add_trace(go.Scatter(x=times, y=slice_24h["baseline_24h_kwh"], mode="lines", name="Baseline: Same Hour Previous Day (t-24)", line=dict(color="#94a3b8", width=2, dash="dash")), row=1, col=1)

        fig.add_hline(y=custom_threshold, line_dash="dot", line_color="#ef4444", line_width=2, annotation_text=f"Peak Alert Threshold ({custom_threshold:.0f} kWh)", row=1, col=1)

        fig.add_trace(go.Scatter(x=times, y=slice_24h["temperature_c"], name="Ambient Temp (°C)", line=dict(color="#ea580c", width=2)), row=2, col=1, secondary_y=False)
        fig.add_trace(go.Bar(x=times, y=slice_24h["is_occupied"], name="Occupancy Indicator", marker_color="rgba(34, 197, 94, 0.3)"), row=2, col=1, secondary_y=True)

        fig.update_layout(height=560, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=30, r=30, t=60, b=30))
        fig.update_yaxes(title_text="Electrical Load (kWh)", row=1, col=1)
        fig.update_yaxes(title_text="Temp (°C)", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Occupancy", row=2, col=1, secondary_y=True, range=[-0.1, 1.2], tickvals=[0, 1], ticktext=["Vacant", "Occupied"])
        st.plotly_chart(fig, use_container_width=True)

        # Operational Action Plan
        st.subheader("🛠️ Recommended Facility Engineering Checks")
        if advisory["n_flagged_hours"] > 0:
            st.markdown(
                f"""
                <div class="alert-banner-warning">
                    <strong>⚠️ HIGH DEMAND ADVISORY:</strong> {advisory['n_flagged_hours']} hours predicted to exceed {custom_threshold:.0f} kWh. 
                    Flagged window: <strong>{', '.join(advisory['flagged_hours'])}</strong>. 
                    Physical drivers: <em>{'; '.join(advisory['peak_drivers'])}</em>.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="alert-banner-ok">
                    <strong>✅ NOMINAL OPERATIONS:</strong> Forecasted demand remains within normal baseline limits. No peak demand threshold exceedance projected.
                </div>
                """,
                unsafe_allow_html=True,
            )

        for act in advisory["action_plan"]:
            priority_icon = "🔴" if act["priority"] == "HIGH" else ("🟠" if act["priority"] == "MEDIUM" else "🟢")
            with st.expander(f"{priority_icon} [{act['priority']}] {act['subsystem']}: {act['recommendation']}", expanded=(act["priority"] == "HIGH")):
                st.write(act["technical_details"])

        st.info(f"🛡️ **System Operational Boundary:** {advisory['governance_disclaimer']}")

    # TAB 2: Custom CSV Telemetry Statistics Generator
    with tab2:
        st.subheader("📈 Custom Building Telemetry Analysis & Automated Statistics")
        st.markdown(
            "Upload any building energy consumption or sub-metering CSV file. The system will automatically "
            "compute distribution metrics, peak demand thresholds (85th/95th percentiles), load duration curves, "
            "and diurnal load profiles."
        )

        col_up, col_btn = st.columns([3, 1])
        with col_up:
            uploaded_file = st.file_uploader(
                "Upload Telemetry CSV",
                type=["csv"],
                help="Upload a CSV with hourly or interval time-series data.",
            )
        with col_btn:
            st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
            if DATA_PATH.exists():
                with open(DATA_PATH, "rb") as f_sample:
                    st.download_button(
                        label="📥 Download Dataset CSV",
                        data=f_sample,
                        file_name="campus_energy_dataset.csv",
                        mime="text/csv",
                    )

        if uploaded_file is not None:
            try:
                user_df = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded `{uploaded_file.name}` ({len(user_df):,} rows, {len(user_df.columns)} columns).")

                col_c1, col_c2, col_c3 = st.columns(3)
                time_candidates = [c for c in user_df.columns if any(k in c.lower() for k in ["time", "date"])]
                default_time_idx = user_df.columns.get_loc(time_candidates[0]) if time_candidates else 0
                with col_c1:
                    time_col_sel = st.selectbox("Timestamp / Date Column", user_df.columns, index=default_time_idx)

                load_candidates = [c for c in user_df.columns if any(k in c.lower() for k in ["kwh", "meter", "load", "energy", "consumption", "reading", "demand"])]
                default_load_idx = user_df.columns.get_loc(load_candidates[0]) if load_candidates else min(1, len(user_df.columns)-1)
                with col_c2:
                    load_col_sel = st.selectbox("Energy / Load Column (kWh/kW)", user_df.columns, index=default_load_idx)

                temp_candidates = [c for c in user_df.columns if any(k in c.lower() for k in ["temp", "weather"])]
                temp_options = ["None"] + list(user_df.columns)
                default_temp_idx = (temp_options.index(temp_candidates[0])) if temp_candidates else 0
                with col_c3:
                    temp_col_sel = st.selectbox("Ambient Temperature Column (Optional)", temp_options, index=default_temp_idx)

                try:
                    user_df[time_col_sel] = pd.to_datetime(user_df[time_col_sel])
                    user_df = user_df.sort_values(time_col_sel).reset_index(drop=True)
                    has_valid_time = True
                except Exception:
                    has_valid_time = False

                t_col = temp_col_sel if temp_col_sel != "None" else None
                stats_res = compute_telemetry_statistics(user_df, load_col=load_col_sel, temp_col=t_col)

                st.markdown("---")
                st.subheader("📊 Key Telemetry & Demand Statistics")

                st1, st2, st3, st4, st5 = st.columns(5)
                with st1:
                    st.metric("Total Records", f"{stats_res['Sample Count']:,}")
                with st2:
                    st.metric("Total Energy", f"{stats_res['Total Energy (MWh)']:,} MWh")
                with st3:
                    st.metric("Peak Demand", f"{stats_res['Maximum Demand Peak (kWh)']} kWh")
                with st4:
                    st.metric("Mean Demand", f"{stats_res['Mean Demand (kWh)']} kWh", f"±{stats_res['Standard Deviation (kWh)']} σ")
                with st5:
                    st.metric("Load Factor", f"{stats_res['Load Factor (%)']}%", help="Mean Load / Peak Load ratio.")

                st6, st7, st8, st9, st10 = st.columns(5)
                with st6:
                    st.metric("Baseload Estimate", f"{stats_res['Baseload Estimate (kWh)']} kWh", help="5th percentile night load.")
                with st7:
                    st.metric("Median (P50)", f"{stats_res['Median Demand (kWh)']} kWh")
                with st8:
                    st.metric("85th Percentile Peak Trigger", f"{stats_res['85th Percentile Peak Trigger (kWh)']} kWh")
                with st9:
                    st.metric("95th Percentile Critical Peak", f"{stats_res['95th Percentile Critical Peak (kWh)']} kWh")
                with st10:
                    corr_display = f"{stats_res.get('Temperature-Load Correlation (r)', 'N/A')}"
                    st.metric("Temp Correlation (r)", corr_display)

                st.markdown("---")
                col_tab, col_ldc = st.columns([1, 1])

                with col_tab:
                    st.subheader("📋 Statistical Distribution Summary")
                    df_stats_table = pd.DataFrame(list(stats_res.items()), columns=["Metric Parameter", "Computed Value"])
                    st.dataframe(df_stats_table, use_container_width=True, height=360)

                    csv_export = df_stats_table.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Export Statistical Summary as CSV",
                        data=csv_export,
                        file_name=f"statistics_{uploaded_file.name}",
                        mime="text/csv",
                    )

                with col_ldc:
                    st.subheader("📉 Load Duration Curve (LDC)")
                    sorted_load = np.sort(user_df[load_col_sel].dropna().values)[::-1]
                    pct_hours = np.linspace(0, 100, len(sorted_load))

                    fig_ldc = go.Figure()
                    fig_ldc.add_trace(go.Scatter(x=pct_hours, y=sorted_load, mode="lines", name="Load Duration Curve", line=dict(color="#0f172a", width=2.5)))
                    fig_ldc.add_hline(y=stats_res["85th Percentile Peak Trigger (kWh)"], line_dash="dash", line_color="#ef4444", annotation_text="85th Pct Peak Threshold")
                    fig_ldc.add_hline(y=stats_res["95th Percentile Critical Peak (kWh)"], line_dash="dot", line_color="#dc2626", annotation_text="95th Pct Critical")
                    fig_ldc.update_layout(
                        title="Building Load Duration Curve",
                        xaxis_title="Percentage of Time Exceeded (%)",
                        yaxis_title="Demand (kWh)",
                        height=360,
                        margin=dict(l=30, r=30, t=50, b=30),
                    )
                    st.plotly_chart(fig_ldc, use_container_width=True)

                if has_valid_time:
                    st.markdown("---")
                    st.subheader("🕒 Diurnal Hourly Consumption Profile")
                    user_df["_hour"] = user_df[time_col_sel].dt.hour
                    hourly_stats = user_df.groupby("_hour")[load_col_sel].agg(["mean", "std", "min", "max"]).reset_index()

                    fig_diurnal = go.Figure()
                    fig_diurnal.add_trace(go.Scatter(
                        x=hourly_stats["_hour"], y=hourly_stats["mean"],
                        mode="lines+markers", name="Mean Hourly Load",
                        line=dict(color="#2563eb", width=3),
                    ))
                    fig_diurnal.add_trace(go.Scatter(
                        x=list(hourly_stats["_hour"]) + list(hourly_stats["_hour"])[::-1],
                        y=list(hourly_stats["mean"] + hourly_stats["std"]) + list(hourly_stats["mean"] - hourly_stats["std"])[::-1],
                        fill="toself", fillcolor="rgba(37, 99, 235, 0.15)",
                        line=dict(color="rgba(255,255,255,0)"),
                        hoverinfo="skip", name="±1 Standard Deviation",
                    ))
                    fig_diurnal.update_layout(
                        title=f"Hourly Diurnal Load Profile for `{load_col_sel}` (00:00 - 23:00)",
                        xaxis_title="Hour of Day",
                        yaxis_title="Average Electricity Demand (kWh)",
                        height=380,
                        xaxis=dict(tickmode="linear", tick0=0, dtick=2),
                    )
                    st.plotly_chart(fig_diurnal, use_container_width=True)

            except Exception as e:
                st.error(f"Error parsing uploaded CSV file: {str(e)}")
        else:
            st.info("👆 Upload an hourly telemetry CSV file above to compute and visualize custom building statistics.")

    # TAB 3: Model Benchmark
    with tab3:
        st.subheader("Out-of-Sample Holdout Test Performance (1,944 Hours)")
        st.markdown(
            "Evaluation performed strictly on chronological holdout data (Apr 11 to Jun 30, 2025). "
            "Zero random shuffling was used, preventing future data leakage."
        )
        st.dataframe(ctx["df_bench"], use_container_width=True)

        bcol1, bcol2 = st.columns(2)
        with bcol1:
            st.image(str(BASE_DIR / "reports" / "figures" / "model_benchmark_comparison.png"), caption="Comparative Holdout Errors: MAE, RMSE, MAPE, and Peak Load MAE")
        with bcol2:
            st.image(str(BASE_DIR / "reports" / "figures" / "feature_importance.png"), caption="XGBoost Feature Importance Weights (Gain)")

    # TAB 4: What-If Simulator
    with tab4:
        st.subheader("🧪 Operational What-If Facility Simulator")
        st.markdown(
            "Simulate extreme weather divergence or unexpected campus schedule modifications "
            "to assess the electrical load impact."
        )

        sim_col1, sim_col2 = st.columns(2)
        with sim_col1:
            sim_temp_delta = st.slider("Simulate Ambient Temperature Offset (Heatwave / Front)", -6.0, 6.0, 3.0, 0.5, format="%+.1f °C")
        with sim_col2:
            sim_evening_occ = st.checkbox("Extend Evening Academic Schedule (Occupied 18:00 - 22:00)", value=True)

        sim_slice = slice_24h.copy()
        sim_slice["temperature_c"] += sim_temp_delta
        if sim_evening_occ:
            sim_slice.loc[sim_slice["hour"].between(18, 22), "is_occupied"] = 1

        sim_slice["cooling_degree_demand"] = np.maximum(sim_slice["temperature_c"] - 18.0, 0.0)
        sim_slice["heating_degree_demand"] = np.maximum(14.0 - sim_slice["temperature_c"], 0.0)
        sim_slice["occ_cooling_interaction"] = sim_slice["is_occupied"] * sim_slice["cooling_degree_demand"]
        sim_slice["occ_heating_interaction"] = sim_slice["is_occupied"] * sim_slice["heating_degree_demand"]

        sim_slice["simulated_kwh"] = models["XGBoost Energy Forecaster"].predict(sim_slice[feature_cols])

        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(x=times, y=slice_24h["predicted_kwh"], name="Nominal 24h AI Forecast", line=dict(color="#2563eb", width=2.5)))
        fig_sim.add_trace(go.Scatter(x=times, y=sim_slice["simulated_kwh"], name="Simulated Scenario Load", line=dict(color="#dc2626", width=3, dash="dash")))
        fig_sim.add_hline(y=custom_threshold, line_dash="dot", line_color="#f59e0b", annotation_text="Peak Threshold")
        fig_sim.update_layout(height=420, title="Scenario Stress Test: Nominal Forecast vs. Simulated Stress Conditions", hovermode="x unified")
        st.plotly_chart(fig_sim, use_container_width=True)

        orig_tot = slice_24h["predicted_kwh"].sum()
        sim_tot = sim_slice["simulated_kwh"].sum()
        pct_change = (sim_tot - orig_tot) / orig_tot * 100.0
        st.write(f"**Scenario Outcome:** Total 24h consumption shifts from **{orig_tot:,.1f} kWh** to **{sim_tot:,.1f} kWh** ({'+' if pct_change > 0 else ''}{pct_change:.1f}%).")

    # TAB 5: Technical Notes & Limitations
    with tab5:
        st.subheader("📋 Engineering Limitations & Governance Boundaries")
        tech_notes = BASE_DIR / "reports" / "engineering_technical_report.md"
        if tech_notes.exists():
            st.markdown(tech_notes.read_text())
        else:
            st.write("Technical report available in reports/engineering_technical_report.md.")


if __name__ == "__main__":
    main()
