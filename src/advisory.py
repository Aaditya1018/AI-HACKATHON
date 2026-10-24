"""
Campus Facility Operations & Peak Advisory Engine.

Detects upcoming peak demand hours and issues human-in-the-loop operational checklists:
- HVAC Thermal Pre-Cooling
- Static Pressure & Air Handling Unit (AHU) VFD optimization
- Non-critical zone temperature deadband setback
- Plug load & unoccupied corridor sweeps
- Fume hood sash management

GOVERNANCE:
This system generates operational guidance for building engineers and facilities managers.
It does NOT assert direct digital control or write to BACnet/SCADA controllers.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd


class CampusAdvisoryEngine:
    """Advisory decision-support engine for campus building managers."""

    def __init__(self, peak_threshold_kwh: float, critical_threshold_kwh: float):
        self.peak_threshold = peak_threshold_kwh
        self.critical_threshold = critical_threshold_kwh

    def assess_24h_window(
        self,
        df_24h: pd.DataFrame,
        pred_col: str = "predicted_kwh",
    ) -> Dict[str, Any]:
        df = df_24h.copy().reset_index(drop=True)

        df["alert_status"] = "NOMINAL"
        df.loc[df[pred_col] >= self.peak_threshold, "alert_status"] = "HIGH DEMAND"
        df.loc[df[pred_col] >= self.critical_threshold, "alert_status"] = "CRITICAL PEAK"

        peak_hours = df[df["alert_status"].isin(["HIGH DEMAND", "CRITICAL PEAK"])]
        n_peak = len(peak_hours)
        peak_demand_kwh = float(df[pred_col].max())
        peak_timestamp = df.loc[df[pred_col].idxmax(), "timestamp"]
        total_energy_kwh = float(df[pred_col].sum())

        # Determine dominant physical drivers
        drivers = []
        if n_peak > 0:
            avg_temp = float(peak_hours["temperature_c"].mean())
            occ_pct = float(peak_hours["is_occupied"].mean())

            if avg_temp >= 26.0:
                drivers.append("High ambient temperature (chiller & cooling tower demand)")
            elif avg_temp <= 5.0:
                drivers.append("Cold ambient condition (heating fan & ventilation demand)")

            if occ_pct >= 0.70:
                drivers.append("Peak campus building occupancy & classroom plug load")

            if not drivers:
                drivers.append("Cumulative diurnal cooling load and persistent ventilation requirement")

        # Compile structured actionable checks
        actions = self._build_action_plan(n_peak, peak_hours, df)

        return {
            "total_24h_kwh": round(total_energy_kwh, 1),
            "peak_demand_kwh": round(peak_demand_kwh, 1),
            "peak_time": str(peak_timestamp),
            "n_flagged_hours": n_peak,
            "flagged_hours": peak_hours["timestamp"].dt.strftime("%H:%M").tolist(),
            "peak_drivers": drivers,
            "action_plan": actions,
            "governance_disclaimer": (
                "OPERATIONAL NOTICE: This AI prototype generates predictive decision-support recommendations "
                "for campus facility personnel. It does not exert direct Supervisory Control and Data Acquisition "
                "(SCADA) or Building Automation System (BAS/BACnet) actuator override. All actions require manual "
                "review and physical verification by licensed operators."
            ),
            "annotated_df": df,
        }

    def _build_action_plan(
        self, n_peak: int, peak_hours: pd.DataFrame, df_24h: pd.DataFrame
    ) -> List[Dict[str, str]]:
        if n_peak == 0:
            return [{
                "subsystem": "Base Scheduling",
                "priority": "LOW",
                "recommendation": "Maintain Nominal Occupancy Setbacks",
                "technical_details": "Predicted electrical demand remains below the 85th percentile threshold. Ensure nighttime setbacks remain active.",
            }]

        plan = []
        earliest_hour = int(peak_hours["hour"].min())
        precool_start = max(earliest_hour - 3, 5)

        # 1. Thermal Pre-cooling
        plan.append({
            "subsystem": "Central Chiller Plant",
            "priority": "HIGH",
            "recommendation": f"Stage Thermal Pre-Cooling ({precool_start:02d}:00 to {earliest_hour:02d}:00)",
            "technical_details": (
                f"Sub-cool primary lecture halls and high-thermal-mass concrete zones by 1.0°C to 1.5°C "
                f"between {precool_start:02d}:00 and {earliest_hour:02d}:00. This shifts chiller electrical "
                "demand away from the high-tariff afternoon peak window."
            ),
        })

        # 2. AHU & Fan Static Pressure
        plan.append({
            "subsystem": "Air Handling Units (AHUs)",
            "priority": "HIGH",
            "recommendation": "Audit Supply Fan Static Pressure Setpoint",
            "technical_details": (
                "Reset AHU duct static pressure setpoint by -0.15 in. w.g. to allow variable frequency "
                "drives (VFDs) to ramp down safely without starving terminal VAV boxes."
            ),
        })

        # 3. Zone Deadband Reset
        plan.append({
            "subsystem": "Terminal VAV Zones",
            "priority": "MEDIUM",
            "recommendation": "Implement +1.0°C Peak Demand Offset in Non-Critical Spaces",
            "technical_details": (
                "Temporarily widen temperature deadbands by +1.0°C in common atria, departmental corridors, "
                "and administrative offices during flagged peak hours to curb compressor staging."
            ),
        })

        # 4. Plug Load & Lighting
        plan.append({
            "subsystem": "Lighting & Plug Loads",
            "priority": "MEDIUM",
            "recommendation": "Conduct Peripheral Lighting & Computer Lab Sweep",
            "technical_details": (
                "Remotely trigger low-power standby on instructional computer labs and sweep exterior "
                "decorative / unoccupied conference room lighting."
            ),
        })

        # 5. Laboratory Safety Check
        plan.append({
            "subsystem": "Wet Labs & Fume Hoods",
            "priority": "SAFETY ADVISORY",
            "recommendation": "Verify Fume Hood Sashes are Lowered",
            "technical_details": (
                "Encourage lab researchers to lower variable air volume (VAV) fume hood sashes to minimum stop "
                "when unmonitored. Maintain all EH&S air change safety compliance."
            ),
        })

        return plan
