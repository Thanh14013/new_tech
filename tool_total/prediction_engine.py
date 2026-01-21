"""
Prediction Engine for Tool Total
Matches external tool logic exactly, supporting both normal and Wool apps.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import json
import warnings

import numpy as np
import pandas as pd

from data_loader import load_config, get_date_range_for_campaigns, is_wool_app, get_observation_end_date

ROOT = Path(__file__).resolve().parents[1]


class PredictionEngine:
    def __init__(self, app_id: str = None):
        self.app_id = app_id
        self.cfg = load_config(app_id)
        self.target_day = self.cfg.get("target", {}).get("target_day", 60)

    def predict(self, df_slice: pd.DataFrame, window: str, 
                app_id: str = None, campaigns: list = None,
                observation_end_date: pd.Timestamp = None,
                target_day: int = None) -> Tuple[dict, pd.DataFrame]:
        """
        Analyze predictions from evaluated predictions dataframe
        
        MATCHING EXTERNAL TOOL LOGIC:
        - max_actual_day = (data_end_date - max_install).days (CALENDAR-based)
        - Actual D0-D60: cumrev_d{day} if available, else pred_cumrev_d{day} for D0-D{last_day}
        - Predicted D0-D{last_day}: Same as Actual (they must match!)
        - Predicted D{last_day+1}-D60: pred_cumrev_d{day}
        """
        # Determine effective target day
        effective_target_day = target_day if target_day is not None else self.target_day

        if df_slice.empty:
            return {}, df_slice
        
        # Get curve columns (pred_cumrev_d* is always available from step4)
        pred_curve_cols = sorted([c for c in df_slice.columns if c.startswith('pred_cumrev_d')],
                                  key=lambda x: int(x.split('_d')[1]))
        
        # Check for actual cumrev columns (may be merged from features file)
        actual_curve_cols = sorted([c for c in df_slice.columns if c.startswith('cumrev_d') and not c.startswith('pred_')],
                                    key=lambda x: int(x.split('_d')[1]))
        has_actual_cols = len(actual_curve_cols) > 0
        
        if not pred_curve_cols:
            return {}, df_slice
        
        # Extract days from pred column names
        pred_days = [int(c.split('_d')[1]) for c in pred_curve_cols]
        days_labels = [f"D{d}" for d in pred_days]
        
        # Get window config
        window_config = self.cfg["windows"].get(window, {})
        last_day = max(window_config.get("feature_days", [7]))
        
        # Ensure install_date is datetime
        if 'install_date' in df_slice.columns:
            if not pd.api.types.is_datetime64_any_dtype(df_slice['install_date']):
                df_slice['install_date'] = pd.to_datetime(df_slice['install_date'])
            
            max_install = df_slice['install_date'].max()
            min_install = df_slice['install_date'].min()
        else:
            max_install = pd.Timestamp.now()
            min_install = pd.Timestamp.now()
        
        # === CALENDAR-BASED MAX_ACTUAL_DAY (Key Fix) ===
        # Get actual data end date.
        
        # PRIORITY: Use valid observation_end_date passed from app (Hard Limit)
        if observation_end_date is not None:
             data_end_date = pd.Timestamp(observation_end_date)
        elif app_id is not None:
             data_end_date = get_observation_end_date(app_id)
        else:
             data_end_date = pd.Timestamp(2025, 12, 31)
        
        # Calculate cohort age for each row (Needed for Cutoff Logic)
        if 'install_date' in df_slice.columns:
             df_slice['cohort_age'] = (data_end_date - df_slice['install_date']).dt.days
        else:
             df_slice['cohort_age'] = 0

        # STRICT CUTOFF: Only show Actuals if ALL cohorts have reached that day.
        # This prevents the "Plateau" effect where the curve flattens because young cohorts drop out of the sum.
        if not df_slice.empty:
            min_age = df_slice['cohort_age'].min()
            
            # Find max available actual data day
            # Scan columns cumrev_d0...cumrev_d60
            avail_days = [int(c.split('cumrev_d')[1]) for c in df_slice.columns if c.startswith('cumrev_d') and c[8:].isdigit()]
            max_data_day = max(avail_days) if avail_days else 0
            
            # Strict Cutoff: min(cohort_age, target_day, max_available_data)
            max_actual_day = max(0, min(min_age, effective_target_day, max_data_day))
            
            # WOOL SPECIFIC CONSTRAINT: Nov/Dec 2025 data only valid up to D30
            # Even if columns exist (populated with 0 or plateau), we treat D30 as the limit.
            if self.app_id == "com.wool.puzzle.game3d":
                try:
                    slice_min_date = pd.to_datetime(df_slice['install_date'].min())
                    if slice_min_date >= pd.Timestamp("2025-11-01"):
                        max_actual_day = min(max_actual_day, 30)
                except:
                    pass
        else:
            max_actual_day = 0
        
        # === BUILD SERIES (Interpolated D0-Target) ===
        # We need daily values for D0-Target.
        
        days_labels = [f"D{i}" for i in range(effective_target_day + 1)]

        actual_series = []
        predicted_series = []
        
        # 1. Calculate Aggregate Values for known days
        # 1. Calculate Aggregate Values for known days
        known_days = sorted(list(set(pred_days + [0, 1, 2, 3, 4, 5, 6, 7, 14, 30, 60])))
        # Filter known days strictly within target range (plus adjacent for interpolation padding)
        known_days = [d for d in known_days if d <= effective_target_day or d <= 60]
        agg_preds = {}
        agg_actuals = {}
        
        for d in known_days:
            pred_col = f"pred_cumrev_d{d}"
            actual_col = f"cumrev_d{d}"
            
            # Pred Sum
            if pred_col in df_slice.columns:
                agg_preds[d] = df_slice[pred_col].sum()
            elif d == 0 and 'pred_cumrev_d1' in df_slice.columns:
                 # Estimate D0 as ratio if missing? Or just 0 if very small. 
                 # Usually D0 is in features.
                 if 'cumrev_d0' in df_slice.columns:
                     agg_preds[d] = df_slice['cumrev_d0'].sum()
                 else:
                     agg_preds[d] = 0
            else:
                agg_preds[d] = np.nan

            # Actual Sum
            if actual_col in df_slice.columns:
                agg_actuals[d] = df_slice[actual_col].sum()
            else:
                agg_actuals[d] = np.nan

        # 2. Interpolate PREDICTED Curve
        # We have points in agg_preds. Fill missing days.
        # Linear interpolation for simplicity on the Aggregated Curve
        pred_curve = []
        
        # Convert agg_preds to Series for interpolation
        ser = pd.Series(index=range(effective_target_day + 1), dtype=float)
        for d, v in agg_preds.items():
            if d <= effective_target_day: ser[d] = v

            
        # Ensure endpoints exist
        if np.isnan(ser[0]): ser[0] = 0
        # If Target Day is missing, fill with max available
        if np.isnan(ser[effective_target_day]): 
            valid_idx = ser.dropna().index
            if len(valid_idx) > 0:
                 ser[effective_target_day] = ser[valid_idx.max()]
        
        # Interpolate
        ser = ser.interpolate(method='linear')
        predicted_series = ser.tolist()

        # 3. Build ACTUAL Curve
        # Only show actual if day <= max_actual_day
        for d in range(effective_target_day + 1):
            if d <= max_actual_day:
                if d in agg_actuals and not np.isnan(agg_actuals[d]):
                    actual_series.append(agg_actuals[d])
                else:
                    col = f"cumrev_d{d}"
                    if col in df_slice.columns:
                        actual_series.append(df_slice[col].sum())
                    else:
                        actual_series.append(np.nan)
            else:
                actual_series.append(np.nan)

        # 4. Interpolate Actuals (Refinement)
        # Create a series for actuals, fill knowns, interp, then mask by max_actual_day
        act_ser = pd.Series(index=range(effective_target_day + 1), dtype=float)
        for d in known_days:
            if d in agg_actuals: act_ser[d] = agg_actuals[d]
        
        act_ser = act_ser.interpolate(method='linear')
        
        # Copy to clean list obeying max_actual_day
        actual_series = []
        for d in range(effective_target_day + 1):
            if d <= max_actual_day:
                actual_series.append(act_ser[d])
            else:
                actual_series.append(np.nan)

        # === ALIGNED PREDICTION LOGIC (User Request) ===
        # Issue: Comparing D60 Prediction with "D50 Actual" (incomplete) creates false error.
        # Solution: Compare "Predicted @ D50" vs "Actual @ D50".
        
        # 1. Determine Interpolation Points (Vectorized)
        avail_pred_cols = {int(c.split('_d')[1]): c for c in df_slice.columns if c.startswith('pred_cumrev_d')}
        # Ensure endpoints
        if 0 not in avail_pred_cols:
             df_slice['pred_cumrev_d0'] = 0
             avail_pred_cols[0] = 'pred_cumrev_d0'
        
        sorted_days = sorted(avail_pred_cols.keys()) # e.g. [0, 7, 14, 30, 60]
        
        # 2. Calculate Last Valid Day per Cohort
        df_slice['last_valid_day'] = df_slice['cohort_age'].clip(upper=effective_target_day)
        
        # 3. Interpolate Predicted Value at 'last_valid_day'
        df_slice['pred_aligned'] = 0.0
        
        for i in range(len(sorted_days) - 1):
            d_start, d_end = sorted_days[i], sorted_days[i+1]
            col_start, col_end = avail_pred_cols[d_start], avail_pred_cols[d_end]
            
            # Mask for rows falling in this interval [d_start, d_end]
            if i == len(sorted_days) - 2:
                mask = (df_slice['last_valid_day'] >= d_start) & (df_slice['last_valid_day'] <= d_end)
            else:
                mask = (df_slice['last_valid_day'] >= d_start) & (df_slice['last_valid_day'] < d_end)
            
            if mask.any():
                # Linear Interp: y = y0 + (y1-y0) * (x-x0)/(x1-x0)
                slope = (df_slice.loc[mask, col_end] - df_slice.loc[mask, col_start]) / (d_end - d_start)
                dx = df_slice.loc[mask, 'last_valid_day'] - d_start
                df_slice.loc[mask, 'pred_aligned'] = df_slice.loc[mask, col_start] + slope * dx

        # For cohorts older than max target day (e.g. > D60), pred_aligned is just D60
        mask_mature = df_slice['last_valid_day'] >= sorted_days[-1]
        if mask_mature.any():
            df_slice.loc[mask_mature, 'pred_aligned'] = df_slice.loc[mask_mature, avail_pred_cols[sorted_days[-1]]]
            
        # 4. Metrics Calculation (Using ALIGNED values)
        has_actual = 'target' in df_slice.columns
        total_actual = None
        total_pred = 0
        
        # --- Interpolate DAILY Prediction Columns for Detail Table ---
        # We need pred_cumrev_d0...d{Target} for the table display
        first_day_idx = min(sorted_days)
        last_day_idx = max(sorted_days)
        
        for d in range(effective_target_day + 1):
            target_col = f"pred_cumrev_d{d}"
            if target_col not in df_slice.columns:
                # Find interpolation bounds
                # sorted_days e.g. [0, 7, 14, 30, 60]
                lower = max([k for k in sorted_days if k <= d], default=first_day_idx)
                upper = min([k for k in sorted_days if k >= d], default=last_day_idx)
                
                if lower == upper:
                    df_slice[target_col] = df_slice[avail_pred_cols[lower]]
                else:
                    # Linear Int: y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
                    ratio = (d - lower) / (upper - lower)
                    df_slice[target_col] = (
                        df_slice[avail_pred_cols[lower]] * (1 - ratio) + 
                        df_slice[avail_pred_cols[upper]] * ratio
                    )

        if f'pred_cumrev_d{effective_target_day}' in df_slice.columns:
            total_pred = df_slice[f'pred_cumrev_d{effective_target_day}'].sum()
        elif 'pred_cumrev_d60' in df_slice.columns and effective_target_day == 60:
             total_pred = df_slice['pred_cumrev_d60'].sum()
        else:
             # If exact column missing (e.g. D30 missing but target 30), rely on interpolated series
             total_pred = predicted_series[-1] if predicted_series else 0
        
        if has_actual:
            # Only show Total Actual D60 if ALL cohorts have reached D60
            min_age = df_slice['cohort_age'].min()
            if min_age >= effective_target_day:
                total_actual = df_slice['target'].sum()
            else:
                total_actual = None
            total_pred_aligned = df_slice['pred_aligned'].sum()
        else:
            total_pred_aligned = total_pred

        # Calculate metrics using Target values
        predicted_series_last = predicted_series[effective_target_day] if len(predicted_series) > effective_target_day else 0
        if not has_actual:
            total_pred_aligned = predicted_series_last

        mape = None
        mae = None
        
        if has_actual:
            errors = df_slice['pred_aligned'] - df_slice['target']
            mae = errors.abs().mean()
            
            # Global MAPE (Total Error / Total Actual)
            # User request: "MAPE là total actual / total Predict"
            sum_actual = df_slice['target'].sum()
            sum_pred = df_slice['pred_aligned'].sum()
            
            if sum_actual > 0:
                mape = abs(sum_pred - sum_actual) / sum_actual * 100
            else:
                mape = None

        # Get install date range
        min_install_date = min_install.strftime('%d/%m/%Y')
        max_install_date = max_install.strftime('%d/%m/%Y')
        
        cohort_count = len(df_slice)
        
        # Build summary
        # Calculate Total Cost for ROAS
        total_cost = df_slice['cost'].sum() if 'cost' in df_slice.columns else 0
        
        # Calculate ROAS Series
        roas_pred_series = []
        roas_act_series = []
        
        if total_cost > 0:
            roas_pred_series = [p / total_cost for p in predicted_series]
            roas_act_series = [a / total_cost if pd.notna(a) else np.nan for a in actual_series]

        # Build summary
        summary = {
            'total_pred': total_pred_aligned,
            'total_actual': total_actual if has_actual else 0,
            'avg_pred': total_pred_aligned / cohort_count if cohort_count > 0 else 0,
            'cohort_count': cohort_count,
            'mape': mape,
            'mae': mae,
            'has_actual': has_actual,
            'has_comparison': has_actual and total_actual is not None,
            'days_labels': days_labels,
            'actual_series': actual_series,
            'predicted_series': predicted_series,
            'roas_predicted_series': roas_pred_series,
            'roas_actual_series': roas_act_series,
            'total_cost': total_cost,
            'target_day': effective_target_day,
            'last_day': last_day,
            'max_actual_day': max_actual_day,
            'min_install_date': min_install_date,
            'max_install_date': max_install_date,
            'window_boundary_date': '',
        }
        
        # Update Detail DF with ROAS
        if 'cost' in df_slice.columns:
            # Handle cost=0 to avoid division by zero
            df_slice['pred_roas'] = df_slice.apply(lambda r: r['pred_aligned'] / r['cost'] if r['cost'] > 0 else 0, axis=1)
        else:
            df_slice['pred_roas'] = 0
        
        # Build detail dataframe
        # Build detail dataframe
        # Base columns
        detail_cols = ['install_date', 'app_id', 'campaign', 'installs', 'cost']
        
        # Add User columns if any (Only up to max_actual_day)
        all_user_cols = [c for c in df_slice.columns if c.startswith('unique_users_day')]
        
        def get_day_idx(c):
             # unique_users_day1 -> 1
            try:
                return int(c.split('unique_users_day')[1])
            except:
                return 9999
        
        user_cols = sorted([c for c in all_user_cols if get_day_idx(c) <= max_actual_day], key=get_day_idx)
        detail_cols.extend(user_cols)
        
        available_detail_cols = [c for c in detail_cols if c in df_slice.columns]
        detail_df = df_slice[available_detail_cols].copy() if available_detail_cols else df_slice[[]].copy()
        
        # Add Revenue History (D0 to Target)
        # We want to show cumrev_d0, d1... up to effective_target_day
        for d in range(effective_target_day + 1):
            col = f"cumrev_d{d}"
            pred_col = f"pred_cumrev_d{d}"
            
            if col in df_slice.columns:
                detail_df[col] = df_slice[col]
            if pred_col in df_slice.columns:
                detail_df[pred_col] = df_slice[pred_col]
        
        # Add Last Day Info
        detail_df['last_day'] = df_slice['last_valid_day'].astype(str).map(lambda x: f"D{x}")
        
        if 'pred_aligned' in df_slice.columns:
            detail_df['predicted_ltv_last_day'] = df_slice['pred_aligned']
        
        if has_actual:
            detail_df['actual_ltv_last_day'] = df_slice['target']
            if 'pred_aligned' in df_slice.columns:
                detail_df['error'] = df_slice['pred_aligned'] - df_slice['target']
                detail_df['pct_error'] = ((df_slice['pred_aligned'] - df_slice['target']) / df_slice['target'] * 100).fillna(0)
        
        if 'pred_roas' in df_slice.columns:
            detail_df['pred_roas'] = df_slice['pred_roas']
        
        return summary, detail_df


def get_engine(app_id: str = None) -> PredictionEngine:
    """Get prediction engine for specific app"""
    return PredictionEngine(app_id)
