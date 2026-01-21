"""
Prediction Engine for D60 LTV Tool
Simplified engine for D60 LTV predictions using our trained models
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import json
import warnings
import sys

import numpy as np
import pandas as pd

# Import from same directory
try:
    from data_loader import load_config, get_date_range_for_campaigns, is_wool_app, get_observation_end_date
except ImportError:
    from tool_total.data_loader import load_config, get_date_range_for_campaigns, is_wool_app, get_observation_end_date

ROOT = Path(__file__).resolve().parents[1]

# Import production pipeline
sys.path.insert(0, str(ROOT))
from tool_total.production_pipeline import ProductionPipeline


class PredictionEngine:
    def __init__(self, app_id: str = None):
        self.app_id = app_id
        self.cfg = load_config(app_id)
        self.target_day = self.cfg.get("target", {}).get("target_day", 60)
        
        # Load production pipeline for predictions
        try:
            self.pipeline = ProductionPipeline()
        except Exception as e:
            warnings.warn(f"Failed to load production pipeline: {e}")
            self.pipeline = None

    def predict(self, df_slice: pd.DataFrame, window: str, 
                app_id: str = None, campaigns: list = None,
                observation_end_date: pd.Timestamp = None,
                target_day: int = None) -> Tuple[dict, pd.DataFrame]:
        """
        Analyze D60 LTV predictions from validation data
        
        For D60 LTV project:
        - Actual: ltv_d30 and ltv_d60 from data
        - Predicted: Use production pipeline to predict D60 LTV
        - Window determines actual data availability:
          * D1 (Early): Only D1 actual available
          * D3 (Standard): Only D3 actual available  
          * D7 (Accurate): Only D7 actual available
        """
        # Determine effective target day
        effective_target_day = target_day if target_day is not None else self.target_day
        
        # Determine actual data availability based on window
        window_days = {
            'window_d1': 1,
            'window_d3': 3,
            'window_d7': 7
        }
        actual_days_available = window_days.get(window, 7)  # Default to D7

        if df_slice.empty:
            return {}, df_slice
        
        # Check if we have ltv columns
        has_ltv_d30 = 'ltv_d30' in df_slice.columns
        has_ltv_d60 = 'ltv_d60' in df_slice.columns
        
        if not has_ltv_d30 and not has_ltv_d60:
            warnings.warn("No LTV columns found in data")
            return {}, df_slice
        
        # Generate predictions if pipeline is available
        df_with_pred = df_slice.copy()
        
        if self.pipeline is not None:
            try:
                # Batch predict
                print(f"🔍 DEBUG: Calling pipeline.predict_batch with {len(df_slice)} rows")
                print(f"🔍 DEBUG: Columns: {list(df_slice.columns)}")
                results = self.pipeline.predict_batch(df_slice)
                
                # Merge predictions
                df_with_pred['predicted_ltv_d60'] = results['predicted_d60_ltv'].values
                df_with_pred['prediction_method'] = results['method'].values
                df_with_pred['prediction_confidence'] = results['confidence'].values
                print(f"✅ DEBUG: Prediction successful, avg predicted: ${df_with_pred['predicted_ltv_d60'].mean():.4f}")
            except Exception as e:
                print(f"❌ DEBUG: Prediction failed with error: {e}")
                import traceback
                traceback.print_exc()
                warnings.warn(f"Prediction failed: {e}")
                # Fallback: use ltv_d60 if available
                if has_ltv_d60:
                    df_with_pred['predicted_ltv_d60'] = df_slice['ltv_d60']
                    print(f"⚠️ DEBUG: Using actual ltv_d60 as fallback")
                else:
                    df_with_pred['predicted_ltv_d60'] = df_slice['ltv_d30'] * 1.5  # Simple estimate
                    print(f"⚠️ DEBUG: Using ltv_d30 * 1.5 as fallback")
        else:
            # No pipeline: use actual or estimate
            if has_ltv_d60:
                df_with_pred['predicted_ltv_d60'] = df_slice['ltv_d60']
            else:
                df_with_pred['predicted_ltv_d60'] = df_slice['ltv_d30'] * 1.5
        
        # Build curve data for visualization
        metrics = self._calculate_metrics(df_with_pred)
        
        # Add curve data
        metrics['days_labels'] = [f"D{i}" for i in range(0, 61)]
        
        # Actual curve: Only up to actual_days_available (window)
        # Predicted curve: Full D0 -> D60
        actual_series = []
        predicted_series = []
        
        # For actual curve, we interpolate from D0 to window_day using rev_sum
        # After window_day, actual = None (no data available)
        total_installs = df_with_pred['installs'].sum()
        
        if has_ltv_d30 and actual_days_available >= 1:
            # Use rev_sum as proxy for actual revenue up to window day
            actual_rev_at_window = df_with_pred['rev_sum'].mean()
            
            for day in range(0, 61):
                if day == 0:
                    actual_series.append(0)
                    predicted_series.append(0)
                elif day <= actual_days_available:
                    # Linear interpolation from 0 to actual revenue at window day
                    actual_ltv = actual_rev_at_window * (day / actual_days_available)
                    actual_series.append(actual_ltv * total_installs)
                    
                    # Predicted: interpolate to D60
                    pred_d60 = df_with_pred['predicted_ltv_d60'].mean()
                    pred_ltv = pred_d60 * (day / 60)
                    predicted_series.append(pred_ltv * total_installs)
                else:
                    # After window day: no actual data available
                    actual_series.append(None)
                    
                    # Predicted: continue to D60
                    pred_d60 = df_with_pred['predicted_ltv_d60'].mean()
                    pred_ltv = pred_d60 * (day / 60)
                    predicted_series.append(pred_ltv * total_installs)
        else:
            # No actual data at all
            for day in range(0, 61):
                actual_series.append(None if day > 0 else 0)
                pred_d60 = df_with_pred['predicted_ltv_d60'].mean()
                pred_ltv = pred_d60 * (day / 60) if day > 0 else 0
                predicted_series.append(pred_ltv * total_installs)
        
        metrics['actual_series'] = actual_series
        metrics['predicted_series'] = predicted_series
        
        # Add additional summary stats
        if 'install_date' in df_with_pred.columns:
            metrics['min_install_date'] = df_with_pred['install_date'].min().strftime('%d/%m/%Y')
            metrics['max_install_date'] = df_with_pred['install_date'].max().strftime('%d/%m/%Y')
        
        metrics['cohort_count'] = len(df_with_pred)
        metrics['campaigns_count'] = df_with_pred['campaign'].nunique() if 'campaign' in df_with_pred.columns else 1
        
        return metrics, df_with_pred
    
    def _calculate_metrics(self, df: pd.DataFrame) -> dict:
        """Calculate performance metrics"""
        metrics = {}
        
        if 'ltv_d60' in df.columns and 'predicted_ltv_d60' in df.columns:
            # Filter valid predictions
            mask = (df['ltv_d60'] >= 0.10) & (df['predicted_ltv_d60'] > 0) & np.isfinite(df['predicted_ltv_d60'])
            
            if mask.sum() > 0:
                actual = df.loc[mask, 'ltv_d60'].values
                predicted = df.loc[mask, 'predicted_ltv_d60'].values
                
                # MAPE
                mape = np.mean(np.abs((actual - predicted) / actual)) * 100
                
                # MAE
                mae = np.mean(np.abs(actual - predicted))
                
                # R² Score
                ss_res = np.sum((actual - predicted) ** 2)
                ss_tot = np.sum((actual - np.mean(actual)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                metrics['mape'] = mape
                metrics['mae'] = mae
                metrics['r2_score'] = r2
                metrics['n_samples'] = int(mask.sum())
        
        # Aggregate totals
        if 'installs' in df.columns:
            metrics['total_installs'] = int(df['installs'].sum())
        
        if 'cost' in df.columns:
            metrics['total_cost'] = float(df['cost'].sum())
        
        if 'rev_sum' in df.columns:
            metrics['total_revenue_d1'] = float(df['rev_sum'].sum())
        
        if 'predicted_ltv_d60' in df.columns and 'installs' in df.columns:
            metrics['predicted_revenue_d60'] = float((df['predicted_ltv_d60'] * df['installs']).sum())
        
        if 'ltv_d60' in df.columns and 'installs' in df.columns:
            metrics['actual_revenue_d60'] = float((df['ltv_d60'] * df['installs']).sum())
        
        # ROI calculations
        if 'total_cost' in metrics and 'predicted_revenue_d60' in metrics and metrics['total_cost'] > 0:
            metrics['predicted_roi'] = ((metrics['predicted_revenue_d60'] - metrics['total_cost']) / metrics['total_cost']) * 100
        
        if 'total_cost' in metrics and 'actual_revenue_d60' in metrics and metrics['total_cost'] > 0:
            metrics['actual_roi'] = ((metrics['actual_revenue_d60'] - metrics['total_cost']) / metrics['total_cost']) * 100
        
        # ROAS calculations
        if 'total_cost' in metrics and metrics['total_cost'] > 0:
            if 'predicted_revenue_d60' in metrics:
                metrics['predicted_roas'] = metrics['predicted_revenue_d60'] / metrics['total_cost']
            
            if 'actual_revenue_d60' in metrics:
                metrics['actual_roas'] = metrics['actual_revenue_d60'] / metrics['total_cost']
        
        # Add aliases for app.py compatibility
        if 'predicted_revenue_d60' in metrics:
            metrics['total_pred'] = metrics['predicted_revenue_d60']
            metrics['avg_pred'] = metrics['predicted_revenue_d60'] / df['installs'].sum() if 'installs' in df.columns and df['installs'].sum() > 0 else 0
        
        if 'actual_revenue_d60' in metrics:
            metrics['total_actual'] = metrics['actual_revenue_d60']
            metrics['has_actual'] = True
        else:
            metrics['has_actual'] = False
        
        # Add target_day info
        metrics['target_day'] = 60  # D60 LTV project
        metrics['last_day'] = 60  # For chart rendering
        
        return metrics


def get_engine(app_id: str = None) -> PredictionEngine:
    """Factory function to get prediction engine"""
    return PredictionEngine(app_id=app_id)
