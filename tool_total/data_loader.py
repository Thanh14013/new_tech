"""
Data Loader for Tool Total
Unified loader supporting both normal apps and Wool-specific data
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]  # tool_for_long_term/tool_total/data_loader.py -> tool_for_long_term

# Wool app identifier
WOOL_APP_ID = "com.wool.puzzle.game3d"

# Wool-specific hard date constraints
WOOL_HARD_MIN_DATE = dt.date(2025, 7, 1)
WOOL_HARD_MAX_DATE = dt.date(2025, 12, 24)
WOOL_DATA_AVAILABLE_UNTIL = dt.date(2025, 12, 31)


def is_wool_app(app_id: str) -> bool:
    """Check if app is Wool"""
    return app_id == WOOL_APP_ID


def _get_config_path(app_id: str = None) -> Path:
    """Get config path based on app"""
    if app_id and is_wool_app(app_id):
        return ROOT / "scripts_for_wool" / "config_wool.yaml"
    return ROOT / "config" / "config.yaml"


def _get_results_dir(app_id: str = None) -> Path:
    """Get results directory based on app"""
    if app_id and is_wool_app(app_id):
        return ROOT / "results_for_wool"
    return ROOT / "results"


def _get_features_dir(app_id: str = None) -> Path:
    """Get processed features directory based on app"""
    if app_id and is_wool_app(app_id):
        return ROOT / "data" / "processed_for_wool"
    return ROOT / "data" / "processed"


@lru_cache(maxsize=2)
def load_config(app_id: str = None) -> dict:
    cfg_path = _get_config_path(app_id)
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _available_window_files(app_id: str = None) -> List[Path]:
    """Get all prediction files for the appropriate results directory"""
    results_dir = _get_results_dir(app_id)
    if not results_dir.exists():
        return []
    
    # Wool uses different file naming: window_*_predictions.csv
    # Normal uses: window_*_evaluated_predictions.csv
    if app_id and is_wool_app(app_id):
        files = list(results_dir.glob("window_*_predictions.csv"))
    else:
        files = list(results_dir.glob("window_*_evaluated_predictions.csv"))
    return sorted(files)


@lru_cache(maxsize=10)
def load_predictions_file(path: Path) -> pd.DataFrame:
    """Load and parse predictions file"""
    df = pd.read_csv(path)
    if "install_date" in df.columns:
        df["install_date"] = pd.to_datetime(df["install_date"])
    if "app_id" in df.columns:
        df["app_id"] = df["app_id"].astype(str).str.strip()
    if "campaign" in df.columns:
        df["campaign"] = df["campaign"].astype(str).str.strip()
    return df


@lru_cache(maxsize=1)
def build_metadata_normal() -> Dict[str, dict]:
    """Build metadata from normal results/ directory"""
    metadata: Dict[str, dict] = {}
    
    for path in _available_window_files(None):  # Normal mode
        df = load_predictions_file(path)
        if df.empty or "app_id" not in df.columns:
            continue
        
        if "campaign" not in df.columns:
            df["campaign"] = "Unknown"
        
        grouped = df.groupby("app_id")
        for app, g in grouped:
            info = metadata.setdefault(app, {
                "min_date": g["install_date"].min(),
                "max_date": g["install_date"].max(),
                "campaigns": set(),
                "cohorts": 0
            })
            info["min_date"] = min(info["min_date"], g["install_date"].min())
            info["max_date"] = max(info["max_date"], g["install_date"].max())
            info["cohorts"] += len(g)
            
            unique_campaigns = g["campaign"].unique()
            info["campaigns"].update(unique_campaigns)
    
    # Convert sets to sorted lists
    for app in metadata:
        metadata[app]["campaigns"] = sorted(list(metadata[app]["campaigns"]))
    
    return metadata


def available_apps() -> List[str]:
    """Get list of all available apps (including Wool if data exists)"""
    apps = set(build_metadata_normal().keys())
    
    # Add Wool if its results exist
    wool_results = _get_results_dir(WOOL_APP_ID)
    if wool_results.exists() and list(wool_results.glob("window_*_predictions.csv")):
        apps.add(WOOL_APP_ID)
    
    return sorted(apps)


def available_campaigns(app_id: str) -> List[str]:
    """Get campaigns for a specific app"""
    if is_wool_app(app_id):
        # Wool doesn't support campaign filtering
        return []
    
    meta = build_metadata_normal().get(app_id)
    if not meta:
        return []
    return meta["campaigns"]


def default_date_range(app_id: str) -> Tuple[dt.date, dt.date]:
    """Get date range for an app"""
    if is_wool_app(app_id):
        return WOOL_HARD_MIN_DATE, WOOL_HARD_MAX_DATE
    
    meta = build_metadata_normal().get(app_id)
    if not meta:
        today = pd.Timestamp.today().normalize()
        return today.date(), today.date()
    return pd.Timestamp(meta["min_date"]).date(), pd.Timestamp(meta["max_date"]).date()


def get_date_range_for_campaigns(app_id: str, campaigns: List[str] | None) -> Tuple[dt.date, dt.date]:
    """
    Get actual date range for specific app and campaign(s)
    Returns: (min_date, max_date) for the filtered data
    """
    if is_wool_app(app_id):
        return WOOL_HARD_MIN_DATE, WOOL_HARD_MAX_DATE
    
    # Load a reference window file
    files = _available_window_files(None)
    if not files:
        return default_date_range(app_id)
    
    # Use window_d3 as reference (most common)
    ref_file = None
    for f in files:
        if "window_d3" in f.name:
            ref_file = f
            break
    if ref_file is None:
        ref_file = files[0]
    
    df = load_predictions_file(ref_file)
    if df.empty or "app_id" not in df.columns:
        return default_date_range(app_id)
    
    # Filter by app
    mask = df["app_id"] == app_id
    
    # Filter by campaigns if specified
    if campaigns and "All" not in campaigns:
        if "campaign" in df.columns:
            mask &= df["campaign"].isin(campaigns)
    
    filtered = df.loc[mask]
    if filtered.empty:
        return default_date_range(app_id)
    
    min_date = filtered["install_date"].min()
    max_date = filtered["install_date"].max()
    
    return pd.Timestamp(min_date).date(), pd.Timestamp(max_date).date()


def load_data_slice(
    app_id: str,
    start_date: dt.date,
    end_date: dt.date,
    window: str,
    campaigns: List[str] | None
) -> Tuple[pd.DataFrame, dt.date, dt.date, bool]:
    """
    Load predictions for specific app, date range, window, and campaigns
    Also merges actual cumulative revenue curves from features file
    
    Returns:
        (dataframe, used_start_date, used_end_date, was_clamped)
    """
    # Load predictions file for the window
    results_dir = _get_results_dir(app_id)
    
    # Different file naming for Wool vs Normal
    if is_wool_app(app_id):
        pred_file = results_dir / f"{window}_predictions.csv"
    else:
        pred_file = results_dir / f"{window}_evaluated_predictions.csv"
    
    if not pred_file.exists():
        print(f"DEBUG: File not found {pred_file}")
        return pd.DataFrame(), start_date, end_date, False
    
    df = load_predictions_file(pred_file)
    
    if df.empty:
        return df, start_date, end_date, False
    
    # Load features file to get actual cumrev columns
    processed_dir = _get_features_dir(app_id)
    features_file = processed_dir / f"{window}_features.csv"
    
    if features_file.exists():
        # Load features with actual cumrev columns
        df_features = pd.read_csv(features_file)
        if "install_date" in df_features.columns:
            df_features["install_date"] = pd.to_datetime(df_features["install_date"])
        
        # Get actual cumrev columns AND meta columns
        actual_cols = [c for c in df_features.columns if c.startswith("cumrev_d")]
        meta_cols = ["installs", "cost"] + [c for c in df_features.columns if c.startswith("unique_users")]
        actual_cols.extend([c for c in meta_cols if c in df_features.columns])
        
        if actual_cols:
            # Merge actual curves with predictions
            merge_cols = ["install_date", "app_id", "campaign"]
            # Ensure merge columns exist in both
            if all(c in df_features.columns for c in merge_cols):
                df_features_subset = df_features[merge_cols + actual_cols].copy()
                
                # Drop meta columns from df if they exist to avoid duplication/conflicts
                for col in meta_cols:
                    if col in df.columns:
                        df = df.drop(columns=[col])

                # Merge on install_date, app_id, campaign
                df = df.merge(df_features_subset, on=merge_cols, how="left")
    
    # Filter by app
    df = df[df["app_id"] == app_id].copy()
    
    # Filter by campaigns (skip for Wool)
    if campaigns and "campaign" in df.columns and not is_wool_app(app_id):
        df = df[df["campaign"].isin(campaigns)]
    
    # Filter by date range
    df = df[
        (df["install_date"] >= pd.Timestamp(start_date)) &
        (df["install_date"] <= pd.Timestamp(end_date))
    ]
    
    # Check if dates were clamped
    was_clamped = False
    used_start = start_date
    used_end = end_date
    
    if not df.empty:
        actual_min = df["install_date"].min().date()
        actual_max = df["install_date"].max().date()
        
        if actual_min != start_date or actual_max != end_date:
            was_clamped = True
            used_start = actual_min
            used_end = actual_max
    
    return df, used_start, used_end, was_clamped


def get_observation_end_date(app_id: str) -> pd.Timestamp:
    """Get observation end date for an app (for age calculation)"""
    if is_wool_app(app_id):
        return pd.Timestamp(WOOL_DATA_AVAILABLE_UNTIL)
    return pd.Timestamp(2025, 12, 31)  # Default fallback
