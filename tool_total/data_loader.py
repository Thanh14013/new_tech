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


ROOT = Path(__file__).resolve().parents[1]  # tool_total/data_loader.py -> new_technology

# Wool app identifier (legacy, keep for compatibility)
WOOL_APP_ID = "com.wool.puzzle.game3d"

# Date constraints from our D60 LTV data
WOOL_HARD_MIN_DATE = dt.date(2025, 1, 1)
WOOL_HARD_MAX_DATE = dt.date(2025, 12, 31)
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
    """Get results directory - point to our D60 LTV data"""
    # Always use data/features for our D60 LTV project
    return ROOT / "data" / "features"


def _get_features_dir(app_id: str = None) -> Path:
    """Get processed features directory - point to our D60 LTV data"""
    # Use data/features for our D60 LTV project
    return ROOT / "data" / "features"


@lru_cache(maxsize=2)
def load_config(app_id: str = None) -> dict:
    cfg_path = _get_config_path(app_id)
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _available_window_files(app_id: str = None) -> List[Path]:
    """Get validation data files for D60 LTV predictions"""
    results_dir = _get_results_dir(app_id)
    if not results_dir.exists():
        return []
    
    # Our D60 LTV project uses: validation_enhanced.csv
    # We'll create pseudo "window" files to match app.py expectations
    # Return files with names like: window_d1_evaluated_predictions.csv (as Path objects)
    files = []
    
    # Check for validation file - treat as all windows (D1, D3, D7)
    val_file = results_dir / "validation_enhanced.csv"
    if val_file.exists():
        # Create pseudo paths for window_d1, window_d3, window_d7
        # These don't need to exist physically - we'll intercept in load_predictions_file
        for window in ['window_d1', 'window_d3', 'window_d7']:
            # Create a pseudo path object
            pseudo_path = results_dir / f"{window}_evaluated_predictions.csv"
            files.append(pseudo_path)
    
    return sorted(files)


@lru_cache(maxsize=10)
def load_predictions_file(path: Path) -> pd.DataFrame:
    """Load and parse predictions file"""
    # For D60 LTV project, all window pseudo-paths point to validation_enhanced.csv
    actual_file = path.parent / "validation_enhanced.csv"
    
    if not actual_file.exists():
        # Try test file as fallback
        actual_file = path.parent / "test_enhanced.csv"
    
    if not actual_file.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(actual_file)
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
    Load D60 LTV predictions from validation/test data
    
    Returns:
        (dataframe, used_start_date, used_end_date, was_clamped)
    """
    results_dir = _get_results_dir(app_id)
    
    # For D60 LTV project, we use validation_enhanced.csv as primary data
    pred_file = results_dir / "validation_enhanced.csv"
    
    if not pred_file.exists():
        print(f"DEBUG: File not found {pred_file}")
        return pd.DataFrame(), start_date, end_date, False
    
    df = load_predictions_file(pred_file)
    
    if df.empty:
        return df, start_date, end_date, False
    
    # Filter by app
    df = df[df["app_id"] == app_id].copy()
    
    # Filter by campaigns
    if campaigns and "campaign" in df.columns:
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
