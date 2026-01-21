"""
Tool Total - Unified LTV Prediction Tool
Merges tool/ and tool_for_wool/ into a single unified interface.

Key Logic:
- When Wool app is selected: Fixed date range (1/7 - 24/12), Wool-specific model
- When other apps selected: Dynamic date range, normal model
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import streamlit as st

# LOCAL IMPORTS
from data_loader import (
    available_apps,
    available_campaigns,
    default_date_range,
    get_date_range_for_campaigns,
    load_data_slice,
    load_config,
    is_wool_app,
    get_observation_end_date,
    _available_window_files,
    WOOL_APP_ID,
    WOOL_HARD_MIN_DATE,
    WOOL_HARD_MAX_DATE,
    WOOL_DATA_AVAILABLE_UNTIL,
)
from prediction_engine import get_engine
from viz import render_bar_chart, render_roas_line_chart

st.set_page_config(page_title="LTV Tool Total", layout="wide")


# --- Cache Wrappers ---
@st.cache_data(show_spinner=False)
def apps_cached():
    return available_apps()

@st.cache_data(show_spinner=False)
def campaigns_cached(app_id: str):
    return available_campaigns(app_id)

@st.cache_data(show_spinner=False)
def date_range_cached(app_id: str):
    return default_date_range(app_id)

@st.cache_data(show_spinner=False, max_entries=256)
def slice_cached(app_id: str, start: dt.date, end: dt.date, window: str, campaigns: tuple | None):
    return load_data_slice(app_id, start, end, window, list(campaigns) if campaigns else None)


# --- UI ---
st.title("🎯 LTV Prediction Tool (Unified)")
st.caption("Extended LTV Analytics with Multi-Horizon Prediction - Supports all apps including Wool")

apps = apps_cached()
if not apps:
    st.error("Data not found. Please run training and prediction scripts first.")
    st.stop()

# Sidebar
st.sidebar.header("Configuration")

# 1. App Selection
# Custom CSS to make Multiselect display clean text
st.markdown("""
<style>
/* 1. Remove background and border from the tag */
div[data-baseweb="select"] span[data-baseweb="tag"] {
    background-color: transparent !important;
    border: none !important;
    padding-left: 0 !important;
}

/* 2. Hide the 'x' button inside the specific tag (keep the main clear button on right) */
div[data-baseweb="select"] span[data-baseweb="tag"] svg {
    display: none !important;
}

/* 3. Adjust text size/padding to look like normal input text */
div[data-baseweb="select"] span[data-baseweb="tag"] span {
    color: white !important; /* Ensure text is visible in dark mode */
    font-size: 1rem;
}
</style>
""", unsafe_allow_html=True)

def enforce_single_selection():
    """Callback to keep only the latest selection"""
    current = st.session_state.get('app_selector', [])
    if len(current) > 1:
        # Keep only the last selected item (the newest one)
        st.session_state.app_selector = [current[-1]]

selected_apps = st.sidebar.multiselect(
    "Select App", 
    apps, 
    key="app_selector",
    default=[apps[0]] if apps else None,
    on_change=enforce_single_selection, # Enforce single selection manually
    format_func=lambda x: f"🧶 {x}" if is_wool_app(x) else x
)

if not selected_apps:
    st.info("👈 Please select an app to proceed.")
    st.stop()

selected_app = selected_apps[0]

# Check if Wool mode
is_wool = is_wool_app(selected_app)

# Show mode indicator
if is_wool:
    st.sidebar.success("🧶 **Wool Mode Active**")
    st.sidebar.caption("Using Wool-specific model & fixed date range")

# 2. Campaign Selection (Disabled for Wool)
if is_wool:
    st.sidebar.markdown("**Campaigns:** All (fixed for Wool)")
    filter_campaigns = None
    display_campaign_label = "All Campaigns"
    campaign_list_for_range = None
else:
    campaigns = campaigns_cached(selected_app)
    campaign_options = ["All"] + campaigns
    selected_campaigns = st.sidebar.multiselect(
        "Select Campaigns", 
        campaign_options, 
        default=["All"]
    )
    
    # Logic: If 'All' is selected, ignore others
    if "All" in selected_campaigns:
        filter_campaigns = None
        display_campaign_label = "All Campaigns"
        campaign_list_for_range = None
    else:
        filter_campaigns = tuple(selected_campaigns)
        display_campaign_label = f"{len(selected_campaigns)} Campaigns"
        campaign_list_for_range = list(selected_campaigns)

# 3. Date Range - Varies by app type
if is_wool:
    # WOOL: Fixed date range
    min_d, max_d = WOOL_HARD_MIN_DATE, WOOL_HARD_MAX_DATE
    
    st.sidebar.info(f"""
    📅 **Data Range (Fixed for Wool)**  
    {min_d.strftime('%d/%m/%Y')} to {max_d.strftime('%d/%m/%Y')}
    """)
    
    default_start = min_d
    default_end = max_d
    
    start_date = st.sidebar.date_input(
        "Start Date",
        value=default_start,
        min_value=min_d,
        max_value=max_d,
        format="DD/MM/YYYY"
    )
    
    end_date = st.sidebar.date_input(
        "End Date", 
        value=default_end,
        min_value=min_d,
        max_value=max_d,
        format="DD/MM/YYYY"
    )
else:
    # NORMAL: Dynamic date range based on data
    min_d, max_d = get_date_range_for_campaigns(selected_app, campaign_list_for_range if 'campaign_list_for_range' in dir() else None)
    
    # Check if data range is sufficient
    data_range_days = (max_d - min_d).days
    if data_range_days < 7:
        st.sidebar.error(f"""
        ⚠️ **Insufficient Data**  
        Available: {min_d.strftime('%d/%m/%Y')} to {max_d.strftime('%d/%m/%Y')} ({data_range_days} days)  
        Need at least 7 days of data.
        """)
        st.error("Cannot display: Selected app/campaign has less than 7 days of data. Please choose different app or campaign.")
        st.stop()
    
    # Adjust max date to ensure window is complete
    conservative_window_days = 7
    max_d_adjusted = max_d - dt.timedelta(days=conservative_window_days)
    
    if max_d_adjusted < min_d:
        st.sidebar.error(f"""
        ⚠️ **Insufficient Data**  
        Available: {min_d.strftime('%d/%m/%Y')} to {max_d.strftime('%d/%m/%Y')}  
        Need at least {conservative_window_days} more days for D7 window completion.
        """)
        st.error("Cannot predict: Not enough data to complete D7 window.")
        st.stop()
    
    st.sidebar.info(f"""
    📅 **Available Cohorts**  
    {min_d.strftime('%d/%m/%Y')} to {max_d_adjusted.strftime('%d/%m/%Y')}
    (Ensures D{conservative_window_days} complete for all windows)
    """)
    
    default_start, default_end = min_d, max_d_adjusted
    
    start_date = st.sidebar.date_input(
        "Start Date",
        value=default_start,
        min_value=min_d,
        max_value=max_d_adjusted,
        format="DD/MM/YYYY"
    )
    
    end_date = st.sidebar.date_input(
        "End Date", 
        value=default_end,
        min_value=min_d,
        max_value=max_d_adjusted,
        format="DD/MM/YYYY"
    )

if start_date > end_date:
    st.sidebar.error("Start date must be before end date")
    st.stop()

# 4. Window Selection
window_options = {
    "D1 (Early)": "window_d1",
    "D3 (Standard)": "window_d3",
    "D7 (Accurate)": "window_d7",
}

available_files = _available_window_files(selected_app)

if is_wool:
    # Wool uses different naming: window_*_predictions.csv
    available_window_names = [f.stem.replace("_predictions", "") for f in available_files]
else:
    # Normal uses: window_*_evaluated_predictions.csv
    available_window_names = [f.stem.replace("_evaluated_predictions", "") for f in available_files]

valid_labels = []
valid_keys = []
for label, key in window_options.items():
    if key in available_window_names:
        valid_labels.append(label)
        valid_keys.append(key)

if not valid_labels:
    st.error("No valid prediction windows available for this app.")
    st.stop()

# Default to D7 for Wool if available, D3 otherwise
default_ix = 0
if is_wool:
    for i, label in enumerate(valid_labels):
        if "D7" in label:
            default_ix = i
            break
elif len(valid_labels) > 1:
    default_ix = 1  # D3 (Standard)

window_label = st.sidebar.radio("Prediction Window", valid_labels, index=default_ix)
window_key = valid_keys[valid_labels.index(window_label)]

# 5. View Mode (LTV vs ROAS)
view_mode = st.sidebar.radio("Display Mode", ["LTV ($)", "ROAS (x)"], index=0)

# Cluster Pattern - Default to "All" (no UI selector)
selected_cluster = "All"

if st.sidebar.button("Run Prediction", type="primary"):
    try:
        print("--- Button Clicked: Run Prediction ---")
        assert window_label is not None, "window_label must be set"
        
        with st.spinner("Loading data & Analyzing predictions..."):
            print("Step 1: Loading Data Slice...")
            # Load Slice
            df_slice, used_start, used_end, clamped = slice_cached(
                selected_app, start_date, end_date, window_key, filter_campaigns if 'filter_campaigns' in dir() else None
            )
            print(f"Step 1 Complete: Loaded {len(df_slice)} rows")
            
            # Apply cluster filter
            if selected_cluster != "All":
                if 'cluster_id' in df_slice.columns:
                    cluster_id = int(selected_cluster)
                    original_count = len(df_slice)
                    df_slice = df_slice[df_slice['cluster_id'] == cluster_id].copy()
                    filtered_count = len(df_slice)
                    st.info(f"🎯 Filtered to Cluster {selected_cluster}: {filtered_count:,} cohorts (from {original_count:,} total)")
                else:
                    st.warning("cluster_id column not found in data. Showing all cohorts.")
            
            if df_slice.empty:
                st.error("No data found for this selection.")
            else:
                # Get engine for this app
                engine = get_engine(selected_app)
                
                # Analyze predictions
                observation_end = pd.Timestamp(WOOL_DATA_AVAILABLE_UNTIL) if is_wool else None
                # Capping logic for Wool Nov/Dec 2025: Removed upon request
                # Prediction handles Actuals truncation internally.
                target_day_override = None

                summary, detail_df = engine.predict(
                    df_slice, 
                    window_key,
                    app_id=selected_app,
                    campaigns=campaign_list_for_range if 'campaign_list_for_range' in dir() else None,
                    observation_end_date=observation_end,
                    target_day=None # Use Default (60)
                )
                
                # Determine display label for target day
                display_target = "D60"
                
                if not summary:
                    st.error("Analysis failed.")
                else:
                    # === VIEW MODE LOGIC ===
                    if view_mode == "ROAS (x)":
                        # Prepare ROAS Data
                        pred_series_viz = summary.get('roas_predicted_series', [])
                        act_series_viz = summary.get('roas_actual_series', [])
                        
                        # Metrics Calculation
                        # Global ROAS = Total Pred / Total Cost
                        total_cost = summary.get('total_cost', 0)
                        roas_pred_val = summary['total_pred'] / total_cost if total_cost > 0 else 0
                        roas_act_val = summary['total_actual'] / total_cost if summary['total_actual'] and total_cost > 0 else None
                        
                        display_metric_1_label = f"Pred ROAS {display_target}"
                        display_metric_1_val = f"{roas_pred_val:.2f}x"
                        
                        display_metric_2_label = f"Actual ROAS {display_target}"
                        display_metric_2_val = f"{roas_act_val:.2f}x" if roas_act_val is not None else "N/A"
                        
                        display_metric_3_label = "Total Cost"
                        display_metric_3_val = f"${total_cost:,.0f}"
                        
                        y_title_viz = "ROAS (x)"
                        series_names_viz = ("Predicted ROAS", "Actual ROAS")
                        show_be_viz = True
                        
                    else:
                        # Standard LTV Mode
                        pred_series_viz = summary['predicted_series']
                        act_series_viz = summary['actual_series']
                        
                        display_metric_1_label = f"Total Predicted {display_target}"
                        display_metric_1_val = f"${summary['total_pred']:,.2f}"
                        
                        display_metric_2_label = f"Total Actual {display_target}"
                        display_metric_2_val = f"${summary['total_actual']:,.2f}" if summary.get('total_actual') is not None else "N/A"
                        
                        display_metric_3_label = "Avg LTV/Cohort"
                        display_metric_3_val = f"${summary['avg_pred']:,.2f}"
                        
                        y_title_viz = "Cumulative Revenue ($)"
                        series_names_viz = ("Predicted LTV", "Actual LTV")
                        show_be_viz = False

                    # Metrics Display
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric(display_metric_1_label, display_metric_1_val)
                    col2.metric(display_metric_2_label, display_metric_2_val)
                    col3.metric(display_metric_3_label, display_metric_3_val)
                    col4.metric("Cohorts", f"{summary['cohort_count']:,}")
                    
                    # Conditional metrics display
                    if summary.get('has_comparison', False):
                        # Comparison mode - have actual data for validation
                        st.markdown(f"### 📊 Prediction Accuracy")
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("MAPE", f"{summary['mape']:.2f}%" if summary['mape'] is not None else "N/A")
                        col2.metric("MAE (per cohort)", f"${summary['mae']:.2f}" if summary['mae'] is not None else "N/A")
                        col3.metric("Target", f"{display_target} Total")
                        
                        st.caption(f"✓ Actual {display_target} data available for validation")
                    else:
                        # Forecast mode - no actual data
                        st.markdown(f"### 🔮 LTV Forecast ({display_target})")
                        st.caption(f"⚠️ No actual {display_target} data available - Pure forecast mode")
                    
                    # Daily bar chart
                    cluster_suffix = f" | Cluster {selected_cluster}" if selected_cluster != "All" else ""
                    app_display = f"🧶 {selected_app}" if is_wool else selected_app
                    print("Step 3: Rendering Plotly Chart...")
                    try:
                        if view_mode == "ROAS (x)":
                            fig = render_roas_line_chart(
                                days_labels=summary['days_labels'],
                                actual_series=act_series_viz,
                                predicted_series=pred_series_viz,
                                title=f"{app_display} - {display_campaign_label}{cluster_suffix}",
                                window_label=window_label,
                                cohort_count=summary['cohort_count'],
                                min_install_date=summary.get('min_install_date', ''),
                                max_install_date=summary.get('max_install_date', ''),
                                last_day=summary['last_day'],
                                show_breakeven=True
                            )
                        else:
                            fig = render_bar_chart(
                                days_labels=summary['days_labels'],
                                actual_series=act_series_viz,
                                predicted_series=pred_series_viz,
                                title=f"{app_display} - {display_campaign_label}{cluster_suffix}",
                                window_label=window_label,
                                target_day=summary['target_day'],
                                cohort_count=summary['cohort_count'],
                                total_pred=summary['total_pred'],
                                total_actual=summary['total_actual'] if summary['has_actual'] else None,
                                mape=summary['mape'],
                                mae=summary['mae'],
                                last_day=summary['last_day'],
                                min_install_date=summary.get('min_install_date', ''),
                                max_install_date=summary.get('max_install_date', ''),
                                window_boundary_date='',
                                y_title=y_title_viz,
                                series_names=series_names_viz,
                                show_breakeven=show_be_viz
                            )
                        print("Step 3 Complete: Plotly Figure Created")
                        st.plotly_chart(fig, use_container_width=True)
                        print("Step 4: st.plotly_chart() completed")
                    except Exception as chart_err:
                        print(f"CHART ERROR: {chart_err}")
                        import traceback
                        traceback.print_exc()
                        st.error(f"Chart rendering failed: {chart_err}")
                    
                    # Detail Table
                    st.subheader("Per-Cohort Details")
                    print("Step 5: Rendering Table...")
                    try:
                        if summary['has_actual']:
                            st.caption(f"Top 20 cohorts by Installs (Comparing Actual vs Predicted at Last Valid Day)")
                            
                            # 0. Format Date
                            if 'install_date' in detail_df.columns:
                                detail_df['install_date'] = pd.to_datetime(detail_df['install_date']).dt.date

                            # 1. Sort by Installs (Top 20)
                            if 'installs' in detail_df.columns:
                                sorted_df = detail_df.sort_values('installs', ascending=False).head(20).copy()
                            else:
                                sorted_df = detail_df.head(20).copy()
                            
                            # 2. Add Display Columns (Merged Act/Pred)
                            # Revenue Cols: Combine "cumrev_dX" and "pred_cumrev_dX"
                            rev_days = sorted([int(c.split('_d')[1]) for c in sorted_df.columns if c.startswith('cumrev_d')])
                            
                            rev_display_cols = []
                            for d in rev_days:
                                col_act = f"cumrev_d{d}"
                                col_pred = f"pred_cumrev_d{d}"
                                
                                # Only process if we have actual data column
                                if col_act in sorted_df.columns:
                                    def format_cell(row):
                                        act = row[col_act]
                                        pred = row[col_pred] if col_pred in row else 0
                                        # Use 2 decimal places for accuracy
                                        if pd.isna(act):
                                            return f"({pred:,.2f})"
                                        else:
                                            return f"{act:,.2f}\n({pred:,.2f})"
                                    
                                    new_col_name = f"Rev D{d}"
                                    sorted_df[new_col_name] = sorted_df.apply(format_cell, axis=1)
                                    rev_display_cols.append(new_col_name)
                            
                            # 3. User Cols
                            user_cols = sorted([c for c in sorted_df.columns if c.startswith('unique_users')], key=lambda x: int(x.split('day')[1]))
                            
                            # 4. Prepare Final Display DataFrame
                            # Fixed Columns: install_date, campaign, installs (Set as Index to "fix" them on left)
                            index_cols = ['install_date', 'campaign', 'installs']
                            available_index = [c for c in index_cols if c in sorted_df.columns]
                            
                            # Other Metrics
                            metric_cols = ['cost', 'last_day', 'pred_roas', 'predicted_ltv_last_day', 'actual_ltv_last_day', 'error', 'pct_error']
                            available_metrics = [c for c in metric_cols if c in sorted_df.columns]
                            
                            # Final Column Order: Index -> Cost -> Rev Display -> Users -> Metrics
                            final_cols = index_cols + available_metrics + rev_display_cols + user_cols
                            available_final_cols = [c for c in final_cols if c in sorted_df.columns]
                            
                            display_df = sorted_df[available_final_cols].copy()
                            
                            # Set Index
                            if available_index:
                                display_df.set_index(available_index, inplace=True)
                            
                            # Formatting for remaining numeric columns
                            format_dict = {
                                'predicted_ltv_last_day': '${:.2f}',
                                'actual_ltv_last_day': '${:.2f}',
                                'error': '${:.2f}',
                                'pct_error': '{:.1f}%',
                                'cost': '${:,.2f}',
                                'pred_roas': '{:.2f}x'
                            }
                            for uc in user_cols:
                                format_dict[uc] = '{:,}'
                            
                            st.dataframe(display_df.style.format(format_dict))
                        else:
                            st.dataframe(detail_df.head(50))
                        print("Step 5 Complete: Table Rendered")
                    except Exception as table_err:
                        print(f"TABLE ERROR: {table_err}")
                        st.error(f"Table rendering failed: {table_err}")

    except Exception as e:
        print(f"CRITICAL ERROR CAUGHT: {e}")
        import traceback
        traceback.print_exc()
        st.error(f"An error occurred: {e}")
        st.code(traceback.format_exc())

    print("--- End of Run Prediction Logic ---")

st.markdown("---")
st.caption("LTV Tool Total | Built with Streamlit | Supports all apps + Wool")
