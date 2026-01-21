"""
Viz Module for Tool Total
"""
from typing import List, Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from PIL import Image
from io import BytesIO

def render_bar_chart(
    days_labels: List[str],
    actual_series: List[float],
    predicted_series: List[float],
    title: str,
    window_label: str,
    target_day: int,
    cohort_count: int,
    total_pred: float,
    total_actual: Optional[float],
    mape: Optional[float],
    mae: Optional[float],
    last_day: int,
    min_install_date: str,
    max_install_date: str,
    window_boundary_date: str,
    y_title: str = "Cumulative Revenue ($)",
    series_names: tuple = ("Predicted LTV", "Actual LTV"),
    show_breakeven: bool = False
) -> go.Figure:
    """Return Plotly figure for Streamlit display"""
    
    # Create Figure
    fig = go.Figure()
    
    # Add Predicted Series (Orange)
    fig.add_trace(go.Bar(
        x=days_labels,
        y=predicted_series,
        name=series_names[0],
        marker_color='#FF8C42',  # Orange
        opacity=0.8
    ))
    
    # Add Actual Series (Green)
    fig.add_trace(go.Bar(
        x=days_labels,
        y=actual_series,
        name=series_names[1],
        marker_color='#4CAF50',  # Green
        opacity=1.0
    ))
    
    # Breakeven Line
    if show_breakeven:
        fig.add_hline(y=1.0, line_dash="dot", line_color="green", annotation_text="Breakeven (1.0x)")
    
    # Add dashed vertical line at prediction window (D7)
    try:
        if f"D{last_day}" in days_labels:
            idx_boundary = days_labels.index(f"D{last_day}")
            fig.add_vline(x=idx_boundary + 0.5, line_width=2, line_dash="dash", line_color="#FF5722")
            
            # Annotation
            max_y = max(
                max(predicted_series) if predicted_series else 0, 
                max([x for x in actual_series if pd.notnull(x)]) if actual_series else 0
            ) * 0.95
            
            fig.add_annotation(
                x=idx_boundary + 0.5,
                y=max_y,
                text=f"Window End (D{last_day})",
                showarrow=False,
                xanchor="left",
                font=dict(color="#FF5722")
            )
    except ValueError:
        pass

    # Title and Layout
    subtitle = f"{window_label} | {cohort_count:,} Cohorts | Install: {min_install_date}-{max_install_date}"
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><sup>{subtitle}</sup>",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Days Since Install",
        yaxis_title=y_title,
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        font=dict(family="sans-serif", size=12, color="black"),
        plot_bgcolor="white",
        margin=dict(l=40, r=40, t=80, b=40),
        width=1000,
        height=500
    )
    
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig


def render_roas_line_chart(
    days_labels: List[str],
    actual_series: List[float],
    predicted_series: List[float],
    title: str,
    window_label: str,
    cohort_count: int,
    min_install_date: str,
    max_install_date: str,
    last_day: int,
    show_breakeven: bool = True
) -> go.Figure:
    """Return Plotly Line Figure for ROAS (Percentage)"""
    
    # Create Figure
    fig = go.Figure()
    
    # Convert to Percentage (0-1 -> 0-100)
    # We'll stick to raw values in data but format the axis to %, 
    # OR multiply everything by 100. User asked for unit to `be %.
    # Let's multiply by 100 for clarity in hover data.
    
    pred_pct = [x * 100 for x in predicted_series]
    act_pct = [x * 100 if pd.notnull(x) else None for x in actual_series]
    
    # Add Predicted Series (Orange Dashed)
    fig.add_trace(go.Scatter(
        x=days_labels,
        y=pred_pct,
        name='Predicted ROAS',
        line=dict(color='#FF8C42', width=3, dash='dash'), # Orange, Dashed
        mode='lines'
    ))
    
    # Add Actual Series (Green Solid)
    fig.add_trace(go.Scatter(
        x=days_labels,
        y=act_pct,
        name='Actual ROAS',
        line=dict(color='#4CAF50', width=3, dash='solid'), # Green, Solid
        mode='lines'
    ))
    
    # Breakeven Line (100%)
    if show_breakeven:
        fig.add_hline(y=100, line_dash="dot", line_color="green", annotation_text="Breakeven (100%)")
    
    # Add vertical line for Window End
    try:
        if f"D{last_day}" in days_labels:
            fig.add_vline(x=days_labels.index(f"D{last_day}"), line_width=1, line_dash="dash", line_color="#FF5722")
            fig.add_annotation(
                x=days_labels.index(f"D{last_day}"),
                y=max(max(pred_pct) if pred_pct else 0, max([x for x in act_pct if x]) if act_pct else 0),
                text=f"Window End (D{last_day})",
                showarrow=False,
                yshift=10,
                font=dict(color="#FF5722")
            )
    except:
        pass

    # Title and Layout
    subtitle = f"{window_label} | {cohort_count:,} Cohorts | Install: {min_install_date}-{max_install_date}"
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><sup>{subtitle}</sup>",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Days Since Install",
        yaxis_title="ROAS (%)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        font=dict(family="sans-serif", size=12, color="black"),
        plot_bgcolor="white",
        margin=dict(l=40, r=40, t=80, b=40),
        width=1000,
        height=500,
        hovermode="x unified"
    )
    
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', ticksuffix="%")
    
    return fig
