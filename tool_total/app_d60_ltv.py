"""
D60 LTV Prediction Tool - Streamlit Web Interface
Production-ready web app for Day 60 LTV predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import plotly.graph_objects as go
import plotly.express as px

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tool_total.production_pipeline import ProductionPipeline

# Page config
st.set_page_config(
    page_title="D60 LTV Prediction Tool",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize pipeline
@st.cache_resource
def load_pipeline():
    """Load production pipeline (cached)"""
    with st.spinner("Loading models..."):
        pipeline = ProductionPipeline()
    return pipeline


# Main app
def main():
    st.markdown('<div class="main-header">💰 D60 LTV Prediction Tool</div>', unsafe_allow_html=True)
    st.markdown("**Predict Day 60 Lifetime Value from Day 1 campaign data**")
    
    # Load pipeline
    try:
        pipeline = load_pipeline()
        st.success("✅ Models loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load models: {str(e)}")
        st.info("Please ensure models are trained. Run: `python scripts/step12_advanced_ensemble.py`")
        st.stop()
    
    # Sidebar - Mode selection
    st.sidebar.title("🎯 Prediction Mode")
    mode = st.sidebar.radio(
        "Choose mode:",
        ["Single Campaign", "Batch Processing", "Model Info"],
        index=0
    )
    
    # Mode routing
    if mode == "Single Campaign":
        single_campaign_mode(pipeline)
    elif mode == "Batch Processing":
        batch_processing_mode(pipeline)
    else:
        model_info_mode(pipeline)


def single_campaign_mode(pipeline):
    """Single campaign prediction interface"""
    st.header("🎯 Single Campaign Prediction")
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Campaign Info")
        app_id = st.text_input("App ID", value="com.game.app", help="Application identifier")
        campaign = st.text_input("Campaign Name", value="summer_sale_2024", help="Campaign name")
    
    with col2:
        st.subheader("Day 1 Metrics")
        installs = st.number_input("Installs", min_value=1, value=5000, step=100, help="Number of installs on Day 1")
        cost = st.number_input("Cost ($)", min_value=0.0, value=2500.0, step=100.0, help="Campaign cost in USD")
        revenue = st.number_input("Revenue ($)", min_value=0.0, value=1200.0, step=100.0, help="Day 1 revenue in USD")
    
    # Optional features
    with st.expander("📊 Optional Features (improve accuracy)"):
        col3, col4 = st.columns(2)
        with col3:
            retention_d7 = st.slider("Retention D7", 0.0, 1.0, 0.35, 0.01, help="7-day retention rate")
            retention_d14 = st.slider("Retention D14", 0.0, 1.0, 0.25, 0.01, help="14-day retention rate")
        with col4:
            retention_d30 = st.slider("Retention D30", 0.0, 1.0, 0.15, 0.01, help="30-day retention rate")
            ltv_d30 = st.number_input("LTV D30 ($)", min_value=0.0, value=0.0, step=0.1, help="Optional: 30-day LTV if known")
    
    # Predict button
    if st.button("🔮 Predict D60 LTV", type="primary", use_container_width=True):
        # Prepare data
        data = {
            'installs': installs,
            'cost': cost,
            'revenue': revenue,
            'retention_d7': retention_d7,
            'retention_d14': retention_d14,
            'retention_d30': retention_d30
        }
        
        if ltv_d30 > 0:
            data['ltv_d30'] = ltv_d30
        
        # Predict
        with st.spinner("Predicting..."):
            try:
                result = pipeline.predict(app_id, campaign, data)
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                # Main metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Predicted D60 LTV",
                        f"${result['predicted_d60_ltv']:.2f}",
                        delta=None,
                        help="Predicted Day 60 Lifetime Value per user"
                    )
                
                with col2:
                    confidence_pct = result['confidence'] * 100
                    st.metric(
                        "Confidence",
                        f"{confidence_pct:.0f}%",
                        delta=None,
                        help="Model confidence in prediction"
                    )
                
                with col3:
                    st.metric(
                        "Method",
                        result['method'].replace('_', ' ').title(),
                        delta=None,
                        help="Model used for prediction"
                    )
                
                with col4:
                    st.metric(
                        "Multiplier",
                        f"{result['multiplier']:.2f}x",
                        delta=None,
                        help="Revenue growth multiplier"
                    )
                
                # ROI Analysis
                st.markdown("---")
                st.subheader("💹 ROI Analysis")
                
                predicted_revenue = result['predicted_d60_ltv'] * installs
                profit = predicted_revenue - cost
                roi = (profit / cost * 100) if cost > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Investment", f"${cost:,.2f}")
                
                with col2:
                    st.metric("Expected D60 Revenue", f"${predicted_revenue:,.2f}")
                
                with col3:
                    st.metric("Expected Profit", f"${profit:,.2f}")
                
                with col4:
                    roi_color = "🟢" if roi > 50 else "🟡" if roi > 0 else "🔴"
                    st.metric("ROI", f"{roi_color} {roi:+.1f}%")
                
                # Assessment
                if roi > 50:
                    st.success("💚 **EXCELLENT** - High profitability expected!")
                elif roi > 0:
                    st.info("💛 **GOOD** - Positive return expected")
                else:
                    st.warning("❌ **POOR** - Negative return expected. Consider optimizing campaign.")
                
                # Visualization
                st.markdown("---")
                st.subheader("📈 Revenue Projection")
                
                # Create projection chart
                days = list(range(1, 61))
                base_ltv = revenue / installs if installs > 0 else 0
                projected_ltv = [base_ltv + (result['predicted_d60_ltv'] - base_ltv) * (d/60) for d in days]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=days,
                    y=projected_ltv,
                    mode='lines',
                    name='Projected LTV',
                    line=dict(color='#1f77b4', width=3)
                ))
                fig.add_hline(y=base_ltv, line_dash="dash", line_color="gray", 
                             annotation_text="Day 1 LTV", annotation_position="right")
                fig.add_hline(y=result['predicted_d60_ltv'], line_dash="dash", line_color="green",
                             annotation_text="Day 60 LTV", annotation_position="right")
                
                fig.update_layout(
                    title="LTV Growth Projection (Day 1 → Day 60)",
                    xaxis_title="Days",
                    yaxis_title="LTV per User ($)",
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")


def batch_processing_mode(pipeline):
    """Batch processing interface"""
    st.header("📊 Batch Processing")
    
    st.markdown("""
    Upload a CSV file with campaign data to get predictions for multiple campaigns at once.
    
    **Required columns:**
    - `app_id`: Application identifier
    - `campaign`: Campaign name
    - `installs`: Number of installs
    - `cost`: Campaign cost (USD)
    - `revenue`: Day 1 revenue (USD)
    
    **Optional columns:** `retention_d7`, `retention_d14`, `retention_d30`, `ltv_d30`
    """)
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Loaded {len(df):,} campaigns")
            
            # Show preview
            with st.expander("📋 Data Preview"):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Validate columns
            required_cols = ['app_id', 'campaign', 'installs', 'cost', 'revenue']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                st.stop()
            
            # Predict button
            if st.button("🔮 Predict All", type="primary", use_container_width=True):
                with st.spinner(f"Predicting {len(df):,} campaigns..."):
                    try:
                        results = pipeline.predict_batch(df)
                        
                        # Display summary
                        st.markdown("---")
                        st.subheader("📊 Batch Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Campaigns", f"{len(results):,}")
                        
                        with col2:
                            st.metric("Avg Predicted LTV", f"${results['predicted_d60_ltv'].mean():.2f}")
                        
                        with col3:
                            st.metric("Min LTV", f"${results['predicted_d60_ltv'].min():.2f}")
                        
                        with col4:
                            st.metric("Max LTV", f"${results['predicted_d60_ltv'].max():.2f}")
                        
                        # Method breakdown
                        st.markdown("---")
                        st.subheader("🔧 Methods Used")
                        
                        method_counts = results['method'].value_counts()
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            for method, count in method_counts.items():
                                pct = count / len(results) * 100
                                st.metric(
                                    method.replace('_', ' ').title(),
                                    f"{count:,} ({pct:.1f}%)"
                                )
                        
                        with col2:
                            fig = px.pie(
                                values=method_counts.values,
                                names=[m.replace('_', ' ').title() for m in method_counts.index],
                                title="Method Distribution",
                                hole=0.4
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Results table
                        st.markdown("---")
                        st.subheader("📋 Detailed Results")
                        
                        # Format display
                        display_df = results[[
                            'app_id', 'campaign', 'predicted_d60_ltv', 
                            'method', 'confidence', 'multiplier'
                        ]].copy()
                        
                        display_df['predicted_d60_ltv'] = display_df['predicted_d60_ltv'].apply(lambda x: f"${x:.2f}")
                        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x*100:.0f}%")
                        display_df['multiplier'] = display_df['multiplier'].apply(lambda x: f"{x:.2f}x")
                        display_df['method'] = display_df['method'].apply(lambda x: x.replace('_', ' ').title())
                        
                        st.dataframe(display_df, use_container_width=True, height=400)
                        
                        # Download button
                        csv = results.to_csv(index=False)
                        st.download_button(
                            label="💾 Download Results (CSV)",
                            data=csv,
                            file_name="d60_ltv_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Batch prediction failed: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Failed to load file: {str(e)}")


def model_info_mode(pipeline):
    """Model information interface"""
    st.header("ℹ️ Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Performance Metrics")
        
        metrics_data = {
            "Metric": ["MAPE (Campaign-level)", "MAPE (User-level)", "R² Score", "MAE", "Coverage", "Speed"],
            "Value": ["7.95%", "9.94%", "0.88", "$1.37", "98.3%", "<1 sec/campaign"],
            "Target": ["≤5%", "≤10%", ">0.80", "<$2", ">95%", "<5 sec"],
            "Status": ["🟡 90%", "🟢 Met", "🟢 Exceeded", "🟢 Excellent", "🟢 Exceeded", "🟢 Exceeded"]
        }
        
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("🔧 Model Architecture")
        
        st.markdown("""
        **Primary Model:** ML Multiplier Tuned
        - Algorithm: LightGBM Regressor
        - Features: 29 features
        - Training: GridSearchCV optimized
        - Confidence: 90%
        
        **Fallback Models:**
        1. Semantic Similarity (6,524 campaigns)
           - Method: TF-IDF + cosine similarity
           - Confidence: 70%
        
        2. Global Mean ($5.28)
           - Confidence: 50%
        """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Training Data")
        st.markdown("""
        - **Total Records:** 4,007,623 rows
        - **Training Set:** 69.3% (2,777,686 rows)
        - **Validation Set:** 15.6% (624,172 rows)
        - **Test Set:** 15.1% (605,765 rows)
        - **Features:** 39 engineered features
        - **Models Trained:** 25 total models
        """)
    
    with col2:
        st.subheader("🎯 Use Cases")
        st.markdown("""
        - **Campaign Planning:** Predict ROI before launch
        - **Portfolio Analysis:** Identify top performers
        - **Budget Allocation:** Data-driven ad spend
        - **Risk Assessment:** Early warning for negative ROI
        - **A/B Testing:** Compare campaign strategies
        """)
    
    st.markdown("---")
    st.subheader("📚 Documentation")
    
    st.markdown("""
    For detailed documentation:
    - **User Guide:** `README_TOOL.md`
    - **Project Summary:** `PROJECT_COMPLETE.md`
    - **API Reference:** `tool_total/production_pipeline.py`
    - **Examples:** `scripts/run_production_tool.py`
    
    **Command Line Usage:**
    ```bash
    # Single prediction
    python tool_total/cli.py predict --app-id com.game.app --campaign test --installs 1000 --cost 500 --revenue 200
    
    # Batch processing
    python tool_total/cli.py batch --input campaigns.csv --output predictions.csv
    ```
    """)


if __name__ == "__main__":
    main()
