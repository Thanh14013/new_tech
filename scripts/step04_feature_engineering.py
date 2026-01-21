"""
Step 04: Feature Engineering (Core Features)
============================================
Create main features for LTV prediction modeling

Author: LTV Prediction System V2.1
Date: 2026-01-21
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Engineer features for LTV prediction"""
    
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def load_data(self):
        """Load clean data with LTV and tiers"""
        print("\n📂 Loading data...")
        
        # Load clean data WITH LTV columns
        df_path = Path(self.config['data']['processed_path']) / 'clean_data_all_with_ltv.csv'
        
        print(f"  Loading: {df_path}")
        print(f"  ⏳ Loading only required columns...")
        
        # Only load columns needed for feature engineering
        required_cols = [
            'app_id', 'campaign', 'install_date', 'geo',
            'installs', 'cost',
            'revenue_d0', 'revenue_d1', 'revenue_d3', 'revenue_d7',
            'ltv_d0', 'ltv_d1', 'ltv_d3', 'ltv_d7', 'ltv_d14', 'ltv_d30', 'ltv_d60',
            'roas_d30', 'cpi'
        ]
        
        df = pd.read_csv(df_path, usecols=required_cols, low_memory=False)
        df['install_date'] = pd.to_datetime(df['install_date'])
        df['month'] = df['install_date'].dt.to_period('M').astype(str)
        
        print(f"  ✓ Loaded: {len(df):,} rows, {len(df.columns)} columns")
        
        # Load tiers
        tier_path = Path(self.config['data']['features_path']) / 'campaign_tiers.csv'
        tiers = pd.read_csv(tier_path)
        
        print(f"  ✓ Loaded tiers: {len(tiers):,} combos")
        
        # Merge tiers
        df = df.merge(tiers[['app_id', 'campaign', 'tier']], 
                     on=['app_id', 'campaign'], how='left')
        
        # Fill missing tiers (if any)
        df['tier'] = df['tier'].fillna('tier3')
        
        print(f"  ✓ Merged with tiers")
        
        return df
    
    def create_revenue_features(self, df):
        """Create revenue-based features"""
        print("\n[1/4] Creating revenue features...")
        
        df = df.copy()
        
        # 1. rev_sum (sum of revenue D0+D1)
        df['rev_sum'] = df['revenue_d0'] + df['revenue_d1']
        
        # 2. rev_max (max revenue in first 2 days)
        df['rev_max'] = df[['revenue_d0', 'revenue_d1']].max(axis=1)
        
        # 3. rev_last (revenue on day 1)
        df['rev_last'] = df['revenue_d1']
        
        # 4. rev_d0_d1_ratio (D1/D0 ratio)
        df['rev_d0_d1_ratio'] = df['revenue_d1'] / (df['revenue_d0'] + 1e-6)
        df['rev_d0_d1_ratio'] = df['rev_d0_d1_ratio'].clip(upper=10)  # Cap extreme values
        
        # 5. rev_growth_rate (growth rate D0 to D1)
        df['rev_growth_rate'] = (df['revenue_d1'] - df['revenue_d0']) / (df['revenue_d0'] + 1e-6)
        df['rev_growth_rate'] = df['rev_growth_rate'].clip(-1, 10)
        
        # 6. rev_volatility (std of first 2 days)
        df['rev_volatility'] = df[['revenue_d0', 'revenue_d1']].std(axis=1)
        
        print(f"  ✓ Created 6 revenue features")
        
        return df
    
    def create_engagement_features(self, df):
        """Create engagement features (estimated from revenue patterns)"""
        print("\n[2/4] Creating engagement features...")
        
        df = df.copy()
        
        # Since we don't have actual engagement data, we estimate from revenue patterns
        
        # 1. retention_d1 (estimated: users with D1 revenue are likely retained)
        df['retention_d1'] = (df['revenue_d1'] > 0).astype(float) * 0.4  # Assume 40% retention if monetized
        
        # 2. avg_session_time_d1 (estimated from revenue intensity)
        # Higher revenue per retained user suggests longer sessions
        df['avg_session_time_d1'] = (
            df['revenue_d1'] / (df['retention_d1'] + 0.01) * 100
        ).clip(0, 3600)  # Max 1 hour
        
        # 3. avg_level_reached_d1 (estimated from session time)
        # Assume ~2 levels per minute of play
        df['avg_level_reached_d1'] = (
            df['avg_session_time_d1'] / 60 * 2
        ).clip(1, 100)
        
        # 4. engagement_score (composite metric)
        df['engagement_score'] = (
            df['retention_d1'] * 0.5 +
            (df['avg_session_time_d1'] / 1800) * 0.3 +  # Normalize to 30 min
            (df['avg_level_reached_d1'] / 20) * 0.2     # Normalize to level 20
        ).clip(0, 1)
        
        # 5. user_quality_index (revenue per engaged user)
        df['user_quality_index'] = df['revenue_d1'] / (df['retention_d1'] + 0.01)
        df['user_quality_index'] = df['user_quality_index'].clip(0, 10)
        
        print(f"  ✓ Created 5 engagement features (estimated from revenue)")
        
        return df
    
    def create_temporal_features(self, df):
        """Create temporal features"""
        print("\n[3/4] Creating temporal features...")
        
        df = df.copy()
        
        # 1. day_of_week (0=Monday, 6=Sunday)
        df['day_of_week'] = df['install_date'].dt.dayofweek
        
        # 2. is_weekend
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # 3. day_of_month
        df['day_of_month'] = df['install_date'].dt.day
        
        # 4. season (0=Winter, 1=Spring, 2=Summer, 3=Fall)
        month = df['install_date'].dt.month
        df['season'] = ((month % 12 + 3) // 3) % 4
        
        print(f"  ✓ Created 4 temporal features")
        
        return df
    
    def create_aggregated_features(self, df):
        """Create aggregated features (campaign-level stats)"""
        print("\n[4/4] Creating aggregated features...")
        
        df = df.copy()
        
        # Calculate campaign-level means
        print("  Calculating campaign-level statistics...")
        campaign_stats = df.groupby(['app_id', 'campaign'], as_index=False).agg({
            'ltv_d30': 'mean',
            'rev_sum': 'mean',
            'engagement_score': 'mean',
            'installs': 'sum'
        })
        
        campaign_stats.columns = [
            'app_id', 'campaign',
            'campaign_ltv_avg', 'campaign_rev_avg', 
            'campaign_engagement_avg', 'campaign_total_installs'
        ]
        
        # Merge back
        df = df.merge(campaign_stats, on=['app_id', 'campaign'], how='left')
        
        # Relative features (vs campaign average)
        df['rev_vs_campaign_avg'] = df['rev_sum'] / (df['campaign_rev_avg'] + 1e-6)
        df['rev_vs_campaign_avg'] = df['rev_vs_campaign_avg'].clip(0, 5)
        
        df['engagement_vs_campaign_avg'] = df['engagement_score'] / (df['campaign_engagement_avg'] + 1e-6)
        df['engagement_vs_campaign_avg'] = df['engagement_vs_campaign_avg'].clip(0, 5)
        
        print(f"  ✓ Created 6 aggregated features")
        
        return df
    
    def select_final_features(self, df):
        """Select final feature set"""
        print("\n📊 Selecting final features...")
        
        feature_cols = [
            # Identifiers
            'app_id', 'campaign', 'install_date', 'month', 'geo', 'tier',
            
            # Base metrics
            'installs', 'cost', 'cpi',
            
            # Revenue features (6)
            'rev_sum', 'rev_max', 'rev_last', 'rev_d0_d1_ratio', 
            'rev_growth_rate', 'rev_volatility',
            
            # Engagement features (5)
            'retention_d1', 'avg_session_time_d1', 'avg_level_reached_d1',
            'engagement_score', 'user_quality_index',
            
            # Temporal features (4)
            'day_of_week', 'is_weekend', 'day_of_month', 'season',
            
            # Aggregated features (6)
            'campaign_ltv_avg', 'campaign_rev_avg', 'campaign_engagement_avg',
            'campaign_total_installs', 'rev_vs_campaign_avg', 'engagement_vs_campaign_avg',
            
            # Targets (for reference)
            'ltv_d30', 'ltv_d60', 'roas_d30'
        ]
        
        # Filter to existing columns
        existing_cols = [c for c in feature_cols if c in df.columns]
        missing_cols = [c for c in feature_cols if c not in df.columns]
        
        if missing_cols:
            print(f"  ⚠ Missing columns: {missing_cols}")
        
        df_features = df[existing_cols].copy()
        
        print(f"  ✓ Selected {len(existing_cols)} features")
        
        # Count feature types
        feature_types = {
            'Identifiers': 6,
            'Base metrics': 3,
            'Revenue features': 6,
            'Engagement features': 5,
            'Temporal features': 4,
            'Aggregated features': 6,
            'Targets': 3
        }
        
        print(f"\n  Feature breakdown:")
        for ftype, count in feature_types.items():
            print(f"    - {ftype}: {count}")
        
        return df_features
    
    def analyze_correlations(self, df):
        """Analyze feature correlations with target"""
        print("\n📈 Analyzing feature correlations...")
        
        # Select numeric features (exclude identifiers and dates)
        exclude_cols = ['app_id', 'campaign', 'install_date', 'month', 'geo', 'tier', 
                       'ltv_d30', 'ltv_d60', 'roas_d30']
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
        
        if len(numeric_cols) == 0:
            print("  ⚠ No numeric features found")
            return
        
        print(f"  Analyzing {len(numeric_cols)} numeric features...")
        
        # Calculate correlation with ltv_d30
        correlations = df[numeric_cols + ['ltv_d30']].corr()['ltv_d30'].drop('ltv_d30')
        correlations_abs = correlations.abs().sort_values(ascending=False)
        
        # Save feature importance
        importance_df = pd.DataFrame({
            'feature': correlations_abs.index,
            'correlation': correlations[correlations_abs.index].values,
            'abs_correlation': correlations_abs.values
        })
        
        output_path = Path(self.config['data']['features_path']) / 'feature_importance.csv'
        importance_df.to_csv(output_path, index=False)
        
        print(f"\n  ✓ Top 10 features by correlation with ltv_d30:")
        for feat, corr in correlations[correlations_abs.index].head(10).items():
            print(f"    - {feat:35s}: {corr:+.3f}")
        
        print(f"\n  ✓ Feature importance saved: {output_path}")
        
        # Create correlation heatmap
        self.plot_correlation_matrix(df[numeric_cols + ['ltv_d30']])
    
    def plot_correlation_matrix(self, df):
        """Plot correlation matrix"""
        print("\n  Creating correlation matrix visualization...")
        
        # Calculate correlation
        corr_matrix = df.corr()
        
        # Plot
        fig, ax = plt.subplots(figsize=(16, 14))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                   vmin=-1, vmax=1, ax=ax)
        ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        # Save PNG
        output_path = Path('results/step04_feature_correlation.png')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Correlation matrix saved: {output_path}")
        
        # Create HTML report
        import base64
        
        with open(output_path, 'rb') as f:
            img_base64 = base64.b64encode(f.read()).decode()
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Feature Engineering Report - Step 4</title>
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 40px; 
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{ 
            color: #333; 
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #666;
            margin-top: 30px;
        }}
        img {{ 
            max-width: 100%; 
            height: auto; 
            border: 1px solid #ddd;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        .info {{
            background-color: #e8f5e9;
            padding: 15px;
            border-left: 4px solid #4CAF50;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Feature Engineering Report - Step 4</h1>
        <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="info">
            <p><strong>Features Created:</strong></p>
            <ul>
                <li>6 Revenue features (rev_sum, rev_max, rev_last, etc.)</li>
                <li>5 Engagement features (retention, session_time, etc.)</li>
                <li>4 Temporal features (day_of_week, season, etc.)</li>
                <li>6 Aggregated features (campaign averages, relative metrics)</li>
            </ul>
        </div>
        
        <h2>Feature Correlation Matrix</h2>
        <p>This heatmap shows correlations between all numeric features and the target (ltv_d30).</p>
        <img src="data:image/png;base64,{img_base64}" alt="Feature Correlation Matrix">
        
        <h2>Key Insights</h2>
        <ul>
            <li>Features are engineered from revenue patterns (D0, D1) and campaign statistics</li>
            <li>Engagement features are estimated as we don't have actual user behavior data</li>
            <li>Aggregated features capture campaign-level patterns for better predictions</li>
            <li>Check feature_importance.csv for detailed correlation rankings</li>
        </ul>
    </div>
</body>
</html>
"""
        
        html_path = Path('results/step04_feature_correlation.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"  ✓ HTML report saved: {html_path}")


def main():
    """Main execution"""
    print("="*70)
    print("STEP 4: FEATURE ENGINEERING (CORE FEATURES)")
    print("="*70)
    print(f"Started: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize
    fe = FeatureEngineer()
    
    # Load data
    df = fe.load_data()
    print(f"\n✓ Data loaded: {len(df):,} rows")
    
    # Create features
    print("\n🔨 Creating features...")
    df = fe.create_revenue_features(df)
    df = fe.create_engagement_features(df)
    df = fe.create_temporal_features(df)
    df = fe.create_aggregated_features(df)
    
    # Select final features
    df_features = fe.select_final_features(df)
    
    # Analyze correlations
    fe.analyze_correlations(df_features)
    
    # Save features
    print("\n💾 Saving features...")
    output_path = Path(fe.config['data']['features_path']) / 'features_all.csv'
    df_features.to_csv(output_path, index=False)
    print(f"  ✓ Saved: {output_path}")
    print(f"  ✓ Size: {len(df_features):,} rows × {len(df_features.columns)} columns")
    
    # Summary
    print("\n" + "="*70)
    print("✅ STEP 4 COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nFeatures Summary:")
    print(f"  - Total features: {len(df_features.columns)}")
    print(f"  - Total rows: {len(df_features):,}")
    print(f"  - Missing values: {df_features.isnull().sum().sum():,}")
    
    print(f"\nOutputs:")
    print(f"  - Features file: {output_path}")
    print(f"  - Feature importance: data/features/feature_importance.csv")
    print(f"  - Correlation matrix: results/step04_feature_correlation.html")
    
    print(f"\nCompleted: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n➡️  Next Step: step05_cpi_features.py")


if __name__ == "__main__":
    main()
