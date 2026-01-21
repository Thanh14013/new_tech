# Step 4: Feature Engineering (Core Features)
## Tạo Features Chính Cho Modeling

**Thời gian:** 1 ngày  
**Độ khó:** ⭐⭐⭐ Khó  
**Prerequisites:** Step 3 completed  

---

## 🎯 MỤC TIÊU

Tạo 3 nhóm features chính:

1. **Revenue Features** (6 features):
   - `rev_sum`, `rev_max`, `rev_last`, `rev_d0_d1_ratio`, `rev_growth_rate`, `rev_volatility`

2. **Engagement Features** (5 features):
   - `retention_d1`, `avg_session_time_d1`, `avg_level_reached_d1`, `engagement_score`, `user_quality_index`

3. **Temporal Features** (4 features):
   - `day_of_week`, `is_weekend`, `day_of_month`, `season`

---

## 📥 INPUT

- `data/processed/clean_data_all.csv`
- `data/features/campaign_tiers.csv`
- `config/config.yaml`

---

## 📤 OUTPUT

- `data/features/features_all.csv` (Tất cả features)
- `data/features/feature_importance.csv` (Feature importance ranking)
- `results/step04_feature_correlation.html` (Correlation matrix)

---

## 🔧 IMPLEMENTATION

### File: `scripts/step04_feature_engineering.py`

```python
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureEngineer:
    """Engineer features for LTV prediction"""
    
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def load_data(self):
        """Load clean data and tiers"""
        # Clean data
        df = pd.read_csv(
            Path(self.config['data']['processed_path']) / 'clean_data_all.csv'
        )
        df['install_date'] = pd.to_datetime(df['install_date'])
        
        # Tiers
        tiers = pd.read_csv(
            Path(self.config['data']['features_path']) / 'campaign_tiers.csv'
        )
        
        # Merge
        df = df.merge(tiers[['app_id', 'campaign', 'tier']], 
                     on=['app_id', 'campaign'], how='left')
        
        return df
    
    def create_revenue_features(self, df):
        """Tạo revenue-based features"""
        print("\n[1/3] Creating revenue features...")
        
        df = df.copy()
        
        # 1. rev_sum (tổng revenue D0+D1)
        df['rev_sum'] = df['rev_d0'] + df['rev_d1']
        
        # 2. rev_max (max revenue trong 2 ngày)
        df['rev_max'] = df[['rev_d0', 'rev_d1']].max(axis=1)
        
        # 3. rev_last (revenue ngày cuối)
        df['rev_last'] = df['rev_d1']
        
        # 4. rev_d0_d1_ratio (tỷ lệ D1/D0)
        df['rev_d0_d1_ratio'] = df['rev_d1'] / (df['rev_d0'] + 1e-6)
        df['rev_d0_d1_ratio'] = df['rev_d0_d1_ratio'].clip(upper=10)  # Cap extreme values
        
        # 5. rev_growth_rate (tốc độ tăng trưởng)
        df['rev_growth_rate'] = (df['rev_d1'] - df['rev_d0']) / (df['rev_d0'] + 1e-6)
        df['rev_growth_rate'] = df['rev_growth_rate'].clip(-1, 10)
        
        # 6. rev_volatility (độ biến động)
        df['rev_volatility'] = df[['rev_d0', 'rev_d1']].std(axis=1)
        
        print(f"  ✓ Created 6 revenue features")
        
        return df
    
    def create_engagement_features(self, df):
        """Tạo engagement features"""
        print("\n[2/3] Creating engagement features...")
        
        df = df.copy()
        
        # 1. retention_d1 (nếu chưa có)
        if 'retention_d1' not in df.columns:
            # Estimate từ revenue
            df['retention_d1'] = (df['rev_d1'] > 0).astype(int) * 0.4  # Giả định 40% retention
        
        # 2. avg_session_time_d1 (nếu chưa có)
        if 'avg_session_time_d1' not in df.columns:
            # Estimate từ revenue và retention
            df['avg_session_time_d1'] = (
                df['rev_d1'] / (df['retention_d1'] + 0.01) * 100
            ).clip(0, 3600)  # Max 1 hour
        
        # 3. avg_level_reached_d1 (nếu chưa có)
        if 'avg_level_reached_d1' not in df.columns:
            # Estimate
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
        df['user_quality_index'] = df['rev_d1'] / (df['retention_d1'] + 0.01)
        df['user_quality_index'] = df['user_quality_index'].clip(0, 10)
        
        print(f"  ✓ Created 5 engagement features")
        
        return df
    
    def create_temporal_features(self, df):
        """Tạo temporal features"""
        print("\n[3/3] Creating temporal features...")
        
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
        """Tạo aggregated features (campaign-level stats)"""
        print("\n[4/4] Creating aggregated features...")
        
        df = df.copy()
        
        # Calculate campaign-level means
        campaign_stats = df.groupby(['app_id', 'campaign']).agg({
            'ltv_d30': 'mean',
            'rev_sum': 'mean',
            'engagement_score': 'mean',
            'installs': 'sum'
        }).reset_index()
        
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
        
        print(f"  ✓ Created aggregated features")
        
        return df
    
    def select_final_features(self, df):
        """Select final feature set"""
        
        feature_cols = [
            # Revenue
            'rev_sum', 'rev_max', 'rev_last', 'rev_d0_d1_ratio', 
            'rev_growth_rate', 'rev_volatility',
            
            # Engagement
            'retention_d1', 'avg_session_time_d1', 'avg_level_reached_d1',
            'engagement_score', 'user_quality_index',
            
            # Temporal
            'day_of_week', 'is_weekend', 'day_of_month', 'season',
            
            # Aggregated
            'campaign_ltv_avg', 'campaign_rev_avg', 'campaign_engagement_avg',
            'campaign_total_installs', 'rev_vs_campaign_avg', 'engagement_vs_campaign_avg',
            
            # Metadata
            'app_id', 'campaign', 'country', 'month', 'tier', 'install_date',
            
            # Target
            'ltv_d30', 'ltv_d60'
        ]
        
        # Filter to existing columns
        feature_cols = [c for c in feature_cols if c in df.columns]
        
        df_features = df[feature_cols].copy()
        
        print(f"\n✓ Selected {len(feature_cols)} features")
        
        return df_features
    
    def analyze_correlations(self, df):
        """Analyze feature correlations"""
        print("\nAnalyzing feature correlations...")
        
        # Select numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c not in ['ltv_d30', 'ltv_d60']]
        
        if len(numeric_cols) == 0:
            print("  ⚠ No numeric features found")
            return
        
        # Calculate correlation with target
        correlations = df[numeric_cols + ['ltv_d30']].corr()['ltv_d30'].drop('ltv_d30')
        correlations = correlations.abs().sort_values(ascending=False)
        
        # Save feature importance
        importance_df = pd.DataFrame({
            'feature': correlations.index,
            'correlation': correlations.values
        })
        
        output_path = Path(self.config['data']['features_path']) / 'feature_importance.csv'
        importance_df.to_csv(output_path, index=False)
        
        print(f"  ✓ Top 10 features by correlation:")
        for feat, corr in correlations.head(10).items():
            print(f"    - {feat}: {corr:.3f}")
        
        # Create correlation heatmap
        self.plot_correlation_matrix(df[numeric_cols + ['ltv_d30']])
    
    def plot_correlation_matrix(self, df):
        """Plot correlation matrix"""
        
        # Calculate correlation
        corr_matrix = df.corr()
        
        # Plot
        plt.figure(figsize=(16, 14))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_path = Path('results/step04_feature_correlation.png')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Correlation matrix saved: {output_path}")
        
        # Create HTML report
        import base64
        from io import BytesIO
        
        with open(output_path, 'rb') as f:
            img_base64 = base64.b64encode(f.read()).decode()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Feature Engineering Report - Step 4</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>Feature Correlation Matrix</h1>
            <img src="data:image/png;base64,{img_base64}" alt="Correlation Matrix">
        </body>
        </html>
        """
        
        html_path = Path('results/step04_feature_correlation.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"  ✓ HTML report saved: {html_path}")

def main():
    """Main execution"""
    print("="*60)
    print("STEP 4: FEATURE ENGINEERING")
    print("="*60)
    
    # Initialize
    fe = FeatureEngineer()
    
    # Load data
    print("\nLoading data...")
    df = fe.load_data()
    print(f"✓ Loaded {len(df):,} rows")
    
    # Create features
    print("\nCreating features...")
    df = fe.create_revenue_features(df)
    df = fe.create_engagement_features(df)
    df = fe.create_temporal_features(df)
    df = fe.create_aggregated_features(df)
    
    # Select final features
    df_features = fe.select_final_features(df)
    
    # Analyze correlations
    fe.analyze_correlations(df_features)
    
    # Save features
    output_path = Path(fe.config['data']['features_path']) / 'features_all.csv'
    df_features.to_csv(output_path, index=False)
    print(f"\n✓ Saved features: {output_path}")
    
    print("\n" + "="*60)
    print("✅ STEP 4 COMPLETED!")
    print("="*60)
    print(f"\nTotal features created: {len(df_features.columns)}")
    print("\nNext Step: step05_cpi_features.py")

if __name__ == "__main__":
    main()
```

---

## ✅ SUCCESS CRITERIA

- [x] All features created successfully
- [x] No missing values in features
- [x] Feature correlation matrix generated
- [x] Top features identified (correlation > 0.3)

---

## 🎯 NEXT STEP

➡️ **[Step 5: CPI Features](step05_cpi_features.md)**

---

**Estimated Time:** 6-8 hours  
**Difficulty:** ⭐⭐⭐ Hard
