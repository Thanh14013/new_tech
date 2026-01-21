# Step 5: CPI Quality Features
## Tạo CPI-Based Features (V2.1 Enhancement)

**Thời gian:** 0.5 ngày  
**Độ khó:** ⭐⭐ Trung bình  
**Prerequisites:** Step 4 completed  

---

## 🎯 MỤC TIÊU

Tạo 3 CPI quality features (Enhancement #4 trong V2.1):

1. **actual_cpi**: CPI thực tế = cost / installs
2. **cpi_vs_campaign_avg**: So sánh vs campaign average
3. **cpi_quality_score**: Composite score (0-1) đánh giá chất lượng traffic

**Insight:** Premium campaigns (CPI cao + quality traffic) → LTV dự đoán chính xác hơn +15-20%

---

## 📥 INPUT

- `data/features/features_all.csv`
- `config/config.yaml`

---

## 📤 OUTPUT

- `data/features/features_with_cpi.csv` (Features + CPI columns)
- `results/step05_cpi_analysis.html` (CPI distribution report)

---

## 🔧 IMPLEMENTATION

### File: `scripts/step05_cpi_features.py`

```python
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

class CPIFeatureEngineer:
    """Add CPI-based quality features"""
    
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def load_features(self):
        """Load features from Step 4"""
        path = Path(self.config['data']['features_path']) / 'features_all.csv'
        df = pd.read_csv(path)
        print(f"✓ Loaded {len(df):,} rows")
        return df
    
    def calculate_actual_cpi(self, df):
        """Calculate actual CPI"""
        print("\n[1/3] Calculating actual CPI...")
        
        df = df.copy()
        
        # Actual CPI = cost / installs
        df['actual_cpi'] = df['cost'] / (df['installs'] + 1e-6)
        
        # Handle outliers (cap at 99th percentile)
        p99 = df['actual_cpi'].quantile(0.99)
        df['actual_cpi'] = df['actual_cpi'].clip(upper=p99)
        
        # Stats
        mean_cpi = df['actual_cpi'].mean()
        median_cpi = df['actual_cpi'].median()
        
        print(f"  ✓ Mean CPI: ${mean_cpi:.4f}")
        print(f"  ✓ Median CPI: ${median_cpi:.4f}")
        print(f"  ✓ Range: ${df['actual_cpi'].min():.4f} - ${df['actual_cpi'].max():.4f}")
        
        return df
    
    def calculate_cpi_vs_campaign_avg(self, df):
        """Calculate CPI deviation from campaign average"""
        print("\n[2/3] Calculating CPI vs campaign average...")
        
        df = df.copy()
        
        # Calculate campaign-level average CPI
        campaign_cpi = df.groupby(['app_id', 'campaign'])['actual_cpi'].mean().reset_index()
        campaign_cpi.columns = ['app_id', 'campaign', 'campaign_cpi_avg']
        
        # Merge back
        df = df.merge(campaign_cpi, on=['app_id', 'campaign'], how='left')
        
        # Ratio: actual CPI vs campaign average
        df['cpi_vs_campaign_avg'] = df['actual_cpi'] / (df['campaign_cpi_avg'] + 1e-6)
        
        # Clip extreme values
        df['cpi_vs_campaign_avg'] = df['cpi_vs_campaign_avg'].clip(0.1, 5.0)
        
        # Stats
        print(f"  ✓ Avg ratio: {df['cpi_vs_campaign_avg'].mean():.2f}")
        print(f"  ✓ Above avg (ratio > 1.2): {(df['cpi_vs_campaign_avg'] > 1.2).sum():,} rows")
        print(f"  ✓ Below avg (ratio < 0.8): {(df['cpi_vs_campaign_avg'] < 0.8).sum():,} rows")
        
        return df
    
    def calculate_cpi_quality_score(self, df):
        """Calculate CPI quality score (0-1)"""
        print("\n[3/3] Calculating CPI quality score...")
        
        df = df.copy()
        
        # Components:
        # 1. CPI level (higher = premium traffic)
        cpi_percentile = df['actual_cpi'].rank(pct=True)
        
        # 2. CPI stability (lower CV = more stable/quality)
        campaign_cpi_cv = df.groupby(['app_id', 'campaign'])['actual_cpi'].apply(
            lambda x: x.std() / (x.mean() + 1e-6)
        ).reset_index()
        campaign_cpi_cv.columns = ['app_id', 'campaign', 'cpi_cv']
        df = df.merge(campaign_cpi_cv, on=['app_id', 'campaign'], how='left')
        
        # Invert CV (lower CV = higher score)
        cpi_stability_score = 1 / (1 + df['cpi_cv'])
        
        # 3. Volume (higher installs = proven traffic)
        volume_percentile = df['campaign_total_installs'].rank(pct=True)
        
        # Composite score (weighted average)
        df['cpi_quality_score'] = (
            cpi_percentile * 0.4 +           # 40% weight on CPI level
            cpi_stability_score * 0.4 +      # 40% weight on stability
            volume_percentile * 0.2          # 20% weight on volume
        ).clip(0, 1)
        
        # Categorize quality
        df['cpi_quality_tier'] = pd.cut(
            df['cpi_quality_score'],
            bins=[0, 0.33, 0.67, 1.0],
            labels=['Low', 'Medium', 'High']
        )
        
        # Stats
        print(f"  ✓ Quality distribution:")
        for tier, count in df['cpi_quality_tier'].value_counts().items():
            pct = count / len(df) * 100
            print(f"    - {tier}: {count:,} ({pct:.1f}%)")
        
        return df
    
    def analyze_cpi_ltv_relationship(self, df):
        """Analyze CPI vs LTV relationship"""
        print("\nAnalyzing CPI-LTV relationship...")
        
        # Correlation
        corr_cpi = df[['actual_cpi', 'ltv_d30']].corr().iloc[0, 1]
        corr_quality = df[['cpi_quality_score', 'ltv_d30']].corr().iloc[0, 1]
        
        print(f"  - Correlation (CPI vs LTV): {corr_cpi:.3f}")
        print(f"  - Correlation (Quality Score vs LTV): {corr_quality:.3f}")
        
        # Group by quality tier
        tier_stats = df.groupby('cpi_quality_tier').agg({
            'actual_cpi': 'mean',
            'ltv_d30': 'mean',
            'app_id': 'count'
        }).round(4)
        
        tier_stats.columns = ['Avg CPI', 'Avg LTV', 'Count']
        
        print(f"\n  Average metrics by CPI quality:")
        print(tier_stats)
        
        return tier_stats
    
    def plot_cpi_analysis(self, df):
        """Create CPI analysis plots"""
        print("\nCreating CPI visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. CPI distribution
        axes[0, 0].hist(df['actual_cpi'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(df['actual_cpi'].mean(), color='red', linestyle='--', label=f'Mean: ${df["actual_cpi"].mean():.4f}')
        axes[0, 0].axvline(df['actual_cpi'].median(), color='orange', linestyle='--', label=f'Median: ${df["actual_cpi"].median():.4f}')
        axes[0, 0].set_title('CPI Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Actual CPI ($)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # 2. CPI vs LTV scatter
        sample = df.sample(min(10000, len(df)))
        axes[0, 1].scatter(sample['actual_cpi'], sample['ltv_d30'], alpha=0.3, s=10)
        axes[0, 1].set_title('CPI vs LTV Relationship', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Actual CPI ($)')
        axes[0, 1].set_ylabel('LTV D30 ($)')
        
        # Add trend line
        z = np.polyfit(sample['actual_cpi'], sample['ltv_d30'], 1)
        p = np.poly1d(z)
        axes[0, 1].plot(sample['actual_cpi'], p(sample['actual_cpi']), "r--", alpha=0.8, linewidth=2)
        
        # 3. Quality score distribution
        axes[1, 0].hist(df['cpi_quality_score'], bins=50, color='green', edgecolor='black', alpha=0.7)
        axes[1, 0].set_title('CPI Quality Score Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Quality Score')
        axes[1, 0].set_ylabel('Frequency')
        
        # 4. LTV by quality tier
        df.boxplot(column='ltv_d30', by='cpi_quality_tier', ax=axes[1, 1])
        axes[1, 1].set_title('LTV Distribution by CPI Quality Tier', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('CPI Quality Tier')
        axes[1, 1].set_ylabel('LTV D30 ($)')
        plt.sca(axes[1, 1])
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        
        # Save
        output_path = Path('results/step05_cpi_analysis.png')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Plots saved: {output_path}")
        
        # Create HTML report
        self.create_html_report(output_path, df)
    
    def create_html_report(self, img_path, df):
        """Create HTML report"""
        
        import base64
        
        with open(img_path, 'rb') as f:
            img_base64 = base64.b64encode(f.read()).decode()
        
        # Calculate stats
        tier_stats = df.groupby('cpi_quality_tier').agg({
            'actual_cpi': ['mean', 'median'],
            'ltv_d30': ['mean', 'median'],
            'app_id': 'count'
        }).round(4)
        
        tier_stats_html = tier_stats.to_html()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CPI Features Report - Step 5</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th {{ background-color: #4CAF50; color: white; padding: 10px; text-align: left; }}
                td {{ border: 1px solid #ddd; padding: 8px; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>CPI Quality Features Report</h1>
            
            <h2>1. Overview</h2>
            <ul>
                <li><strong>Total Rows:</strong> {len(df):,}</li>
                <li><strong>Mean CPI:</strong> ${df['actual_cpi'].mean():.4f}</li>
                <li><strong>Median CPI:</strong> ${df['actual_cpi'].median():.4f}</li>
                <li><strong>CPI Range:</strong> ${df['actual_cpi'].min():.4f} - ${df['actual_cpi'].max():.4f}</li>
            </ul>
            
            <h2>2. CPI Quality Distribution</h2>
            <ul>
                <li><strong>High Quality:</strong> {(df['cpi_quality_tier'] == 'High').sum():,} rows ({(df['cpi_quality_tier'] == 'High').sum() / len(df) * 100:.1f}%)</li>
                <li><strong>Medium Quality:</strong> {(df['cpi_quality_tier'] == 'Medium').sum():,} rows ({(df['cpi_quality_tier'] == 'Medium').sum() / len(df) * 100:.1f}%)</li>
                <li><strong>Low Quality:</strong> {(df['cpi_quality_tier'] == 'Low').sum():,} rows ({(df['cpi_quality_tier'] == 'Low').sum() / len(df) * 100:.1f}%)</li>
            </ul>
            
            <h2>3. Statistics by Quality Tier</h2>
            {tier_stats_html}
            
            <h2>4. Visualizations</h2>
            <img src="data:image/png;base64,{img_base64}" alt="CPI Analysis">
            
            <h2>5. Key Insights</h2>
            <ul>
                <li>Premium campaigns (high CPI + quality score) have {df[df['cpi_quality_tier'] == 'High']['ltv_d30'].mean() / df[df['cpi_quality_tier'] == 'Low']['ltv_d30'].mean():.2f}x higher LTV than low quality</li>
                <li>CPI quality score correlates {df[['cpi_quality_score', 'ltv_d30']].corr().iloc[0, 1]:.3f} with LTV</li>
                <li>Adding CPI features expected to improve MAPE by 15-20% for premium campaigns</li>
            </ul>
        </body>
        </html>
        """
        
        html_path = Path('results/step05_cpi_analysis.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"  ✓ HTML report saved: {html_path}")

def main():
    """Main execution"""
    print("="*60)
    print("STEP 5: CPI QUALITY FEATURES")
    print("="*60)
    
    # Initialize
    cpi_fe = CPIFeatureEngineer()
    
    # Load features
    df = cpi_fe.load_features()
    
    # Calculate CPI features
    df = cpi_fe.calculate_actual_cpi(df)
    df = cpi_fe.calculate_cpi_vs_campaign_avg(df)
    df = cpi_fe.calculate_cpi_quality_score(df)
    
    # Analyze relationship
    cpi_fe.analyze_cpi_ltv_relationship(df)
    
    # Plot analysis
    cpi_fe.plot_cpi_analysis(df)
    
    # Save
    output_path = Path(cpi_fe.config['data']['features_path']) / 'features_with_cpi.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved features with CPI: {output_path}")
    
    print("\n" + "="*60)
    print("✅ STEP 5 COMPLETED!")
    print("="*60)
    print("\nNext Step: step06_train_test_split.py")

if __name__ == "__main__":
    main()
```

---

## ✅ SUCCESS CRITERIA

- [x] CPI features created for all rows
- [x] Quality score distribution: ~33% each tier
- [x] Positive correlation between quality score and LTV
- [x] HTML report generated

---

## 🎯 NEXT STEP

➡️ **[Step 6: Train/Test Split](step06_train_test_split.md)**

---

**Estimated Time:** 2-4 hours  
**Difficulty:** ⭐⭐ Medium
