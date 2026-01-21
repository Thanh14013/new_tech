# Step 3: Campaign Tier Classification
## Phân Loại App+Campaign Theo Tiers (Chất Lượng Data)

**Thời gian:** 0.5 ngày  
**Độ khó:** ⭐⭐ Trung bình  
**Prerequisites:** Step 2 completed  

---

## 🎯 MỤC TIÊU

1. Phân loại 4,800 combos thành 3 tiers dựa trên:
   - **Coefficient of Variation (CV)** của ltv_d30
   - **Số tháng có data**
   - **Số lượng rows**

2. Tier definitions:
   - **Tier 1 (Premium):** CV ≤ 1.5, ≥3 tháng, ≥1000 rows → Dùng cả 4 methods
   - **Tier 2 (Standard):** 1.5 < CV ≤ 2.5, ≥2 tháng, ≥300 rows → Dùng 3 methods
   - **Tier 3 (Sparse):** Còn lại → Dùng 2 methods + fallback

---

## 📥 INPUT

- `data/processed/clean_data_all.csv` (Clean data từ Step 2)
- `config/config.yaml`

---

## 📤 OUTPUT

- `data/features/campaign_tiers.csv` (Tier classification cho mỗi combo)
- `results/step03_tier_distribution.html` (HTML report)
- `results/step03_tier_summary.csv` (Summary stats)

---

## 🔧 IMPLEMENTATION

### File: `scripts/step03_tier_classification.py`

```python
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

class TierClassifier:
    """Classify app+campaign combos into tiers"""
    
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.tier_config = self.config['tiers']
    
    def load_clean_data(self):
        """Load clean data"""
        path = Path(self.config['data']['processed_path']) / 'clean_data_all.csv'
        df = pd.read_csv(path)
        
        # Convert install_date
        df['install_date'] = pd.to_datetime(df['install_date'])
        
        return df
    
    def calculate_campaign_stats(self, df):
        """Tính toán stats cho mỗi app+campaign"""
        print("\nCalculating campaign statistics...")
        
        # Group by app_id + campaign
        grouped = df.groupby(['app_id', 'campaign']).agg({
            'ltv_d30': ['mean', 'std', 'count', 'median'],
            'month': 'nunique',
            'install_date': ['min', 'max']
        }).reset_index()
        
        # Flatten columns
        grouped.columns = [
            'app_id', 'campaign',
            'ltv_mean', 'ltv_std', 'n_rows', 'ltv_median',
            'n_months', 'first_date', 'last_date'
        ]
        
        # Calculate CV (Coefficient of Variation)
        grouped['cv'] = grouped['ltv_std'] / (grouped['ltv_mean'] + 1e-6)
        
        # Handle infinite/NaN CV
        grouped['cv'] = grouped['cv'].replace([np.inf, -np.inf], 999)
        grouped['cv'] = grouped['cv'].fillna(999)
        
        print(f"✓ Calculated stats for {len(grouped):,} combos")
        
        return grouped
    
    def assign_tiers(self, stats_df):
        """Gán tier cho mỗi combo"""
        print("\nAssigning tiers...")
        
        df = stats_df.copy()
        
        # Initialize tier column
        df['tier'] = 'tier3'  # Default
        
        # Tier 1 criteria
        tier1_mask = (
            (df['cv'] <= self.tier_config['tier1']['cv_threshold']) &
            (df['n_months'] >= self.tier_config['tier1']['min_months']) &
            (df['n_rows'] >= self.config['modeling']['min_rows_tier1'])
        )
        df.loc[tier1_mask, 'tier'] = 'tier1'
        
        # Tier 2 criteria
        tier2_mask = (
            (df['cv'] > self.tier_config['tier1']['cv_threshold']) &
            (df['cv'] <= self.tier_config['tier2']['cv_threshold']) &
            (df['n_months'] >= self.tier_config['tier2']['min_months']) &
            (df['n_rows'] >= self.config['modeling']['min_rows_tier2']) &
            (df['tier'] != 'tier1')  # Not already tier1
        )
        df.loc[tier2_mask, 'tier'] = 'tier2'
        
        # Tier 3: everything else (already set as default)
        
        # Count distribution
        tier_dist = df['tier'].value_counts()
        print(f"\n✓ Tier Distribution:")
        for tier, count in tier_dist.items():
            pct = count / len(df) * 100
            print(f"  - {tier}: {count:,} combos ({pct:.1f}%)")
        
        return df
    
    def add_metadata(self, tier_df):
        """Add metadata columns"""
        df = tier_df.copy()
        
        # Target MAPE by tier
        df['target_mape'] = df['tier'].map({
            'tier1': self.tier_config['tier1']['target_mape'],
            'tier2': self.tier_config['tier2']['target_mape'],
            'tier3': self.tier_config['tier3']['target_mape']
        })
        
        # Recommended methods
        df['recommended_methods'] = df['tier'].map({
            'tier1': 'hurdle,curve_fitting,ml_multiplier,lookalike',
            'tier2': 'hurdle,ml_multiplier,lookalike',
            'tier3': 'lookalike,semantic_fallback'
        })
        
        # Priority score (for training order)
        df['priority_score'] = (
            (df['tier'] == 'tier1').astype(int) * 3 +
            (df['tier'] == 'tier2').astype(int) * 2 +
            (df['tier'] == 'tier3').astype(int) * 1
        ) * df['n_rows']
        
        return df
    
    def save_tiers(self, tier_df):
        """Save tier classification"""
        output_path = Path(self.config['data']['features_path']) / 'campaign_tiers.csv'
        tier_df.to_csv(output_path, index=False)
        print(f"\n✓ Saved tiers: {output_path}")
    
    def generate_report(self, tier_df):
        """Generate HTML report with visualizations"""
        print("\nGenerating tier distribution report...")
        
        # Prepare stats
        tier_summary = tier_df.groupby('tier').agg({
            'n_rows': ['sum', 'mean', 'median'],
            'cv': ['mean', 'median'],
            'n_months': ['mean', 'median'],
            'ltv_mean': ['mean', 'median']
        }).round(2)
        
        tier_summary_html = tier_summary.to_html()
        
        # Create visualizations (saved as base64 for embedding)
        import base64
        from io import BytesIO
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Tier distribution (count)
        tier_counts = tier_df['tier'].value_counts()
        axes[0, 0].bar(tier_counts.index, tier_counts.values, color=['#4CAF50', '#FFC107', '#F44336'])
        axes[0, 0].set_title('Tier Distribution (Combos)', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Number of Combos')
        for i, v in enumerate(tier_counts.values):
            axes[0, 0].text(i, v + 50, str(v), ha='center', fontweight='bold')
        
        # 2. CV distribution by tier
        for tier in ['tier1', 'tier2', 'tier3']:
            tier_data = tier_df[tier_df['tier'] == tier]
            # Cap CV at 10 for visualization
            cv_capped = tier_data['cv'].clip(upper=10)
            axes[0, 1].hist(cv_capped, bins=30, alpha=0.5, label=tier)
        axes[0, 1].set_title('CV Distribution by Tier', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Coefficient of Variation')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].axvline(1.5, color='green', linestyle='--', label='Tier1 threshold')
        axes[0, 1].axvline(2.5, color='orange', linestyle='--', label='Tier2 threshold')
        
        # 3. Rows per tier (box plot)
        tier_df_plot = tier_df.copy()
        tier_df_plot['n_rows_log'] = np.log10(tier_df_plot['n_rows'] + 1)
        axes[1, 0].boxplot([
            tier_df_plot[tier_df_plot['tier'] == 'tier1']['n_rows_log'],
            tier_df_plot[tier_df_plot['tier'] == 'tier2']['n_rows_log'],
            tier_df_plot[tier_df_plot['tier'] == 'tier3']['n_rows_log']
        ], labels=['Tier1', 'Tier2', 'Tier3'])
        axes[1, 0].set_title('Rows per Combo (Log Scale)', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Log10(Rows)')
        
        # 4. Months per tier
        tier_df.groupby('tier')['n_months'].value_counts().unstack(fill_value=0).T.plot(
            kind='bar', stacked=True, ax=axes[1, 1], color=['#4CAF50', '#FFC107', '#F44336']
        )
        axes[1, 1].set_title('Months of Data by Tier', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Number of Months')
        axes[1, 1].set_ylabel('Number of Combos')
        axes[1, 1].legend(title='Tier')
        
        plt.tight_layout()
        
        # Save to BytesIO
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        # HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Tier Classification Report - Step 3</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th {{ background-color: #4CAF50; color: white; padding: 10px; text-align: left; }}
                td {{ border: 1px solid #ddd; padding: 8px; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .tier1 {{ color: #4CAF50; font-weight: bold; }}
                .tier2 {{ color: #FFC107; font-weight: bold; }}
                .tier3 {{ color: #F44336; font-weight: bold; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>Campaign Tier Classification Report</h1>
            
            <h2>1. Tier Definitions</h2>
            <table>
                <tr>
                    <th>Tier</th>
                    <th>CV Threshold</th>
                    <th>Min Months</th>
                    <th>Min Rows</th>
                    <th>Target MAPE</th>
                    <th>Methods</th>
                </tr>
                <tr>
                    <td class="tier1">Tier 1 (Premium)</td>
                    <td>≤ 1.5</td>
                    <td>≥ 3</td>
                    <td>≥ 1,000</td>
                    <td>4%</td>
                    <td>All 4 methods</td>
                </tr>
                <tr>
                    <td class="tier2">Tier 2 (Standard)</td>
                    <td>1.5 - 2.5</td>
                    <td>≥ 2</td>
                    <td>≥ 300</td>
                    <td>6%</td>
                    <td>3 methods (no curve fitting)</td>
                </tr>
                <tr>
                    <td class="tier3">Tier 3 (Sparse)</td>
                    <td>> 2.5</td>
                    <td>Any</td>
                    <td>< 300</td>
                    <td>10%</td>
                    <td>Lookalike + Semantic</td>
                </tr>
            </table>
            
            <h2>2. Tier Summary Statistics</h2>
            {tier_summary_html}
            
            <h2>3. Visualizations</h2>
            <img src="data:image/png;base64,{img_base64}" alt="Tier Distributions">
            
            <h2>4. Key Insights</h2>
            <ul>
                <li><strong>Total Combos:</strong> {len(tier_df):,}</li>
                <li><strong>Tier 1 Count:</strong> {len(tier_df[tier_df['tier'] == 'tier1']):,} ({len(tier_df[tier_df['tier'] == 'tier1']) / len(tier_df) * 100:.1f}%)</li>
                <li><strong>Tier 2 Count:</strong> {len(tier_df[tier_df['tier'] == 'tier2']):,} ({len(tier_df[tier_df['tier'] == 'tier2']) / len(tier_df) * 100:.1f}%)</li>
                <li><strong>Tier 3 Count:</strong> {len(tier_df[tier_df['tier'] == 'tier3']):,} ({len(tier_df[tier_df['tier'] == 'tier3']) / len(tier_df) * 100:.1f}%)</li>
            </ul>
        </body>
        </html>
        """
        
        # Save report
        report_path = Path('results/step03_tier_distribution.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✓ Tier report saved: {report_path}")

def main():
    """Main execution"""
    print("="*60)
    print("STEP 3: CAMPAIGN TIER CLASSIFICATION")
    print("="*60)
    
    # Initialize classifier
    classifier = TierClassifier()
    
    # Load data
    df = classifier.load_clean_data()
    print(f"Loaded {len(df):,} rows")
    
    # Calculate stats
    stats_df = classifier.calculate_campaign_stats(df)
    
    # Assign tiers
    tier_df = classifier.assign_tiers(stats_df)
    
    # Add metadata
    tier_df = classifier.add_metadata(tier_df)
    
    # Save tiers
    classifier.save_tiers(tier_df)
    
    # Generate report
    classifier.generate_report(tier_df)
    
    # Save summary
    summary = tier_df.groupby('tier').agg({
        'app_id': 'count',
        'n_rows': ['sum', 'mean'],
        'cv': 'mean',
        'n_months': 'mean',
        'target_mape': 'first'
    }).round(2)
    
    summary.to_csv('results/step03_tier_summary.csv')
    
    print("\n" + "="*60)
    print("✅ STEP 3 COMPLETED!")
    print("="*60)
    print("\nNext Step: step04_feature_engineering.py")

if __name__ == "__main__":
    main()
```

---

## ✅ SUCCESS CRITERIA

- [x] All 4,800 combos classified into tiers
- [x] Tier 1: ~15-25% of combos
- [x] Tier 2: ~30-40% of combos
- [x] Tier 3: ~35-50% of combos
- [x] HTML report generated

---

## 🎯 NEXT STEP

➡️ **[Step 4: Feature Engineering](step04_feature_engineering.md)**

---

**Estimated Time:** 2-4 hours  
**Difficulty:** ⭐⭐ Medium
