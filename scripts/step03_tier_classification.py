"""
Step 03: Campaign Tier Classification
======================================
Classify app+campaign combos into 3 tiers based on data quality

Tier 1 (Premium): CV ≤ 1.5, ≥3 months, ≥1000 rows → All 4 methods
Tier 2 (Standard): 1.5 < CV ≤ 2.5, ≥2 months, ≥300 rows → 3 methods
Tier 3 (Sparse): Rest → Lookalike + Semantic fallback

Author: LTV Prediction System V2.1
Date: 2026-01-21
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)


class TierClassifier:
    """Classify app+campaign combos into tiers based on data quality"""
    
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.tier_config = self.config['tiers']
        self.modeling_config = self.config['modeling']
    
    def load_clean_data(self):
        """Load clean data with LTV columns"""
        # FIX: Use clean_data_all_with_ltv.csv instead of clean_data_all.csv
        path = Path(self.config['data']['processed_path']) / 'clean_data_all_with_ltv.csv'
        
        print(f"\n📂 Loading data from: {path}")
        print(f"  ⏳ Loading only required columns (file is 2.5GB)...")
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        # OPTIMIZATION: Only load columns needed for tier classification
        # This reduces memory usage and speeds up loading significantly
        required_cols = ['app_id', 'campaign', 'install_date', 'ltv_d30']
        
        df = pd.read_csv(path, usecols=required_cols, low_memory=False)
        
        # Convert install_date to datetime
        df['install_date'] = pd.to_datetime(df['install_date'])
        
        # FIX: Extract month from install_date
        df['month'] = df['install_date'].dt.to_period('M').astype(str)
        
        print(f"  ✓ Loaded: {len(df):,} rows, {len(df.columns)} columns")
        print(f"  ✓ Verified: ltv_d30 column exists")
        
        return df
    
    def calculate_campaign_stats(self, df):
        """Calculate statistics for each app+campaign combo"""
        print("\n📊 Calculating campaign statistics...")
        
        # Group by app_id + campaign
        grouped = df.groupby(['app_id', 'campaign'], as_index=False).agg({
            'ltv_d30': ['mean', 'std', 'count', 'median'],
            'month': 'nunique',
            'install_date': ['min', 'max']
        })
        
        # Flatten column names
        grouped.columns = [
            'app_id', 'campaign',
            'ltv_mean', 'ltv_std', 'n_rows', 'ltv_median',
            'n_months', 'first_date', 'last_date'
        ]
        
        # Calculate CV (Coefficient of Variation)
        # CV = std / mean (measure of relative variability)
        grouped['cv'] = grouped['ltv_std'] / (grouped['ltv_mean'] + 1e-6)
        
        # Handle infinite/NaN CV values
        # These occur when mean is 0 or std is NaN → assign high CV for tier3
        grouped['cv'] = grouped['cv'].replace([np.inf, -np.inf], 999)
        grouped['cv'] = grouped['cv'].fillna(999)
        
        print(f"  ✓ Calculated stats for {len(grouped):,} app+campaign combos")
        print(f"  ✓ Total rows across all combos: {grouped['n_rows'].sum():,}")
        print(f"  ✓ Average rows per combo: {grouped['n_rows'].mean():.0f}")
        print(f"  ✓ Average CV: {grouped['cv'].mean():.2f}")
        
        return grouped
    
    def assign_tiers(self, stats_df):
        """Assign tier to each combo based on thresholds"""
        print("\n🎯 Assigning tiers based on criteria...")
        
        df = stats_df.copy()
        
        # Initialize tier column (default tier3)
        df['tier'] = 'tier3'
        
        # TIER 1 CRITERIA (Premium Quality)
        tier1_mask = (
            (df['cv'] <= self.tier_config['tier1']['cv_threshold']) &
            (df['n_months'] >= self.tier_config['tier1']['min_months']) &
            (df['n_rows'] >= self.modeling_config['min_rows_tier1'])
        )
        df.loc[tier1_mask, 'tier'] = 'tier1'
        
        # TIER 2 CRITERIA (Standard Quality)
        tier2_mask = (
            (df['cv'] > self.tier_config['tier1']['cv_threshold']) &
            (df['cv'] <= self.tier_config['tier2']['cv_threshold']) &
            (df['n_months'] >= self.tier_config['tier2']['min_months']) &
            (df['n_rows'] >= self.modeling_config['min_rows_tier2']) &
            (df['tier'] != 'tier1')  # Not already classified as tier1
        )
        df.loc[tier2_mask, 'tier'] = 'tier2'
        
        # TIER 3: Everything else (already set as default)
        
        # Count distribution
        tier_dist = df['tier'].value_counts().sort_index()
        
        print(f"\n  ✅ Tier Distribution:")
        for tier in ['tier1', 'tier2', 'tier3']:
            count = tier_dist.get(tier, 0)
            pct = count / len(df) * 100
            total_rows = df[df['tier'] == tier]['n_rows'].sum()
            print(f"    - {tier.upper()}: {count:,} combos ({pct:.1f}%) | {total_rows:,} total rows")
        
        # Special report for Wool app
        wool_app_id = 'com.wool.puzzle.game3d'
        wool_data = df[df['app_id'] == wool_app_id]
        
        if len(wool_data) > 0:
            print(f"\n  📦 WOOL APP Classification:")
            for _, row in wool_data.iterrows():
                print(f"    - Campaign: {row['campaign']}")
                print(f"    - Tier: {row['tier'].upper()}")
                print(f"    - CV: {row['cv']:.2f}")
                print(f"    - Months: {row['n_months']}")
                print(f"    - Rows: {row['n_rows']:,}")
                print(f"    - Status: {'✓ Premium Quality' if row['tier'] == 'tier1' else '✓ Classified as ' + row['tier'].upper()}")
        
        return df
    
    def add_metadata(self, tier_df):
        """Add metadata columns for modeling"""
        df = tier_df.copy()
        
        # Target MAPE by tier
        df['target_mape'] = df['tier'].map({
            'tier1': self.tier_config['tier1']['target_mape'],
            'tier2': self.tier_config['tier2']['target_mape'],
            'tier3': self.tier_config['tier3']['target_mape']
        })
        
        # Recommended methods by tier
        df['recommended_methods'] = df['tier'].map({
            'tier1': 'hurdle,curve_fitting,ml_multiplier,lookalike',
            'tier2': 'hurdle,ml_multiplier,lookalike',
            'tier3': 'lookalike,semantic_fallback'
        })
        
        # Priority score (for training order)
        # Higher tier and more rows = higher priority
        df['priority_score'] = (
            (df['tier'] == 'tier1').astype(int) * 3 +
            (df['tier'] == 'tier2').astype(int) * 2 +
            (df['tier'] == 'tier3').astype(int) * 1
        ) * df['n_rows']
        
        return df
    
    def save_tiers(self, tier_df):
        """Save tier classification to CSV"""
        # Ensure output directory exists
        output_dir = Path(self.config['data']['features_path'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / 'campaign_tiers.csv'
        tier_df.to_csv(output_path, index=False)
        
        print(f"\n💾 Saved tier classification: {output_path}")
        print(f"   Columns: {len(tier_df.columns)}")
        print(f"   Rows: {len(tier_df):,}")
    
    def generate_report(self, tier_df):
        """Generate HTML report with visualizations"""
        print("\n📈 Generating tier distribution report...")
        
        # Prepare summary statistics
        tier_summary = tier_df.groupby('tier').agg({
            'n_rows': ['sum', 'mean', 'median', 'min', 'max'],
            'cv': ['mean', 'median', 'min', 'max'],
            'n_months': ['mean', 'median', 'min', 'max'],
            'ltv_mean': ['mean', 'median']
        }).round(2)
        
        tier_summary_html = tier_summary.to_html(classes='summary-table')
        
        # Create visualizations
        import base64
        from io import BytesIO
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Tier distribution (count)
        tier_counts = tier_df['tier'].value_counts().sort_index()
        colors = ['#4CAF50', '#FFC107', '#F44336']
        axes[0, 0].bar(tier_counts.index, tier_counts.values, color=colors)
        axes[0, 0].set_title('Tier Distribution (Number of Combos)', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Number of Combos', fontsize=12)
        axes[0, 0].set_xlabel('Tier', fontsize=12)
        for i, v in enumerate(tier_counts.values):
            axes[0, 0].text(i, v + 50, f'{v:,}\n({v/len(tier_df)*100:.1f}%)', 
                           ha='center', fontweight='bold', fontsize=10)
        
        # 2. CV distribution by tier
        cv_data = []
        tier_labels = []
        for tier in ['tier1', 'tier2', 'tier3']:
            tier_data = tier_df[tier_df['tier'] == tier]['cv'].clip(upper=10)  # Cap at 10 for viz
            if len(tier_data) > 0:
                cv_data.append(tier_data)
                tier_labels.append(tier.upper())
        
        axes[0, 1].hist(cv_data, bins=30, alpha=0.6, label=tier_labels, color=colors)
        axes[0, 1].set_title('CV Distribution by Tier', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Coefficient of Variation (capped at 10)', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].axvline(1.5, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Tier1 threshold')
        axes[0, 1].axvline(2.5, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Tier2 threshold')
        
        # 3. Rows per tier (box plot with log scale)
        tier_df_plot = tier_df.copy()
        tier_df_plot['n_rows_log'] = np.log10(tier_df_plot['n_rows'] + 1)
        
        box_data = [
            tier_df_plot[tier_df_plot['tier'] == 'tier1']['n_rows_log'],
            tier_df_plot[tier_df_plot['tier'] == 'tier2']['n_rows_log'],
            tier_df_plot[tier_df_plot['tier'] == 'tier3']['n_rows_log']
        ]
        bp = axes[1, 0].boxplot(box_data, labels=['TIER1', 'TIER2', 'TIER3'], patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        axes[1, 0].set_title('Data Volume per Combo (Log Scale)', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Log10(Number of Rows)', fontsize=12)
        axes[1, 0].set_xlabel('Tier', fontsize=12)
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. Months per tier (stacked bar)
        months_data = tier_df.groupby(['tier', 'n_months']).size().unstack(fill_value=0)
        months_data = months_data.reindex(['tier1', 'tier2', 'tier3'])
        months_data.T.plot(kind='bar', stacked=True, ax=axes[1, 1], color=colors)
        axes[1, 1].set_title('Months of Data by Tier', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Number of Months', fontsize=12)
        axes[1, 1].set_ylabel('Number of Combos', fontsize=12)
        axes[1, 1].legend(title='Tier', labels=['TIER1', 'TIER2', 'TIER3'], fontsize=10)
        axes[1, 1].tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        
        # Save to BytesIO and encode as base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        # Generate HTML report
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Tier Classification Report - Step 3</title>
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
            border-bottom: 2px solid #ddd; 
            padding-bottom: 5px; 
            margin-top: 30px;
        }}
        table {{ 
            border-collapse: collapse; 
            width: 100%; 
            margin: 20px 0;
            font-size: 14px;
        }}
        th {{ 
            background-color: #4CAF50; 
            color: white; 
            padding: 12px; 
            text-align: left;
            font-weight: bold;
        }}
        td {{ 
            border: 1px solid #ddd; 
            padding: 10px; 
        }}
        tr:nth-child(even) {{ 
            background-color: #f9f9f9; 
        }}
        .tier1 {{ color: #4CAF50; font-weight: bold; }}
        .tier2 {{ color: #FFC107; font-weight: bold; }}
        .tier3 {{ color: #F44336; font-weight: bold; }}
        img {{ 
            max-width: 100%; 
            height: auto; 
            margin: 20px 0;
            border: 1px solid #ddd;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary-box {{
            background-color: #f0f8ff;
            border-left: 4px solid #4CAF50;
            padding: 15px;
            margin: 20px 0;
        }}
        .summary-table {{
            font-size: 13px;
        }}
        ul {{
            line-height: 1.8;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Campaign Tier Classification Report</h1>
        <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Total Combos Classified:</strong> {len(tier_df):,}</p>
        
        <h2>1. Tier Definitions</h2>
        <table>
            <tr>
                <th>Tier</th>
                <th>CV Threshold</th>
                <th>Min Months</th>
                <th>Min Rows</th>
                <th>Target MAPE</th>
                <th>Recommended Methods</th>
            </tr>
            <tr>
                <td class="tier1">Tier 1 (Premium)</td>
                <td>≤ 1.5</td>
                <td>≥ 3</td>
                <td>≥ 1,000</td>
                <td>4%</td>
                <td>All 4 methods (hurdle, curve, ML, lookalike)</td>
            </tr>
            <tr>
                <td class="tier2">Tier 2 (Standard)</td>
                <td>1.5 - 2.5</td>
                <td>≥ 2</td>
                <td>≥ 300</td>
                <td>6%</td>
                <td>3 methods (hurdle, ML, lookalike)</td>
            </tr>
            <tr>
                <td class="tier3">Tier 3 (Sparse)</td>
                <td>> 2.5 or sparse</td>
                <td>Any</td>
                <td>< 300</td>
                <td>10%</td>
                <td>2 methods (lookalike, semantic fallback)</td>
            </tr>
        </table>
        
        <h2>2. Distribution Summary</h2>
        <div class="summary-box">
            <ul>
                <li><span class="tier1">TIER 1:</span> {len(tier_df[tier_df['tier'] == 'tier1']):,} combos ({len(tier_df[tier_df['tier'] == 'tier1']) / len(tier_df) * 100:.1f}%) | {tier_df[tier_df['tier'] == 'tier1']['n_rows'].sum():,} total rows</li>
                <li><span class="tier2">TIER 2:</span> {len(tier_df[tier_df['tier'] == 'tier2']):,} combos ({len(tier_df[tier_df['tier'] == 'tier2']) / len(tier_df) * 100:.1f}%) | {tier_df[tier_df['tier'] == 'tier2']['n_rows'].sum():,} total rows</li>
                <li><span class="tier3">TIER 3:</span> {len(tier_df[tier_df['tier'] == 'tier3']):,} combos ({len(tier_df[tier_df['tier'] == 'tier3']) / len(tier_df) * 100:.1f}%) | {tier_df[tier_df['tier'] == 'tier3']['n_rows'].sum():,} total rows</li>
            </ul>
        </div>
        
        <h2>3. Tier Summary Statistics</h2>
        {tier_summary_html}
        
        <h2>4. Visualizations</h2>
        <img src="data:image/png;base64,{img_base64}" alt="Tier Distribution Visualizations">
        
        <h2>5. Key Insights</h2>
        <ul>
            <li><strong>Data Coverage:</strong> {tier_df['n_rows'].sum():,} total rows across all combos</li>
            <li><strong>Average CV:</strong> {tier_df['cv'].mean():.2f} (lower is better)</li>
            <li><strong>Premium Quality Rate:</strong> {len(tier_df[tier_df['tier'] == 'tier1']) / len(tier_df) * 100:.1f}% of combos meet tier1 standards</li>
            <li><strong>Data Months Range:</strong> {tier_df['n_months'].min()}-{tier_df['n_months'].max()} months</li>
        </ul>
    </div>
</body>
</html>
"""
        
        # Save HTML report
        report_path = Path('results/step03_tier_distribution.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"  ✓ HTML report saved: {report_path}")
        
        return report_path


def main():
    """Main execution"""
    print("="*70)
    print("STEP 3: CAMPAIGN TIER CLASSIFICATION")
    print("="*70)
    print(f"Started: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize classifier
    classifier = TierClassifier()
    
    # Load data with LTV
    df = classifier.load_clean_data()
    print(f"\n✓ Data loaded: {len(df):,} rows")
    
    # Calculate campaign-level statistics
    stats_df = classifier.calculate_campaign_stats(df)
    
    # Assign tiers based on quality criteria
    tier_df = classifier.assign_tiers(stats_df)
    
    # Add metadata for modeling
    tier_df = classifier.add_metadata(tier_df)
    
    # Save tier classification
    classifier.save_tiers(tier_df)
    
    # Generate HTML report
    report_path = classifier.generate_report(tier_df)
    
    # Save summary CSV
    print("\n📊 Generating summary statistics...")
    summary = tier_df.groupby('tier').agg({
        'app_id': 'count',
        'n_rows': ['sum', 'mean', 'median'],
        'cv': ['mean', 'median'],
        'n_months': ['mean', 'median'],
        'ltv_mean': ['mean', 'median'],
        'target_mape': 'first'
    }).round(2)
    
    summary_path = Path('results/step03_tier_summary.csv')
    summary.to_csv(summary_path)
    print(f"  ✓ Summary CSV saved: {summary_path}")
    
    print("\n" + "="*70)
    print("✅ STEP 3 COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nOutputs:")
    print(f"  - Tier classification: data/features/campaign_tiers.csv")
    print(f"  - HTML report: {report_path}")
    print(f"  - Summary stats: {summary_path}")
    print(f"\nCompleted: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n➡️  Next Step: step04_feature_engineering.py")


if __name__ == "__main__":
    main()
