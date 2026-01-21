"""
STEP 4B: ENHANCED FEATURES - CHURN & RETENTION DECAY
Add advanced features to detect churn and retention decay patterns
"""

import pandas as pd
import numpy as np
from pathlib import Path


def add_churn_decay_features(df):
    """Add churn and retention decay proxy features
    
    Since we don't have daily user-level data, we create proxy features
    from aggregated campaign metrics to detect churn and decay patterns.
    """
    df = df.copy()
    
    print("\n📊 Adding Churn & Decay Features...")
    
    # ========================================================================
    # 1. REVENUE DISTRIBUTION FEATURES (Churn Indicators)
    # ========================================================================
    
    # rev_last_ratio: Proportion of recent revenue
    # Low value → front-loaded revenue → likely churned
    df['rev_last_ratio'] = df['rev_last'] / (df['rev_sum'] + 1e-6)
    df['rev_last_ratio'] = df['rev_last_ratio'].clip(0, 1)
    
    # rev_max_ratio: Peak vs total revenue
    # High value → spike then drop → unstable monetization
    df['rev_max_ratio'] = df['rev_max'] / (df['rev_sum'] + 1e-6)
    df['rev_max_ratio'] = df['rev_max_ratio'].clip(0, 1)
    
    # revenue_concentration: How concentrated is revenue?
    # From volatility: high volatility = concentrated = risky
    df['revenue_concentration'] = df['rev_volatility']
    
    print("   ✅ Revenue distribution features (3)")
    
    # ========================================================================
    # 2. LTV GROWTH FEATURES (Decay Indicators)
    # ========================================================================
    
    # ltv_growth_d30_d60: Relative growth from D30 to D60
    # Low/negative → plateauing/declining → user churning
    df['ltv_growth_d30_d60'] = (df['ltv_d60'] - df['ltv_d30']) / (df['ltv_d30'] + 1e-6)
    df['ltv_growth_d30_d60'] = df['ltv_growth_d30_d60'].clip(-1, 10)
    
    # ltv_growth_absolute: Absolute growth amount
    # Low absolute growth → not much incremental value
    df['ltv_growth_absolute'] = df['ltv_d60'] - df['ltv_d30']
    
    # ltv_decay_rate: Estimate decay rate assuming exponential model
    # ltv(60) = ltv(30) * (60/30)^decay_rate
    # decay_rate = log(ltv(60)/ltv(30)) / log(2)
    # Low decay_rate → slowing down
    with np.errstate(divide='ignore', invalid='ignore'):
        ltv_ratio = df['ltv_d60'] / (df['ltv_d30'] + 1e-6)
        df['ltv_decay_rate'] = np.log(ltv_ratio + 1e-6) / np.log(2)
        df['ltv_decay_rate'] = df['ltv_decay_rate'].fillna(0).clip(-2, 2)
    
    print("   ✅ LTV growth/decay features (3)")
    
    # ========================================================================
    # 3. ENGAGEMENT MONETIZATION (Churn Risk)
    # ========================================================================
    
    # engagement_per_dollar: Engagement relative to revenue
    # High → lots of engagement, low monetization → freemium → will churn
    df['engagement_per_dollar'] = df['engagement_score'] / (df['rev_sum'] + 1e-6)
    df['engagement_per_dollar'] = df['engagement_per_dollar'].clip(0, 100)
    
    # revenue_stability: Inverse of volatility
    # High → stable revenue → less churn risk
    df['revenue_stability'] = 1 - df['rev_volatility'].clip(0, 1)
    
    print("   ✅ Engagement monetization features (2)")
    
    # ========================================================================
    # 4. COMPOSITE CHURN RISK SCORE
    # ========================================================================
    
    # Normalize components to [0, 1]
    rev_last_norm = df['rev_last_ratio'].fillna(0)
    rev_max_norm = df['rev_max_ratio'].fillna(0.5)
    ltv_growth_norm = (df['ltv_growth_d30_d60'].fillna(0) + 1) / 11  # Map [-1, 10] to [0, 1]
    eng_per_dollar_norm = (df['engagement_per_dollar'].fillna(0) / 100).clip(0, 1)
    
    # Churn risk = weighted combination
    # High risk if: low rev_last, high rev_max, low ltv_growth, high eng_per_dollar
    df['churn_risk_score'] = (
        (1 - rev_last_norm) * 0.3 +        # Low recent revenue
        rev_max_norm * 0.2 +                # High spike
        (1 - ltv_growth_norm) * 0.3 +      # Low growth
        eng_per_dollar_norm * 0.2           # High engagement, low revenue
    )
    df['churn_risk_score'] = df['churn_risk_score'].clip(0, 1)
    
    print("   ✅ Composite churn risk score (1)")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    new_features = [
        'rev_last_ratio', 'rev_max_ratio', 'revenue_concentration',
        'ltv_growth_d30_d60', 'ltv_growth_absolute', 'ltv_decay_rate',
        'engagement_per_dollar', 'revenue_stability',
        'churn_risk_score'
    ]
    
    print(f"\n   📊 Total new features: {len(new_features)}")
    print(f"   📊 Feature list: {', '.join(new_features)}")
    
    return df, new_features


def analyze_new_features(df, split_name):
    """Analyze distribution of new features"""
    print(f"\n{'='*80}")
    print(f"📊 FEATURE ANALYSIS: {split_name.upper()}")
    print(f"{'='*80}")
    
    new_features = [
        'rev_last_ratio', 'rev_max_ratio', 'ltv_growth_d30_d60',
        'engagement_per_dollar', 'churn_risk_score'
    ]
    
    for tier in ['tier1', 'tier2']:
        df_tier = df[df['tier'] == tier]
        
        if len(df_tier) == 0:
            continue
        
        print(f"\n{tier.upper()}:")
        print(f"  {'Feature':<30} {'Mean':<10} {'Median':<10} {'P25':<10} {'P75':<10}")
        print(f"  {'-'*70}")
        
        for feat in new_features:
            if feat in df_tier.columns:
                mean = df_tier[feat].mean()
                median = df_tier[feat].median()
                p25 = df_tier[feat].quantile(0.25)
                p75 = df_tier[feat].quantile(0.75)
                
                print(f"  {feat:<30} {mean:<10.4f} {median:<10.4f} {p25:<10.4f} {p75:<10.4f}")


def main():
    """Main execution"""
    print("="*80)
    print("🚀 STEP 4B: ENHANCED FEATURES - CHURN & RETENTION DECAY")
    print("="*80)
    print("\nGoal: Add features to detect churn and retention decay patterns")
    print("      to improve Tier2 prediction accuracy")
    
    # Load existing features
    print("\n📂 Loading existing features...")
    
    splits = ['train', 'validation', 'test']
    enhanced_data = {}
    all_new_features = None
    
    for split in splits:
        input_path = f'data/features/{split}.csv'
        
        if not Path(input_path).exists():
            print(f"   ⚠️  {input_path} not found, skipping...")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing: {split.upper()}")
        print(f"{'='*80}")
        
        df = pd.read_csv(input_path)
        print(f"   Original shape: {df.shape}")
        print(f"   Original features: {len(df.columns)}")
        
        # Add enhanced features
        df_enhanced, new_features = add_churn_decay_features(df)
        
        if all_new_features is None:
            all_new_features = new_features
        
        print(f"\n   Enhanced shape: {df_enhanced.shape}")
        print(f"   Enhanced features: {len(df_enhanced.columns)}")
        
        # Analyze
        analyze_new_features(df_enhanced, split)
        
        # Save
        output_path = f'data/features/{split}_enhanced.csv'
        df_enhanced.to_csv(output_path, index=False)
        print(f"\n   ✅ Saved to: {output_path}")
        
        enhanced_data[split] = df_enhanced
    
    # Summary
    print("\n" + "="*80)
    print("📊 ENHANCEMENT SUMMARY")
    print("="*80)
    
    print(f"\n✅ Enhanced datasets created:")
    for split in splits:
        if split in enhanced_data:
            print(f"   - data/features/{split}_enhanced.csv ({len(enhanced_data[split]):,} rows)")
    
    print(f"\n✅ New features added ({len(all_new_features)}):")
    for i, feat in enumerate(all_new_features, 1):
        print(f"   {i}. {feat}")
    
    print(f"\n📊 Feature Purpose:")
    print(f"   Churn Detection:")
    print(f"      • rev_last_ratio - Recent revenue activity")
    print(f"      • rev_max_ratio - Revenue spike pattern")
    print(f"      • revenue_concentration - Revenue stability")
    
    print(f"\n   Decay Detection:")
    print(f"      • ltv_growth_d30_d60 - Growth momentum")
    print(f"      • ltv_growth_absolute - Absolute growth")
    print(f"      • ltv_decay_rate - Decay speed")
    
    print(f"\n   Risk Assessment:")
    print(f"      • engagement_per_dollar - Monetization efficiency")
    print(f"      • revenue_stability - Revenue consistency")
    print(f"      • churn_risk_score - Composite risk score")
    
    print("\n" + "="*80)
    print("✅ STEP 4B COMPLETED!")
    print("="*80)
    print("\n➡️  Next: Retrain Step 9 ML Multiplier with enhanced features")
    print("➡️  Expected: Tier2 R² improvement from 0.64 to 0.75+")


if __name__ == "__main__":
    main()
