"""
Feature Engineering: Create Cumulative LTV and ROAS
====================================================
Transform discrete revenue columns into cumulative LTV for dashboard display

Author: LTV Prediction System V2.1
Date: 2026-01-21
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def create_cumulative_ltv(df):
    """
    Create cumulative LTV columns from discrete revenue columns
    
    Revenue columns are DISCRETE (revenue only on that day):
    - revenue_d0 = revenue on day 0 only
    - revenue_d1 = revenue on day 1 only
    - revenue_d30 = revenue on day 30 only
    
    LTV columns are CUMULATIVE (total revenue from day 0 to day X):
    - ltv_d0 = revenue_d0
    - ltv_d1 = revenue_d0 + revenue_d1
    - ltv_d30 = revenue_d0 + revenue_d1 + ... + revenue_d30
    """
    
    print("\n📊 Creating Cumulative LTV Columns...")
    
    df_ltv = df.copy()
    
    # Find all revenue columns
    rev_cols = sorted([c for c in df.columns if c.startswith('revenue_d')])
    
    if not rev_cols:
        print("  ⚠ No revenue columns found!")
        return df_ltv
    
    print(f"  Found {len(rev_cols)} revenue columns")
    
    # Extract day numbers
    day_nums = []
    for col in rev_cols:
        try:
            day = int(col.split('_d')[1])
            day_nums.append(day)
        except:
            continue
    
    day_nums = sorted(set(day_nums))
    max_day = max(day_nums) if day_nums else 60
    
    print(f"  Creating cumulative LTV for ALL days: D0 to D{max_day}")
    
    # Create cumulative LTV for ALL days from 0 to max_day
    for target_day in range(max_day + 1):
        # Sum all revenue from day 0 to target_day
        ltv_cols_to_sum = []
        for day in range(target_day + 1):
            col_name = f'revenue_d{day}'
            if col_name in df.columns:
                ltv_cols_to_sum.append(col_name)
        
        if ltv_cols_to_sum:
            df_ltv[f'ltv_d{target_day}'] = df[ltv_cols_to_sum].sum(axis=1)
            if target_day % 10 == 0 or target_day in [1, 3, 7, 14]:
                print(f"  ✓ Created ltv_d{target_day} (sum of {len(ltv_cols_to_sum)} days)")
    
    return df_ltv


def create_roas_columns(df):
    """
    Create ROAS columns from cumulative LTV and cost
    
    ROAS (Return On Ad Spend) = LTV / Cost
    """
    
    print("\n💰 Creating ROAS Columns...")
    
    df_roas = df.copy()
    
    # Find all LTV columns
    ltv_cols = sorted([c for c in df.columns if c.startswith('ltv_d')])
    
    if 'cost' not in df.columns:
        print("  ⚠ Cost column not found! Cannot calculate ROAS")
        return df_roas
    
    if not ltv_cols:
        print("  ⚠ No LTV columns found!")
        return df_roas
    
    print(f"  Found {len(ltv_cols)} LTV columns")
    
    # Create ROAS for ALL LTV columns
    for ltv_col in ltv_cols:
        day = ltv_col.split('_d')[1]
        roas_col = f'roas_d{day}'
        
        # ROAS = LTV / Cost (avoid division by zero)
        df_roas[roas_col] = np.where(
            df_roas['cost'] > 0,
            df_roas[ltv_col] / df_roas['cost'],
            0
        )
    
    # Print only milestone days for readability
    milestone_days = [0, 1, 3, 7, 14, 30, 60]
    for day in milestone_days:
        roas_col = f'roas_d{day}'
        if roas_col in df_roas.columns:
            print(f"  ✓ Created {roas_col}")
    
    print(f"  ✓ Created {len(ltv_cols)} ROAS columns total")
    
    return df_roas


def create_cpi_column(df):
    """
    Create CPI (Cost Per Install) column
    
    CPI = Cost / Installs
    """
    
    print("\n📱 Creating CPI Column...")
    
    df_cpi = df.copy()
    
    if 'cost' not in df.columns or 'installs' not in df.columns:
        print("  ⚠ Cost or Installs column not found!")
        return df_cpi
    
    # CPI = Cost / Installs
    df_cpi['cpi'] = np.where(
        df_cpi['installs'] > 0,
        df_cpi['cost'] / df_cpi['installs'],
        0
    )
    
    print(f"  ✓ Created CPI column")
    
    return df_cpi


def add_features_to_cleaned_data():
    """
    Add cumulative LTV, ROAS, and CPI features to cleaned data
    """
    
    print("="*70)
    print("ADDING CUMULATIVE LTV & ROAS FEATURES TO CLEANED DATA")
    print("="*70)
    print(f"\nDate: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load combined clean data
    clean_file = Path('data/processed/clean_data_all.csv')
    
    if not clean_file.exists():
        print(f"\n❌ Error: {clean_file} not found!")
        print("Please run step02_data_cleaning.py first.")
        return
    
    print(f"\n📂 Loading: {clean_file}")
    df = pd.read_csv(clean_file)
    print(f"  ✓ Loaded: {len(df):,} rows, {len(df.columns)} columns")
    
    # Create cumulative LTV
    df = create_cumulative_ltv(df)
    
    # Create ROAS
    df = create_roas_columns(df)
    
    # Create CPI
    df = create_cpi_column(df)
    
    # Save enhanced data
    print(f"\n💾 Saving enhanced data...")
    output_file = Path('data/processed/clean_data_all_with_ltv.csv')
    df.to_csv(output_file, index=False)
    print(f"  ✓ Saved: {output_file}")
    print(f"  ✓ Total columns: {len(df.columns)} (added {len(df.columns) - pd.read_csv(clean_file).shape[1]} new columns)")
    
    # Show summary
    print(f"\n📊 Summary of New Features:")
    
    ltv_cols = sorted([c for c in df.columns if c.startswith('ltv_d')])
    roas_cols = sorted([c for c in df.columns if c.startswith('roas_d')])
    
    print(f"\n  LTV Columns ({len(ltv_cols)}):")
    for col in ltv_cols:
        mean_val = df[col].mean()
        median_val = df[col].median()
        print(f"    - {col:15s}: Mean=${mean_val:.4f}, Median=${median_val:.4f}")
    
    print(f"\n  ROAS Columns ({len(roas_cols)}):")
    for col in roas_cols:
        mean_val = df[col].mean()
        median_val = df[col].median()
        print(f"    - {col:15s}: Mean={mean_val:.4f}x, Median={median_val:.4f}x")
    
    if 'cpi' in df.columns:
        print(f"\n  CPI:")
        print(f"    - Mean CPI: ${df['cpi'].mean():.4f}")
        print(f"    - Median CPI: ${df['cpi'].median():.4f}")
    
    # Verify cumulative property
    print(f"\n✅ Verification: LTV is cumulative")
    if 'ltv_d30' in df.columns and 'ltv_d7' in df.columns:
        pct_correct = (df['ltv_d30'] >= df['ltv_d7']).sum() / len(df) * 100
        print(f"   ltv_d30 >= ltv_d7: {pct_correct:.1f}% of cases ✓")
    
    if 'ltv_d7' in df.columns and 'ltv_d1' in df.columns:
        pct_correct = (df['ltv_d7'] >= df['ltv_d1']).sum() / len(df) * 100
        print(f"   ltv_d7 >= ltv_d1: {pct_correct:.1f}% of cases ✓")
    
    print("\n" + "="*70)
    print("✅ FEATURE ENGINEERING COMPLETED!")
    print("="*70)
    print(f"\n📁 Output file: {output_file}")
    print(f"   Use this file for dashboard and modeling")
    print("="*70)


if __name__ == "__main__":
    add_features_to_cleaned_data()
