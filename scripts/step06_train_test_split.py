"""
Step 6: Train/Test Split
=========================
Split data temporally: M1-M10 (train), M11 (validation), M12 (test)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml


class DataSplitter:
    """Split data into train/val/test by temporal order"""
    
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def load_features(self):
        """Load features with CPI"""
        path = Path(self.config['data']['features_path']) / 'features_with_cpi.csv'
        print(f"Loading features from {path}...")
        
        df = pd.read_csv(path)
        print(f"✓ Loaded {len(df):,} rows × {len(df.columns)} columns")
        
        return df
    
    def add_month_column(self, df):
        """Extract month from install_date and create M1-M12 labels"""
        print("\nExtracting month information...")
        
        df = df.copy()
        df['install_date'] = pd.to_datetime(df['install_date'])
        
        # Extract month number (1-12)
        df['month_num'] = df['install_date'].dt.month
        
        # Create M1-M12 labels
        df['month'] = 'M' + df['month_num'].astype(str)
        
        # Show distribution
        print(f"\nMonth distribution:")
        month_dist = df['month'].value_counts().sort_index()
        for month in [f'M{i}' for i in range(1, 13)]:
            if month in month_dist.index:
                count = month_dist[month]
                pct = count / len(df) * 100
                print(f"  {month:4s}: {count:,} rows ({pct:.1f}%)")
        
        return df
    
    def split_by_month(self, df):
        """Split by month: M1-M10 train, M11 validation, M12 test"""
        print("\n" + "="*70)
        print("SPLITTING DATA BY MONTH")
        print("="*70)
        
        train_months = self.config['data']['train_months']  # M1-M10
        val_month = self.config['data']['validation_month']  # M11
        test_months = self.config['data']['test_months']  # [M12]
        
        # Train: M1-M10 (Jan-Oct)
        df_train = df[df['month'].isin(train_months)].copy()
        print(f"\n✓ Train set (M1-M10):")
        print(f"    {len(df_train):,} rows ({len(df_train)/len(df)*100:.1f}%)")
        
        # Validation: M11 (Nov)
        df_val = df[df['month'] == val_month].copy()
        print(f"\n✓ Validation set (M11):")
        print(f"    {len(df_val):,} rows ({len(df_val)/len(df)*100:.1f}%)")
        
        # Test: M12 (Dec)
        df_test = df[df['month'].isin(test_months)].copy()
        print(f"\n✓ Test set (M12):")
        print(f"    {len(df_test):,} rows ({len(df_test)/len(df)*100:.1f}%)")
        
        # Verify no missing data
        total_split = len(df_train) + len(df_val) + len(df_test)
        if total_split != len(df):
            print(f"\n⚠ Warning: Split total ({total_split:,}) != Original ({len(df):,})")
        else:
            print(f"\n✓ Split verification: {total_split:,} = {len(df):,} rows")
        
        return df_train, df_val, df_test
    
    def validate_split(self, df_train, df_val, df_test):
        """Validate split quality"""
        print("\n" + "="*70)
        print("VALIDATING SPLIT QUALITY")
        print("="*70)
        
        # 1. Check campaign coverage
        print("\n1️⃣  Campaign Coverage:")
        train_combos = set(zip(df_train['app_id'], df_train['campaign']))
        val_combos = set(zip(df_val['app_id'], df_val['campaign']))
        test_combos = set(zip(df_test['app_id'], df_test['campaign']))
        
        print(f"  Train campaigns: {len(train_combos):,}")
        print(f"  Val campaigns: {len(val_combos):,}")
        print(f"  Test campaigns: {len(test_combos):,}")
        
        # Campaigns in val/test that were also in train (good for temporal prediction)
        val_in_train = len(val_combos & train_combos)
        test_in_train = len(test_combos & train_combos)
        
        print(f"\n  Val campaigns also in train: {val_in_train} ({val_in_train/len(val_combos)*100:.1f}%)")
        print(f"  Test campaigns also in train: {test_in_train} ({test_in_train/len(test_combos)*100:.1f}%)")
        
        # 2. Check tier distribution
        print("\n2️⃣  Tier Distribution:")
        
        for dataset_name, dataset in [('Train', df_train), ('Validation', df_val), ('Test', df_test)]:
            tier_dist = dataset['tier'].value_counts(normalize=True).sort_index() * 100
            
            print(f"\n  {dataset_name}:")
            for tier in ['tier1', 'tier2', 'tier3']:
                if tier in tier_dist.index:
                    pct = tier_dist[tier]
                    count = (dataset['tier'] == tier).sum()
                    print(f"    {tier}: {pct:5.1f}% ({count:,} rows)")
        
        # 3. Check target variable distribution
        print("\n3️⃣  Target Variable (ltv_d30) Distribution:")
        
        for dataset_name, dataset in [('Train', df_train), ('Validation', df_val), ('Test', df_test)]:
            if 'ltv_d30' in dataset.columns:
                mean_ltv = dataset['ltv_d30'].mean()
                median_ltv = dataset['ltv_d30'].median()
                std_ltv = dataset['ltv_d30'].std()
                
                print(f"\n  {dataset_name}:")
                print(f"    Mean: ${mean_ltv:.2f}")
                print(f"    Median: ${median_ltv:.2f}")
                print(f"    Std: ${std_ltv:.2f}")
        
        # 4. Check no temporal leakage
        print("\n4️⃣  Temporal Validation:")
        
        train_max_date = df_train['install_date'].max()
        val_min_date = df_val['install_date'].min()
        val_max_date = df_val['install_date'].max()
        test_min_date = df_test['install_date'].min()
        
        print(f"  Train ends: {train_max_date.date()}")
        print(f"  Val starts: {val_min_date.date()}")
        print(f"  Val ends: {val_max_date.date()}")
        print(f"  Test starts: {test_min_date.date()}")
        
        if train_max_date < val_min_date and val_max_date < test_min_date:
            print(f"  ✓ No temporal leakage: Train < Val < Test")
        else:
            print(f"  ⚠ Warning: Potential temporal overlap")
        
        print("\n✓ Split validation completed")
    
    def save_splits(self, df_train, df_val, df_test):
        """Save train/val/test sets"""
        print("\n" + "="*70)
        print("SAVING SPLITS")
        print("="*70)
        
        features_path = Path(self.config['data']['features_path'])
        
        # Save datasets
        train_path = features_path / 'train.csv'
        val_path = features_path / 'validation.csv'
        test_path = features_path / 'test.csv'
        
        print(f"\nSaving datasets...")
        df_train.to_csv(train_path, index=False)
        print(f"  ✓ Train: {train_path} ({len(df_train):,} rows)")
        
        df_val.to_csv(val_path, index=False)
        print(f"  ✓ Validation: {val_path} ({len(df_val):,} rows)")
        
        df_test.to_csv(test_path, index=False)
        print(f"  ✓ Test: {test_path} ({len(df_test):,} rows)")
        
        # Create summary
        total_rows = len(df_train) + len(df_val) + len(df_test)
        
        summary = pd.DataFrame({
            'dataset': ['train', 'validation', 'test'],
            'months': ['M1-M10', 'M11', 'M12'],
            'rows': [len(df_train), len(df_val), len(df_test)],
            'percentage': [
                len(df_train)/total_rows*100,
                len(df_val)/total_rows*100,
                len(df_test)/total_rows*100
            ],
            'tier1_pct': [
                (df_train['tier'] == 'tier1').sum() / len(df_train) * 100,
                (df_val['tier'] == 'tier1').sum() / len(df_val) * 100,
                (df_test['tier'] == 'tier1').sum() / len(df_test) * 100
            ],
            'tier2_pct': [
                (df_train['tier'] == 'tier2').sum() / len(df_train) * 100,
                (df_val['tier'] == 'tier2').sum() / len(df_val) * 100,
                (df_test['tier'] == 'tier2').sum() / len(df_test) * 100
            ],
            'tier3_pct': [
                (df_train['tier'] == 'tier3').sum() / len(df_train) * 100,
                (df_val['tier'] == 'tier3').sum() / len(df_val) * 100,
                (df_test['tier'] == 'tier3').sum() / len(df_test) * 100
            ]
        })
        
        summary_path = Path('results/step06_split_summary.csv')
        summary.to_csv(summary_path, index=False)
        print(f"\n✓ Summary: {summary_path}")
        
        # Display summary
        print(f"\nSplit Summary:")
        print(summary.to_string(index=False))


def main():
    """Main execution"""
    print("="*70)
    print("STEP 6: TRAIN/TEST SPLIT")
    print("="*70)
    
    # Initialize
    splitter = DataSplitter()
    
    # Load features
    df = splitter.load_features()
    
    # Add month column
    df = splitter.add_month_column(df)
    
    # Split by month
    df_train, df_val, df_test = splitter.split_by_month(df)
    
    # Validate
    splitter.validate_split(df_train, df_val, df_test)
    
    # Save
    splitter.save_splits(df_train, df_val, df_test)
    
    print("\n" + "="*70)
    print("✅ STEP 6 COMPLETED!")
    print("="*70)
    print("\nOutputs:")
    print("  📁 data/features/train.csv")
    print("  📁 data/features/validation.csv")
    print("  📁 data/features/test.csv")
    print("  📊 results/step06_split_summary.csv")
    print("\n➡️  Next Step: step07_train_hurdle_model.py")


if __name__ == "__main__":
    main()
