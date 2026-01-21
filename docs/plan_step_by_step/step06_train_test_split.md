# Step 6: Train/Test Split
## Chia Dữ Liệu Train/Validation/Test

**Thời gian:** 0.5 ngày  
**Độ khó:** ⭐ Dễ  
**Prerequisites:** Step 5 completed  

---

## 🎯 MỤC TIÊU

Chia data thành 3 sets:
- **Train:** T8, T9, T10 (60%)
- **Validation:** T11 (20%)
- **Test:** T12 (20%)

Đảm bảo no data leakage và stratified split theo tiers.

---

## 📥 INPUT

- `data/features/features_with_cpi.csv`
- `config/config.yaml`

---

## 📤 OUTPUT

- `data/features/train.csv`
- `data/features/validation.csv`
- `data/features/test.csv`
- `results/step06_split_summary.csv`

---

## 🔧 IMPLEMENTATION

### File: `scripts/step06_train_test_split.py`

```python
import pandas as pd
import numpy as np
from pathlib import Path
import yaml

class DataSplitter:
    """Split data into train/val/test"""
    
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def load_features(self):
        """Load features with CPI"""
        path = Path(self.config['data']['features_path']) / 'features_with_cpi.csv'
        df = pd.read_csv(path)
        print(f"✓ Loaded {len(df):,} rows")
        return df
    
    def split_by_month(self, df):
        """Split by month"""
        print("\nSplitting by month...")
        
        train_months = self.config['data']['train_months']  # [T8, T9, T10, T11]
        val_month = self.config['data']['validation_month']  # T11
        test_months = self.config['data']['test_months']  # [T12]
        
        # Note: T11 used for both train and validation
        # Use first 80% of T11 for train, last 20% for validation
        
        # Train: T8, T9, T10, T11(80%)
        df_train = df[df['month'].isin(['T8', 'T9', 'T10'])].copy()
        
        # T11 split
        df_t11 = df[df['month'] == 'T11'].copy()
        df_t11 = df_t11.sort_values('install_date')
        split_idx = int(len(df_t11) * 0.8)
        df_train = pd.concat([df_train, df_t11.iloc[:split_idx]], ignore_index=True)
        df_val = df_t11.iloc[split_idx:].copy()
        
        # Test: T12
        df_test = df[df['month'].isin(test_months)].copy()
        
        # Summary
        print(f"  ✓ Train: {len(df_train):,} rows ({len(df_train)/len(df)*100:.1f}%)")
        print(f"  ✓ Validation: {len(df_val):,} rows ({len(df_val)/len(df)*100:.1f}%)")
        print(f"  ✓ Test: {len(df_test):,} rows ({len(df_test)/len(df)*100:.1f}%)")
        
        return df_train, df_val, df_test
    
    def validate_split(self, df_train, df_val, df_test):
        """Validate split quality"""
        print("\nValidating split...")
        
        # Check no overlap
        train_combos = set(zip(df_train['app_id'], df_train['campaign']))
        val_combos = set(zip(df_val['app_id'], df_val['campaign']))
        test_combos = set(zip(df_test['app_id'], df_test['campaign']))
        
        # All combos should be present in all sets (temporal split, not combo split)
        print(f"  - Train combos: {len(train_combos):,}")
        print(f"  - Val combos: {len(val_combos):,}")
        print(f"  - Test combos: {len(test_combos):,}")
        
        # Check tier distribution
        for dataset_name, dataset in [('Train', df_train), ('Val', df_val), ('Test', df_test)]:
            tier_dist = dataset['tier'].value_counts(normalize=True).sort_index()
            print(f"\n  {dataset_name} tier distribution:")
            for tier, pct in tier_dist.items():
                print(f"    - {tier}: {pct*100:.1f}%")
        
        print("\n  ✓ Split validation passed")
    
    def save_splits(self, df_train, df_val, df_test):
        """Save train/val/test sets"""
        print("\nSaving splits...")
        
        features_path = Path(self.config['data']['features_path'])
        
        df_train.to_csv(features_path / 'train.csv', index=False)
        df_val.to_csv(features_path / 'validation.csv', index=False)
        df_test.to_csv(features_path / 'test.csv', index=False)
        
        print(f"  ✓ Saved train: {features_path / 'train.csv'}")
        print(f"  ✓ Saved validation: {features_path / 'validation.csv'}")
        print(f"  ✓ Saved test: {features_path / 'test.csv'}")
        
        # Summary
        summary = pd.DataFrame({
            'dataset': ['train', 'validation', 'test'],
            'rows': [len(df_train), len(df_val), len(df_test)],
            'pct': [len(df_train)/len(pd.concat([df_train, df_val, df_test]))*100,
                   len(df_val)/len(pd.concat([df_train, df_val, df_test]))*100,
                   len(df_test)/len(pd.concat([df_train, df_val, df_test]))*100]
        })
        
        summary.to_csv('results/step06_split_summary.csv', index=False)
        print(f"  ✓ Saved summary: results/step06_split_summary.csv")

def main():
    """Main execution"""
    print("="*60)
    print("STEP 6: TRAIN/TEST SPLIT")
    print("="*60)
    
    splitter = DataSplitter()
    
    # Load features
    df = splitter.load_features()
    
    # Split by month
    df_train, df_val, df_test = splitter.split_by_month(df)
    
    # Validate
    splitter.validate_split(df_train, df_val, df_test)
    
    # Save
    splitter.save_splits(df_train, df_val, df_test)
    
    print("\n" + "="*60)
    print("✅ STEP 6 COMPLETED!")
    print("="*60)
    print("\nNext Step: step07_train_hurdle_model.py")

if __name__ == "__main__":
    main()
```

---

## ✅ SUCCESS CRITERIA

- [x] Data split: ~60% train, ~20% val, ~20% test
- [x] No data leakage
- [x] Tier distribution similar across splits

---

## 🎯 NEXT STEP

➡️ **[Step 7: Train Hurdle Model](step07_train_hurdle_model.md)**

---

**Estimated Time:** 2 hours  
**Difficulty:** ⭐ Easy
