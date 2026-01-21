# Step 9: ML Multiplier Method
## Train LightGBM để Predict D30/D1 Multiplier

**Thời gian:** 0.5 ngày  
**Độ khó:** ⭐⭐⭐ Khó  
**Prerequisites:** Step 8 completed  

---

## 🎯 MỤC TIÊU

Train LightGBM model để predict:
```
multiplier = ltv_d30 / (rev_d0 + rev_d1)
ltv_pred = (rev_d0 + rev_d1) × multiplier_pred
```

**Ưu điểm:**  
- Tận dụng actual revenue D0-D1
- Fast inference
- Dùng cho Tier 1, 2, 3

---

## 📥 INPUT

- `data/features/train.csv`
- `data/features/validation.csv`

---

## 📤 OUTPUT

- `models/tier1/ml_multiplier.pkl`
- `models/tier2/ml_multiplier.pkl`
- `models/tier3/ml_multiplier.pkl`
- `results/step09_multiplier_evaluation.html`

---

## 🔧 IMPLEMENTATION

### File: `scripts/step09_train_ml_multiplier.py`

```python
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import yaml
import joblib
from sklearn.metrics import mean_absolute_percentage_error, r2_score

class MLMultiplierModel:
    """ML-based multiplier prediction"""
    
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
    
    def prepare_features(self, df):
        """Prepare features"""
        feature_cols = self.config['features']['revenue_features'] + \
                      self.config['features']['engagement_features'] + \
                      self.config['features']['cpi_features'] + \
                      ['day_of_week', 'is_weekend', 'season',
                       'campaign_ltv_avg', 'campaign_rev_avg']
        
        feature_cols = [c for c in feature_cols if c in df.columns]
        
        X = df[feature_cols].copy()
        X = X.select_dtypes(include=[np.number])
        
        return X
    
    def calculate_target_multiplier(self, df):
        """Calculate target multiplier"""
        df = df.copy()
        
        # Target multiplier = ltv_d30 / (rev_d0 + rev_d1)
        df['target_multiplier'] = df['ltv_d30'] / (df['rev_sum'] + 1e-6)
        
        # Clip extreme values
        df['target_multiplier'] = df['target_multiplier'].clip(0, 50)
        
        return df['target_multiplier']
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train LightGBM"""
        print("\n[Training] LightGBM Multiplier Model...")
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Parameters
        params = {
            'objective': 'regression',
            'metric': 'mape',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbosity': -1
        }
        
        # Train
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(50)]
        )
        
        print(f"\n  ✓ Best iteration: {self.model.best_iteration}")
    
    def predict_ltv(self, X, rev_sum):
        """Predict LTV using multiplier"""
        # Predict multiplier
        multiplier_pred = self.model.predict(X, num_iteration=self.model.best_iteration)
        multiplier_pred = np.maximum(multiplier_pred, 0)  # No negative
        
        # LTV = rev_sum × multiplier
        ltv_pred = rev_sum * multiplier_pred
        
        return ltv_pred
    
    def evaluate(self, X_val, y_val, rev_sum_val):
        """Evaluate model"""
        print("\n[Evaluation] ML Multiplier Model...")
        
        # Predict
        ltv_pred = self.predict_ltv(X_val, rev_sum_val)
        
        # Metrics
        mape = mean_absolute_percentage_error(y_val, ltv_pred)
        r2 = r2_score(y_val, ltv_pred)
        
        print(f"  ✓ MAPE: {mape:.4f}")
        print(f"  ✓ R²: {r2:.4f}")
        
        return {'mape': mape, 'r2': r2}
    
    def save_model(self, tier, models_path='models'):
        """Save model"""
        tier_path = Path(models_path) / tier
        tier_path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.model, tier_path / 'ml_multiplier.pkl')
        print(f"\n✓ Saved model to: {tier_path}")

def train_multiplier_for_tier(tier, df_train, df_val):
    """Train multiplier model for tier"""
    print(f"\n{'='*60}")
    print(f"TRAINING ML MULTIPLIER: {tier.upper()}")
    print(f"{'='*60}")
    
    # Filter
    df_train_tier = df_train[df_train['tier'] == tier].copy()
    df_val_tier = df_val[df_val['tier'] == tier].copy()
    
    print(f"Train: {len(df_train_tier):,} rows")
    print(f"Val: {len(df_val_tier):,} rows")
    
    # Initialize
    ml_mult = MLMultiplierModel()
    
    # Prepare features
    X_train = ml_mult.prepare_features(df_train_tier)
    X_val = ml_mult.prepare_features(df_val_tier)
    
    # Target multiplier
    y_train = ml_mult.calculate_target_multiplier(df_train_tier)
    
    # Train
    ml_mult.train(X_train, y_train, X_val, ml_mult.calculate_target_multiplier(df_val_tier))
    
    # Evaluate (using actual LTV)
    results = ml_mult.evaluate(X_val, df_val_tier['ltv_d30'], df_val_tier['rev_sum'])
    
    # Save
    ml_mult.save_model(tier)
    
    return results

def main():
    """Main execution"""
    print("="*60)
    print("STEP 9: ML MULTIPLIER METHOD")
    print("="*60)
    
    # Load data
    df_train = pd.read_csv('data/features/train.csv')
    df_val = pd.read_csv('data/features/validation.csv')
    
    # Train for all tiers
    results_summary = []
    
    for tier in ['tier1', 'tier2', 'tier3']:
        results = train_multiplier_for_tier(tier, df_train, df_val)
        results_summary.append({
            'tier': tier,
            'mape': results['mape'],
            'r2': results['r2']
        })
    
    # Save summary
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv('results/step09_multiplier_summary.csv', index=False)
    
    print("\n" + "="*60)
    print("✅ STEP 9 COMPLETED!")
    print("="*60)
    print("\nNext Step: step10_train_lookalike.py")

if __name__ == "__main__":
    main()
```

---

## ✅ SUCCESS CRITERIA

- [x] Models trained for tier1, tier2, tier3
- [x] MAPE ≤ 5% (Tier 1), ≤ 7% (Tier 2), ≤ 10% (Tier 3)
- [x] Models saved

---

## 🎯 NEXT STEP

➡️ **[Step 10: Look-alike Method](step10_train_lookalike.md)**

---

**Estimated Time:** 3-4 hours  
**Difficulty:** ⭐⭐⭐ Hard
