# Step 7: Two-Stage Hurdle Model
## Train XGBClassifier (Stage 1) + XGBRegressor (Stage 2)

**Thời gian:** 1 ngày  
**Độ khó:** ⭐⭐⭐⭐ Rất khó  
**Prerequisites:** Step 6 completed  

---

## 🎯 MỤC TIÊU

Train Two-Stage Hurdle Model (V2.1 Enhancement #1):

**Stage 1 (Classification):** Dự đoán "will pay?" (XGBClassifier)  
- Target: `is_payer` = (ltv_d60 > 0)
- Metric: AUC ≥ 0.75

**Stage 2 (Regression):** Dự đoán "how much?" (XGBRegressor)  
- Target: `ltv_d60` (chỉ payers)
- Metric: R² ≥ 0.6, MAPE ≤ 8%

**Final Prediction:**  
```
ltv_pred_d60 = P(payer_d60) × E[LTV_D60 | payer]
```

**⭐ YÊU CẦU:** Model phải output D60 prediction cho mọi campaign, không phụ thuộc vào actual data có đến D60 hay không.

---

## 📥 INPUT

- `data/features/train.csv`
- `data/features/validation.csv`
- `config/config.yaml`

---

## 📤 OUTPUT

- `models/tier1/hurdle_stage1_classifier.pkl`
- `models/tier1/hurdle_stage2_regressor.pkl`
- `models/tier2/hurdle_*.pkl` (same structure)
- `results/step07_hurdle_evaluation.html`

---

## 🔧 IMPLEMENTATION

### File: `scripts/step07_train_hurdle_model.py`

```python
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import joblib
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import roc_auc_score, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt

class HurdleModel:
    """Two-Stage Hurdle Model for Zero-Inflated Data"""
    
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.stage1_model = None  # Classifier
        self.stage2_model = None  # Regressor
    
    def prepare_features(self, df):
        """Prepare feature matrix"""
        feature_cols = self.config['features']['revenue_features'] + \
                      self.config['features']['engagement_features'] + \
                      self.config['features']['cpi_features'] + \
                      ['day_of_week', 'is_weekend', 'season']
        
        # Filter existing columns
        feature_cols = [c for c in feature_cols if c in df.columns]
        
        X = df[feature_cols].copy()
        
        # Handle categorical if needed
        X = X.select_dtypes(include=[np.number])
        
        return X
    
    def train_stage1_classifier(self, X_train, y_train, X_val, y_val):
        """Train Stage 1: Will Pay? (Binary Classification)"""
        print("\n[Stage 1] Training Payer Classifier...")
        
        # Model
        self.stage1_model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.config['modeling']['random_seed'],
            eval_metric='auc'
        )
        
        # Train
        self.stage1_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=50
        )
        
        # Predict probabilities
        y_train_proba = self.stage1_model.predict_proba(X_train)[:, 1]
        y_val_proba = self.stage1_model.predict_proba(X_val)[:, 1]
        
        # Metrics
        auc_train = roc_auc_score(y_train, y_train_proba)
        auc_val = roc_auc_score(y_val, y_val_proba)
        
        print(f"\n  ✓ Stage 1 AUC:")
        print(f"    - Train: {auc_train:.4f}")
        print(f"    - Val: {auc_val:.4f}")
        
        return y_train_proba, y_val_proba
    
    def train_stage2_regressor(self, X_train_payers, y_train_payers, 
                               X_val_payers, y_val_payers):
        """Train Stage 2: How Much? (Regression on Payers Only)"""
        print("\n[Stage 2] Training Amount Regressor (Payers Only)...")
        
        print(f"  - Train payers: {len(X_train_payers):,}")
        print(f"  - Val payers: {len(X_val_payers):,}")
        
        # Model
        self.stage2_model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.config['modeling']['random_seed']
        )
        
        # Train
        self.stage2_model.fit(
            X_train_payers, y_train_payers,
            eval_set=[(X_val_payers, y_val_payers)],
            early_stopping_rounds=20,
            verbose=50
        )
        
        # Predict
        y_train_pred = self.stage2_model.predict(X_train_payers)
        y_val_pred = self.stage2_model.predict(X_val_payers)
        
        # Metrics
        r2_train = r2_score(y_train_payers, y_train_pred)
        r2_val = r2_score(y_val_payers, y_val_pred)
        
        mape_train = mean_absolute_percentage_error(y_train_payers, y_train_pred)
        mape_val = mean_absolute_percentage_error(y_val_payers, y_val_pred)
        
        print(f"\n  ✓ Stage 2 Metrics (Payers Only):")
        print(f"    - Train R²: {r2_train:.4f}, MAPE: {mape_train:.4f}")
        print(f"    - Val R²: {r2_val:.4f}, MAPE: {mape_val:.4f}")
        
        return y_train_pred, y_val_pred
    
    def predict(self, X):
        """Two-stage prediction"""
        # Stage 1: Probability of being payer
        p_payer = self.stage1_model.predict_proba(X)[:, 1]
        
        # Stage 2: Expected LTV given payer
        ltv_given_payer = self.stage2_model.predict(X)
        ltv_given_payer = np.maximum(ltv_given_payer, 0)  # No negative LTV
        
        # Final prediction
        ltv_pred = p_payer * ltv_given_payer
        
        return ltv_pred, p_payer, ltv_given_payer
    
    def evaluate(self, X_val, y_val):
        """Evaluate full two-stage model"""
        print("\n[Evaluation] Two-Stage Hurdle Model...")
        
        # Predict
        ltv_pred, p_payer, ltv_given_payer = self.predict(X_val)
        
        # Overall MAPE
        mape = mean_absolute_percentage_error(y_val, ltv_pred)
        
        # R²
        r2 = r2_score(y_val, ltv_pred)
        
        print(f"\n  ✓ Overall Metrics:")
        print(f"    - MAPE: {mape:.4f}")
        print(f"    - R²: {r2:.4f}")
        
        # By payer status
        is_payer = y_val > 0
        
        if is_payer.sum() > 0:
            mape_payers = mean_absolute_percentage_error(
                y_val[is_payer], ltv_pred[is_payer]
            )
            print(f"    - MAPE (Payers): {mape_payers:.4f}")
        
        if (~is_payer).sum() > 0:
            mae_non_payers = np.mean(np.abs(y_val[~is_payer] - ltv_pred[~is_payer]))
            print(f"    - MAE (Non-Payers): ${mae_non_payers:.6f}")
        
        return {
            'mape': mape,
            'r2': r2,
            'predictions': ltv_pred,
            'p_payer': p_payer
        }
    
    def save_models(self, tier, models_path='models'):
        """Save both stage models"""
        tier_path = Path(models_path) / tier
        tier_path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.stage1_model, tier_path / 'hurdle_stage1_classifier.pkl')
        joblib.dump(self.stage2_model, tier_path / 'hurdle_stage2_regressor.pkl')
        
        print(f"\n✓ Saved models to: {tier_path}")

def train_hurdle_for_tier(tier, df_train, df_val):
    """Train hurdle model for specific tier"""
    print(f"\n{'='*60}")
    print(f"TRAINING HURDLE MODEL: {tier.upper()}")
    print(f"{'='*60}")
    
    # Filter data for tier
    df_train_tier = df_train[df_train['tier'] == tier].copy()
    df_val_tier = df_val[df_val['tier'] == tier].copy()
    
    print(f"Train: {len(df_train_tier):,} rows")
    print(f"Val: {len(df_val_tier):,} rows")
    
    # Initialize model
    hurdle = HurdleModel()
    
    # Prepare features
    X_train = hurdle.prepare_features(df_train_tier)
    X_val = hurdle.prepare_features(df_val_tier)
    
    # Stage 1 targets (binary)
    y_train_is_payer = (df_train_tier['ltv_d60'] > 0).astype(int)
    y_val_is_payer = (df_val_tier['ltv_d60'] > 0).astype(int)
    
    print(f"\nPayer distribution:")
    print(f"  - Train: {y_train_is_payer.mean()*100:.1f}% payers")
    print(f"  - Val: {y_val_is_payer.mean()*100:.1f}% payers")
    
    # Train Stage 1
    p_train_proba, p_val_proba = hurdle.train_stage1_classifier(
        X_train, y_train_is_payer, X_val, y_val_is_payer
    )
    
    # Stage 2: Only payers
    train_payer_mask = y_train_is_payer == 1
    val_payer_mask = y_val_is_payer == 1
    
    X_train_payers = X_train[train_payer_mask]
    y_train_payers = df_train_tier.loc[train_payer_mask, 'ltv_d60']
    
    X_val_payers = X_val[val_payer_mask]
    y_val_payers = df_val_tier.loc[val_payer_mask, 'ltv_d60']
    
    # Train Stage 2
    ltv_train_pred, ltv_val_pred = hurdle.train_stage2_regressor(
        X_train_payers, y_train_payers, X_val_payers, y_val_payers
    )
    
    # Evaluate full model
    results = hurdle.evaluate(X_val, df_val_tier['ltv_d60'])
    
    # Save models
    hurdle.save_models(tier)
    
    return hurdle, results

def main():
    """Main execution"""
    print("="*60)
    print("STEP 7: TWO-STAGE HURDLE MODEL")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    df_train = pd.read_csv('data/features/train.csv')
    df_val = pd.read_csv('data/features/validation.csv')
    
    # Train for Tier 1 and Tier 2
    results_summary = []
    
    for tier in ['tier1', 'tier2']:
        hurdle, results = train_hurdle_for_tier(tier, df_train, df_val)
        results_summary.append({
            'tier': tier,
            'mape': results['mape'],
            'r2': results['r2']
        })
    
    # Save summary
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv('results/step07_hurdle_summary.csv', index=False)
    
    print("\n" + "="*60)
    print("✅ STEP 7 COMPLETED!")
    print("="*60)
    print("\nNext Step: step08_train_curve_fitting.py")

if __name__ == "__main__":
    main()
```

---

## ✅ SUCCESS CRITERIA

- [x] Stage 1 AUC ≥ 0.75
- [x] Stage 2 R² ≥ 0.6, MAPE ≤ 8%
- [x] Overall MAPE ≤ 5% (Tier 1), ≤ 8% (Tier 2)
- [x] Models saved for tier1 and tier2

---

## 🎯 NEXT STEP

➡️ **[Step 8: Curve Fitting (Bayesian Priors)](step08_train_curve_fitting.md)**

---

**Estimated Time:** 6-8 hours  
**Difficulty:** ⭐⭐⭐⭐ Very Hard
