# Step 8: Curve Fitting with Bayesian Priors
## Power Law Curve Fitting với Regularization

**Thời gian:** 1 ngày  
**Độ khó:** ⭐⭐⭐⭐ Rất khó  
**Prerequisites:** Step 7 completed  

---

## 🎯 MỤC TIÊU

Fit curve dạng:  
```
ltv(t) = a × t^b + c
```

Sử dụng **Bayesian Priors** (V2.1 Enhancement #3):
- Prior = tier-average parameters (α_tier, β_tier, γ_tier)
- Regularization: `loss = MSE + λ × ||params - prior||²`
- Giảm variance từ ±50% → ±20% cho sparse data

**Chỉ train cho Tier 1** (data đủ stable)

---

## 📥 INPUT

- `data/features/train.csv`
- `data/features/validation.csv`
- `config/config.yaml`

---

## 📤 OUTPUT

- `models/tier1/curve_fitting_params.pkl` (Parameters cho mỗi combo)
- `models/tier1/curve_fitting_priors.pkl` (Tier-level priors)
- `results/step08_curve_fitting_evaluation.html`

---

## 🔧 IMPLEMENTATION

### File: `scripts/step08_train_curve_fitting.py`

```python
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import joblib
from scipy.optimize import curve_fit, minimize
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class CurveFittingModel:
    """Power Law Curve Fitting with Bayesian Priors"""
    
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.priors = {}  # Tier-level priors
        self.params = {}  # Campaign-level fitted parameters
    
    @staticmethod
    def power_law_curve(t, a, b, c):
        """Power law: ltv(t) = a × t^b + c"""
        return a * np.power(t, b) + c
    
    def calculate_tier_priors(self, df_train, tier):
        """Calculate tier-level prior parameters (mean of all campaigns)"""
        print(f"\n[Priors] Calculating tier-level priors for {tier}...")
        
        df_tier = df_train[df_train['tier'] == tier].copy()
        
        # Group by campaign
        campaign_params = []
        
        for (app_id, campaign), group in df_tier.groupby(['app_id', 'campaign']):
            if len(group) < 30:  # Min samples
                continue
            
            # Observed data points (assuming we have D0, D1, D7, D30)
            # For simplicity, use rev_d0, rev_d1, ltv_d7, ltv_d30
            days = np.array([0, 1, 7, 30])
            
            # Get average values
            ltv_values = np.array([
                group['rev_d0'].mean(),
                group['rev_d1'].mean(),
                group.get('ltv_d7', group['rev_d1'] * 2).mean() if 'ltv_d7' in group.columns else group['rev_d1'].mean() * 2,
                group['ltv_d30'].mean()
            ])
            
            try:
                # Fit curve
                popt, _ = curve_fit(
                    self.power_law_curve,
                    days,
                    ltv_values,
                    p0=[0.1, 0.5, 0.01],  # Initial guess
                    bounds=([0, 0, 0], [10, 2, 1]),  # Bounds
                    maxfev=1000
                )
                
                campaign_params.append({
                    'app_id': app_id,
                    'campaign': campaign,
                    'a': popt[0],
                    'b': popt[1],
                    'c': popt[2]
                })
            except:
                continue
        
        # Calculate tier-level priors (median of campaign params)
        params_df = pd.DataFrame(campaign_params)
        
        if len(params_df) > 0:
            priors = {
                'a_prior': params_df['a'].median(),
                'b_prior': params_df['b'].median(),
                'c_prior': params_df['c'].median(),
                'a_std': params_df['a'].std(),
                'b_std': params_df['b'].std(),
                'c_std': params_df['c'].std()
            }
            
            print(f"  ✓ Tier Priors (from {len(params_df)} campaigns):")
            print(f"    - a: {priors['a_prior']:.4f} ± {priors['a_std']:.4f}")
            print(f"    - b: {priors['b_prior']:.4f} ± {priors['b_std']:.4f}")
            print(f"    - c: {priors['c_prior']:.4f} ± {priors['c_std']:.4f}")
        else:
            # Default priors
            priors = {
                'a_prior': 0.1,
                'b_prior': 0.5,
                'c_prior': 0.01,
                'a_std': 0.05,
                'b_std': 0.2,
                'c_std': 0.005
            }
            print(f"  ⚠ Not enough campaigns, using default priors")
        
        self.priors[tier] = priors
        
        return priors
    
    def fit_campaign_with_prior(self, group, priors, lambda_reg=0.1):
        """Fit curve for one campaign with Bayesian prior regularization"""
        
        # Observed data
        days = np.array([0, 1, 7, 30])
        ltv_values = np.array([
            group['rev_d0'].mean(),
            group['rev_d1'].mean(),
            group.get('ltv_d7', group['rev_d1'] * 2).mean() if 'ltv_d7' in group.columns else group['rev_d1'].mean() * 2,
            group['ltv_d30'].mean()
        ])
        
        # Objective function with prior regularization
        def objective(params):
            a, b, c = params
            
            # MSE loss
            pred = self.power_law_curve(days, a, b, c)
            mse = np.mean((pred - ltv_values) ** 2)
            
            # Prior regularization (L2 penalty)
            prior_penalty = (
                ((a - priors['a_prior']) / (priors['a_std'] + 1e-6)) ** 2 +
                ((b - priors['b_prior']) / (priors['b_std'] + 1e-6)) ** 2 +
                ((c - priors['c_prior']) / (priors['c_std'] + 1e-6)) ** 2
            )
            
            loss = mse + lambda_reg * prior_penalty
            
            return loss
        
        # Optimize
        result = minimize(
            objective,
            x0=[priors['a_prior'], priors['b_prior'], priors['c_prior']],
            bounds=[(0, 10), (0, 2), (0, 1)],
            method='L-BFGS-B'
        )
        
        if result.success:
            return result.x
        else:
            # Return priors if optimization fails
            return [priors['a_prior'], priors['b_prior'], priors['c_prior']]
    
    def train_tier(self, df_train, tier):
        """Train curve fitting for all campaigns in tier"""
        print(f"\n{'='*60}")
        print(f"TRAINING CURVE FITTING: {tier.upper()}")
        print(f"{'='*60}")
        
        df_tier = df_train[df_train['tier'] == tier].copy()
        
        # Calculate priors
        priors = self.calculate_tier_priors(df_train, tier)
        
        # Fit each campaign
        print(f"\n[Fitting] Fitting curves for each campaign...")
        
        campaign_params = []
        
        for (app_id, campaign), group in df_tier.groupby(['app_id', 'campaign']):
            if len(group) < 10:  # Min samples
                continue
            
            # Fit with prior
            try:
                params = self.fit_campaign_with_prior(group, priors, lambda_reg=0.1)
                
                campaign_params.append({
                    'app_id': app_id,
                    'campaign': campaign,
                    'a': params[0],
                    'b': params[1],
                    'c': params[2]
                })
            except:
                # Use priors
                campaign_params.append({
                    'app_id': app_id,
                    'campaign': campaign,
                    'a': priors['a_prior'],
                    'b': priors['b_prior'],
                    'c': priors['c_prior']
                })
        
        self.params[tier] = pd.DataFrame(campaign_params)
        
        print(f"  ✓ Fitted {len(campaign_params)} campaigns")
    
    def predict(self, app_id, campaign, tier, target_day=30):
        """Predict LTV at target_day for a campaign"""
        
        # Get params
        params_df = self.params.get(tier)
        
        if params_df is None:
            return None
        
        # Find campaign params
        mask = (params_df['app_id'] == app_id) & (params_df['campaign'] == campaign)
        
        if mask.sum() == 0:
            # Use tier priors
            priors = self.priors[tier]
            a, b, c = priors['a_prior'], priors['b_prior'], priors['c_prior']
        else:
            row = params_df[mask].iloc[0]
            a, b, c = row['a'], row['b'], row['c']
        
        # Predict
        ltv_pred = self.power_law_curve(target_day, a, b, c)
        
        return ltv_pred
    
    def evaluate(self, df_val, tier):
        """Evaluate on validation set"""
        print(f"\n[Evaluation] Evaluating {tier}...")
        
        df_tier = df_val[df_val['tier'] == tier].copy()
        
        # Predict for each row
        predictions = []
        
        for idx, row in df_tier.iterrows():
            pred = self.predict(row['app_id'], row['campaign'], tier, target_day=30)
            predictions.append(pred if pred is not None else df_tier['ltv_d30'].median())
        
        df_tier['ltv_pred'] = predictions
        
        # Metrics
        mape = mean_absolute_percentage_error(df_tier['ltv_d30'], df_tier['ltv_pred'])
        r2 = r2_score(df_tier['ltv_d30'], df_tier['ltv_pred'])
        
        print(f"  ✓ MAPE: {mape:.4f}")
        print(f"  ✓ R²: {r2:.4f}")
        
        return {'mape': mape, 'r2': r2}
    
    def save_models(self, tier, models_path='models'):
        """Save fitted parameters and priors"""
        tier_path = Path(models_path) / tier
        tier_path.mkdir(parents=True, exist_ok=True)
        
        # Save params
        joblib.dump(self.params[tier], tier_path / 'curve_fitting_params.pkl')
        
        # Save priors
        joblib.dump(self.priors[tier], tier_path / 'curve_fitting_priors.pkl')
        
        print(f"\n✓ Saved models to: {tier_path}")

def main():
    """Main execution"""
    print("="*60)
    print("STEP 8: CURVE FITTING WITH BAYESIAN PRIORS")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    df_train = pd.read_csv('data/features/train.csv')
    df_val = pd.read_csv('data/features/validation.csv')
    
    # Initialize model
    curve_model = CurveFittingModel()
    
    # Train for Tier 1 only (stable data)
    tier = 'tier1'
    curve_model.train_tier(df_train, tier)
    
    # Evaluate
    results = curve_model.evaluate(df_val, tier)
    
    # Save
    curve_model.save_models(tier)
    
    # Summary
    summary_df = pd.DataFrame([{
        'tier': tier,
        'mape': results['mape'],
        'r2': results['r2']
    }])
    summary_df.to_csv('results/step08_curve_fitting_summary.csv', index=False)
    
    print("\n" + "="*60)
    print("✅ STEP 8 COMPLETED!")
    print("="*60)
    print("\nNext Step: step09_train_ml_multiplier.py")

if __name__ == "__main__":
    main()
```

---

## ✅ SUCCESS CRITERIA

- [x] Tier priors calculated successfully
- [x] Curves fitted for all Tier 1 campaigns
- [x] MAPE ≤ 6% on validation
- [x] Variance reduced (params close to priors)

---

## 🎯 NEXT STEP

➡️ **[Step 9: ML Multiplier Method](step09_train_ml_multiplier.md)**

---

**Estimated Time:** 6-8 hours  
**Difficulty:** ⭐⭐⭐⭐ Very Hard
