"""
Step 08: Curve Fitting with Bayesian Priors
===========================================
Power Law Curve Fitting: ltv(t) = a × t^b + c

Key innovations:
- Bayesian priors from tier-level averages
- Regularization to reduce variance
- Focus on Tier1 (stable data)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from scipy.optimize import curve_fit, minimize
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class CurveFittingModel:
    """Power Law Curve Fitting with Bayesian Priors"""
    
    def __init__(self):
        self.priors = {}  # Tier-level priors
        self.params = {}  # Campaign-level fitted parameters
    
    @staticmethod
    def power_law_curve(t, a, b, c):
        """
        Power law curve: ltv(t) = a × t^b + c
        
        Parameters:
        - a: scale parameter (initial growth rate)
        - b: shape parameter (growth exponent, 0<b<1 for diminishing returns)
        - c: baseline/offset (initial LTV at t=0)
        """
        return a * np.power(t + 1, b) + c  # t+1 to handle t=0
    
    def calculate_tier_priors(self, df_train, tier):
        """
        Calculate tier-level prior parameters from historical campaigns
        These serve as Bayesian priors for new campaigns
        """
        print(f"\n📊 Calculating tier-level priors for {tier.upper()}...")
        
        df_tier = df_train[df_train['tier'] == tier].copy()
        
        if len(df_tier) == 0:
            print(f"  ⚠️  No data for {tier}")
            return None
        
        # Extract temporal LTV features
        # We have: ltv_d30, ltv_d60, rev_sum (cumulative), rev_last (most recent)
        # We'll estimate: D0, D7, D30, D60
        campaign_params = []
        
        for (campaign, geo), group in df_tier.groupby(['campaign', 'geo']):
            if len(group) < 20:  # Minimum samples for reliable fit
                continue
            
            # Time points and LTV values
            days = np.array([0, 7, 30, 60])
            
            # Estimate LTV at each time point
            # D0: approximately 10% of total revenue (early monetization)
            # D7: approximately 40% of D30
            # D30: actual ltv_d30
            # D60: actual ltv_d60
            ltv_d0 = group['rev_sum'].mean() * 0.05  # 5% of total at D0
            ltv_d7 = group['ltv_d30'].mean() * 0.4   # 40% of D30
            ltv_d30 = group['ltv_d30'].mean()
            ltv_d60 = group['ltv_d60'].mean()
            
            ltv_values = np.array([ltv_d0, ltv_d7, ltv_d30, ltv_d60])
            
            # Skip if data is invalid
            if not np.all(ltv_values >= 0) or ltv_values[-1] == 0:
                continue
            
            # Skip if not monotonically increasing (data quality issue)
            if not np.all(np.diff(ltv_values) >= -0.01):  # Allow tiny decreases
                continue
            
            try:
                # Fit power law curve
                popt, pcov = curve_fit(
                    self.power_law_curve,
                    days,
                    ltv_values,
                    p0=[0.5, 0.3, 0.1],  # Initial guess
                    bounds=([0, 0, 0], [100, 1, 50]),  # Bounds: a>0, 0<b<1, c>0
                    maxfev=5000
                )
                
                # Check if fit is reasonable
                predictions = self.power_law_curve(days, *popt)
                r2 = r2_score(ltv_values, predictions)
                
                if r2 > 0.7:  # Only keep good fits
                    campaign_params.append({
                        'campaign': campaign,
                        'geo': geo,
                        'a': popt[0],
                        'b': popt[1],
                        'c': popt[2],
                        'r2': r2,
                        'ltv_d60': ltv_d60
                    })
            except Exception as e:
                continue
        
        # Calculate tier-level priors (robust statistics)
        params_df = pd.DataFrame(campaign_params)
        
        if len(params_df) >= 3:
            # Use median and MAD for robustness
            priors = {
                'a_prior': params_df['a'].median(),
                'b_prior': params_df['b'].median(),
                'c_prior': params_df['c'].median(),
                'a_std': params_df['a'].std() if params_df['a'].std() > 0 else 0.1,
                'b_std': params_df['b'].std() if params_df['b'].std() > 0 else 0.1,
                'c_std': params_df['c'].std() if params_df['c'].std() > 0 else 0.1,
                'n_campaigns': len(params_df),
                'avg_r2': params_df['r2'].mean()
            }
            
            print(f"  ✅ Priors from {len(params_df)} campaigns:")
            print(f"     a (scale):    {priors['a_prior']:.4f} ± {priors['a_std']:.4f}")
            print(f"     b (shape):    {priors['b_prior']:.4f} ± {priors['b_std']:.4f}")
            print(f"     c (baseline): {priors['c_prior']:.4f} ± {priors['c_std']:.4f}")
            print(f"     Avg R²:       {priors['avg_r2']:.4f}")
        else:
            # Default priors for power law curve
            priors = {
                'a_prior': 1.0,
                'b_prior': 0.3,
                'c_prior': 0.5,
                'a_std': 0.5,
                'b_std': 0.15,
                'c_std': 0.25,
                'n_campaigns': 0,
                'avg_r2': 0.0
            }
            print(f"  ⚠️  Insufficient campaigns ({len(params_df)}), using defaults")
        
        self.priors[tier] = priors
        
        return priors, params_df
    
    def fit_campaign_with_prior(self, days, ltv_values, priors, lambda_reg=0.5):
        """
        Fit curve for one campaign with Bayesian prior regularization
        
        Loss = MSE + λ × ||params - prior||²
        """
        
        # Objective function with prior regularization
        def objective(params):
            a, b, c = params
            
            # Predictions
            pred = self.power_law_curve(days, a, b, c)
            
            # MSE loss
            mse = np.mean((pred - ltv_values) ** 2)
            
            # Prior regularization (normalized L2 penalty)
            prior_penalty = (
                ((a - priors['a_prior']) / (priors['a_std'] + 1e-6)) ** 2 +
                ((b - priors['b_prior']) / (priors['b_std'] + 1e-6)) ** 2 +
                ((c - priors['c_prior']) / (priors['c_std'] + 1e-6)) ** 2
            )
            
            # Total loss
            loss = mse + lambda_reg * prior_penalty
            
            return loss
        
        # Optimize with priors as initial guess
        result = minimize(
            objective,
            x0=[priors['a_prior'], priors['b_prior'], priors['c_prior']],
            bounds=[(0, 100), (0, 1), (0, 50)],
            method='L-BFGS-B',
            options={'maxiter': 1000}
        )
        
        if result.success:
            return result.x, result.fun
        else:
            # Fall back to priors
            return [priors['a_prior'], priors['b_prior'], priors['c_prior']], float('inf')
    
    def train_tier(self, df_train, tier):
        """Train curve fitting for all campaigns in tier"""
        print(f"\n{'='*80}")
        print(f"🎯 CURVE FITTING: {tier.upper()}")
        print(f"{'='*80}")
        
        df_tier = df_train[df_train['tier'] == tier].copy()
        
        if len(df_tier) == 0:
            print(f"  ⚠️  No data for {tier}")
            return
        
        # Step 1: Calculate priors
        priors_result = self.calculate_tier_priors(df_train, tier)
        if priors_result is None:
            return
        
        priors, prior_params_df = priors_result
        
        # Step 2: Fit each campaign with priors
        print(f"\n🔧 Fitting individual campaigns...")
        
        campaign_fits = []
        
        for (campaign, geo), group in df_tier.groupby(['campaign', 'geo']):
            if len(group) < 10:  # Minimum samples
                continue
            
            # Time points and LTV values
            days = np.array([0, 7, 30, 60])
            
            ltv_d0 = group['rev_sum'].mean() * 0.05
            ltv_d7 = group['ltv_d30'].mean() * 0.4
            ltv_d30 = group['ltv_d30'].mean()
            ltv_d60 = group['ltv_d60'].mean()
            
            ltv_values = np.array([ltv_d0, ltv_d7, ltv_d30, ltv_d60])
            
            if not np.all(ltv_values >= 0):
                continue
            
            try:
                # Fit with Bayesian priors
                params, loss = self.fit_campaign_with_prior(
                    days, ltv_values, priors, lambda_reg=0.5
                )
                
                # Calculate fit quality
                predictions = self.power_law_curve(days, *params)
                r2 = r2_score(ltv_values, predictions)
                mape = mean_absolute_percentage_error(
                    ltv_values[ltv_values > 0.01], 
                    predictions[ltv_values > 0.01]
                ) if np.sum(ltv_values > 0.01) > 0 else 999
                
                campaign_fits.append({
                    'campaign': campaign,
                    'geo': geo,
                    'n_samples': len(group),
                    'a': params[0],
                    'b': params[1],
                    'c': params[2],
                    'r2': r2,
                    'mape': mape,
                    'loss': loss,
                    'ltv_d60_actual': ltv_d60,
                    'ltv_d60_pred': self.power_law_curve(60, *params)
                })
            except Exception as e:
                # Fall back to priors
                campaign_fits.append({
                    'campaign': campaign,
                    'geo': geo,
                    'n_samples': len(group),
                    'a': priors['a_prior'],
                    'b': priors['b_prior'],
                    'c': priors['c_prior'],
                    'r2': 0.0,
                    'mape': 999,
                    'loss': float('inf'),
                    'ltv_d60_actual': ltv_d60,
                    'ltv_d60_pred': self.power_law_curve(60, priors['a_prior'], priors['b_prior'], priors['c_prior'])
                })
        
        self.params[tier] = pd.DataFrame(campaign_fits)
        
        print(f"  ✅ Fitted {len(campaign_fits)} campaigns")
        
        # Summary statistics
        params_df = self.params[tier]
        good_fits = params_df[params_df['r2'] > 0.7]
        
        print(f"\n📈 Fit Quality Summary:")
        print(f"   Good fits (R²>0.7): {len(good_fits)}/{len(params_df)} ({len(good_fits)/len(params_df)*100:.1f}%)")
        print(f"   Avg R²:             {params_df['r2'].mean():.4f}")
        print(f"   Median MAPE:        {params_df[params_df['mape'] < 100]['mape'].median():.2f}%")
    
    def evaluate(self, df_val, tier):
        """Evaluate curve fitting on validation set"""
        print(f"\n{'='*80}")
        print(f"📊 EVALUATION: {tier.upper()}")
        print(f"{'='*80}")
        
        df_tier = df_val[df_val['tier'] == tier].copy()
        params_df = self.params.get(tier)
        
        if params_df is None or len(df_tier) == 0:
            print(f"  ⚠️  No data to evaluate")
            return None
        
        # Predict for each campaign-geo combo
        predictions = []
        actuals = []
        
        for (campaign, geo), group in df_tier.groupby(['campaign', 'geo']):
            # Get fitted parameters
            fit = params_df[
                (params_df['campaign'] == campaign) & 
                (params_df['geo'] == geo)
            ]
            
            if len(fit) == 0:
                # Use priors
                if tier in self.priors:
                    a, b, c = self.priors[tier]['a_prior'], self.priors[tier]['b_prior'], self.priors[tier]['c_prior']
                else:
                    continue
            else:
                a, b, c = fit.iloc[0]['a'], fit.iloc[0]['b'], fit.iloc[0]['c']
            
            # Predict D60
            pred_d60 = self.power_law_curve(60, a, b, c)
            actual_d60 = group['ltv_d60'].mean()
            
            predictions.append(pred_d60)
            actuals.append(actual_d60)
        
        if len(predictions) == 0:
            print(f"  ⚠️  No predictions made")
            return None
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        mask = actuals > 0.01
        mape = mean_absolute_percentage_error(actuals[mask], predictions[mask]) if mask.sum() > 0 else 999
        r2 = r2_score(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        
        print(f"\n✅ Validation Results:")
        print(f"   Campaigns evaluated: {len(predictions)}")
        print(f"   MAPE:                {mape*100:.2f}%")
        print(f"   R²:                  {r2:.4f}")
        print(f"   MAE:                 ${mae:.2f}")
        
        return {
            'tier': tier,
            'n_campaigns': len(predictions),
            'mape': mape,
            'r2': r2,
            'mae': mae
        }
    
    def save_models(self, tier):
        """Save fitted parameters and priors"""
        output_dir = Path(f'models/{tier}')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save priors
        if tier in self.priors:
            priors_path = output_dir / 'curve_fitting_priors.pkl'
            with open(priors_path, 'wb') as f:
                pickle.dump(self.priors[tier], f)
            print(f"\n💾 Saved priors: {priors_path}")
        
        # Save campaign parameters
        if tier in self.params:
            params_path = output_dir / 'curve_fitting_params.csv'
            self.params[tier].to_csv(params_path, index=False)
            print(f"💾 Saved params: {params_path}")
    
    def predict_campaign(self, campaign, geo, tier, target_day=90):
        """
        Predict LTV at target_day for a specific campaign
        Useful for extrapolating beyond D60
        """
        params_df = self.params.get(tier)
        
        if params_df is None:
            return None
        
        # Find campaign params
        fit = params_df[
            (params_df['campaign'] == campaign) & 
            (params_df['geo'] == geo)
        ]
        
        if len(fit) == 0:
            # Use priors
            if tier in self.priors:
                a, b, c = self.priors[tier]['a_prior'], self.priors[tier]['b_prior'], self.priors[tier]['c_prior']
            else:
                return None
        else:
            a, b, c = fit.iloc[0]['a'], fit.iloc[0]['b'], fit.iloc[0]['c']
        
        # Predict at target day
        prediction = self.power_law_curve(target_day, a, b, c)
        
        return prediction


def main():
    """Main execution"""
    print("="*80)
    print("STEP 8: CURVE FITTING WITH BAYESIAN PRIORS")
    print("="*80)
    print()
    print("Power Law Curve: ltv(t) = a × (t+1)^b + c")
    print("With Bayesian Priors for regularization")
    print()
    
    # Load data
    print("📂 Loading data...")
    train_df = pd.read_csv('data/features/train.csv')
    val_df = pd.read_csv('data/features/validation.csv')
    
    print(f"   Train: {len(train_df):,} rows")
    print(f"   Val:   {len(val_df):,} rows")
    
    # Initialize model
    model = CurveFittingModel()
    
    # Train on Tier1 (most stable data)
    model.train_tier(train_df, 'tier1')
    
    # Evaluate
    results_tier1 = model.evaluate(val_df, 'tier1')
    
    # Save models
    model.save_models('tier1')
    
    # Optional: Train on Tier2 if requested
    print("\n" + "="*80)
    print("Note: Tier2 has more variance, curve fitting may be less effective")
    print("Proceeding with Tier2 for completeness...")
    print("="*80)
    
    model.train_tier(train_df, 'tier2')
    results_tier2 = model.evaluate(val_df, 'tier2')
    model.save_models('tier2')
    
    # Save summary
    results_df = pd.DataFrame([r for r in [results_tier1, results_tier2] if r is not None])
    results_path = Path('results/step08_curve_fitting_summary.csv')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    
    print("\n" + "="*80)
    print("✅ STEP 8 COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {results_path}")
    print("\nCurve fitting models can extrapolate LTV beyond D60")
    print("Example: Predict D90, D180, or lifetime value")
    print()


if __name__ == "__main__":
    main()
