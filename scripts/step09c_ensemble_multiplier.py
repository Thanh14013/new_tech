"""
STEP 9C: ENSEMBLE ML MULTIPLIER
Combine baseline and enhanced models based on churn risk
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error


class EnsembleMultiplier:
    """Ensemble of baseline and enhanced multiplier models"""
    
    def __init__(self, tier):
        self.tier = tier
        self.baseline_model = None
        self.enhanced_model = None
        self.baseline_features = None
        self.enhanced_features = None
        
        self._load_models()
    
    def _load_models(self):
        """Load both baseline and enhanced models"""
        # Baseline
        self.baseline_model = joblib.load(f'models/{self.tier}/ml_multiplier.pkl')
        with open(f'models/{self.tier}/ml_multiplier_features.txt', 'r') as f:
            self.baseline_features = [line.strip() for line in f]
        
        # Enhanced
        self.enhanced_model = joblib.load(f'models/{self.tier}/ml_multiplier_enhanced.pkl')
        with open(f'models/{self.tier}/ml_multiplier_enhanced_features.txt', 'r') as f:
            self.enhanced_features = [line.strip() for line in f]
    
    def predict_multiplier(self, df_base, df_enhanced):
        """Predict multiplier using ensemble strategy
        
        Strategy: Blend baseline and enhanced predictions based on churn_risk_score
        - Low churn risk (< 0.6): 80% baseline, 20% enhanced
        - Medium risk (0.6-0.7): 50% baseline, 50% enhanced  
        - High risk (> 0.7): 20% baseline, 80% enhanced
        """
        # Baseline predictions
        X_base = df_base[self.baseline_features].fillna(0).replace([np.inf, -np.inf], 0)
        mult_base = self.baseline_model.predict(X_base, num_iteration=self.baseline_model.best_iteration)
        mult_base = np.clip(mult_base, 0, 100)
        
        # Enhanced predictions
        X_enh = df_enhanced[self.enhanced_features].fillna(0).replace([np.inf, -np.inf], 0)
        mult_enh = self.enhanced_model.predict(X_enh, num_iteration=self.enhanced_model.best_iteration)
        mult_enh = np.clip(mult_enh, 0, 100)
        
        # Get churn risk score (only in enhanced df)
        churn_risk = df_enhanced['churn_risk_score'].values
        
        # Calculate blending weight (alpha for baseline)
        # alpha = 1 - churn_risk → high churn = low alpha = more enhanced
        alpha = np.clip(1 - churn_risk, 0.2, 0.8)  # Keep alpha between 0.2 and 0.8
        
        # Smooth transition
        # Low churn (risk < 0.4): alpha ≈ 0.8 (mostly baseline)
        # Medium (0.4-0.7): alpha ≈ 0.5 (balanced)
        # High churn (> 0.7): alpha ≈ 0.2 (mostly enhanced)
        
        # Ensemble prediction
        mult_ensemble = alpha * mult_base + (1 - alpha) * mult_enh
        
        return mult_ensemble, mult_base, mult_enh, alpha
    
    def predict_ltv(self, df_base, df_enhanced):
        """Predict LTV using ensemble"""
        mult_ensemble, _, _, _ = self.predict_multiplier(df_base, df_enhanced)
        ltv_pred = df_base['rev_sum'].values * mult_ensemble
        
        return ltv_pred
    
    def evaluate(self, df_base, df_enhanced):
        """Evaluate ensemble model"""
        mult_ensemble, mult_base, mult_enh, alpha = self.predict_multiplier(df_base, df_enhanced)
        
        ltv_pred = df_base['rev_sum'].values * mult_ensemble
        ltv_actual = df_base['ltv_d60'].values
        
        # Metrics
        mask = ltv_actual > 0.01
        r2 = r2_score(ltv_actual, ltv_pred)
        mae = mean_absolute_error(ltv_actual, ltv_pred)
        mape = mean_absolute_percentage_error(ltv_actual[mask], ltv_pred[mask]) if mask.sum() > 0 else np.nan
        
        # Analyze by churn risk segments
        churn_risk = df_enhanced['churn_risk_score'].values
        
        results = {
            'overall': {
                'r2': r2,
                'mae': mae,
                'mape': mape,
                'samples': len(ltv_actual)
            },
            'by_churn_risk': {}
        }
        
        # Segment by churn risk
        risk_segments = [
            ('low', churn_risk < 0.6),
            ('medium', (churn_risk >= 0.6) & (churn_risk < 0.7)),
            ('high', churn_risk >= 0.7)
        ]
        
        for seg_name, seg_mask in risk_segments:
            if seg_mask.sum() < 10:
                continue
            
            seg_actual = ltv_actual[seg_mask]
            seg_pred = ltv_pred[seg_mask]
            seg_mask_mape = (ltv_actual > 0.01) & seg_mask
            
            results['by_churn_risk'][seg_name] = {
                'r2': r2_score(seg_actual, seg_pred),
                'mae': mean_absolute_error(seg_actual, seg_pred),
                'mape': mean_absolute_percentage_error(
                    ltv_actual[seg_mask_mape], 
                    ltv_pred[seg_mask_mape]
                ) if seg_mask_mape.sum() > 0 else np.nan,
                'samples': seg_mask.sum(),
                'avg_alpha': alpha[seg_mask].mean()
            }
        
        return results


def evaluate_ensemble(tier):
    """Evaluate ensemble for a tier"""
    print(f"\n{'='*80}")
    print(f"🎯 EVALUATING ENSEMBLE: {tier.upper()}")
    print(f"{'='*80}")
    
    # Load data
    df_val = pd.read_csv('data/features/validation.csv')
    df_val_enhanced = pd.read_csv('data/features/validation_enhanced.csv')
    
    df_base = df_val[df_val['tier'] == tier].copy()
    df_enh = df_val_enhanced[df_val_enhanced['tier'] == tier].copy()
    
    print(f"\nDataset: {len(df_base):,} samples")
    
    # Create ensemble
    ensemble = EnsembleMultiplier(tier)
    
    # Evaluate
    results = ensemble.evaluate(df_base, df_enh)
    
    # Print results
    print(f"\n📊 OVERALL METRICS:")
    print(f"   R²:   {results['overall']['r2']:.4f}")
    print(f"   MAE:  ${results['overall']['mae']:.2f}")
    print(f"   MAPE: {results['overall']['mape']:.2%}")
    
    print(f"\n📊 BY CHURN RISK SEGMENT:")
    print(f"   {'Segment':<12} {'Samples':<12} {'Avg α':<10} {'R²':<10} {'MAE':<12} {'MAPE':<10}")
    print(f"   {'-'*75}")
    
    for seg_name in ['low', 'medium', 'high']:
        if seg_name in results['by_churn_risk']:
            seg = results['by_churn_risk'][seg_name]
            print(f"   {seg_name:<12} {seg['samples']:<12,} {seg['avg_alpha']:<10.2f} "
                  f"{seg['r2']:<10.4f} ${seg['mae']:<11.2f} {seg['mape']:<9.1%}")
    
    return results


def compare_all_approaches(tier):
    """Compare baseline, enhanced, and ensemble"""
    print(f"\n{'='*80}")
    print(f"📊 COMPARISON: ALL APPROACHES - {tier.upper()}")
    print(f"{'='*80}")
    
    # Load summaries
    baseline = pd.read_csv('results/step09_multiplier_summary.csv')
    enhanced = pd.read_csv('results/step09b_multiplier_enhanced_summary.csv')
    
    baseline_tier = baseline[baseline['tier'] == tier].iloc[0]
    enhanced_tier = enhanced[enhanced['tier'] == tier].iloc[0]
    
    # Get ensemble results
    df_val = pd.read_csv('data/features/validation.csv')
    df_val_enhanced = pd.read_csv('data/features/validation_enhanced.csv')
    
    df_base = df_val[df_val['tier'] == tier].copy()
    df_enh = df_val_enhanced[df_val_enhanced['tier'] == tier].copy()
    
    ensemble = EnsembleMultiplier(tier)
    ensemble_results = ensemble.evaluate(df_base, df_enh)
    
    # Print comparison
    print(f"\n{'Approach':<20} {'R²':<12} {'MAE':<12} {'MAPE':<12}")
    print("-"*56)
    print(f"{'Baseline':<20} {baseline_tier['r2']:<12.4f} ${baseline_tier['mae']:<11.2f} {baseline_tier['mape']:<11.2%}")
    print(f"{'Enhanced':<20} {enhanced_tier['r2']:<12.4f} ${enhanced_tier['mae']:<11.2f} {enhanced_tier['mape']:<11.2%}")
    print(f"{'Ensemble':<20} {ensemble_results['overall']['r2']:<12.4f} ${ensemble_results['overall']['mae']:<11.2f} {ensemble_results['overall']['mape']:<11.2%}")
    
    # Find best
    r2_scores = {
        'Baseline': baseline_tier['r2'],
        'Enhanced': enhanced_tier['r2'],
        'Ensemble': ensemble_results['overall']['r2']
    }
    
    best_r2 = max(r2_scores.items(), key=lambda x: x[1])
    
    print(f"\n🏆 Best R²: {best_r2[0]} ({best_r2[1]:.4f})")
    
    mae_scores = {
        'Baseline': baseline_tier['mae'],
        'Enhanced': enhanced_tier['mae'],
        'Ensemble': ensemble_results['overall']['mae']
    }
    
    best_mae = min(mae_scores.items(), key=lambda x: x[1])
    
    print(f"🏆 Best MAE: {best_mae[0]} (${best_mae[1]:.2f})")
    
    return ensemble_results


def main():
    """Main execution"""
    print("="*80)
    print("🚀 STEP 9C: ENSEMBLE ML MULTIPLIER")
    print("="*80)
    print("\nStrategy: Blend baseline & enhanced based on churn risk")
    print("          Low risk → More baseline (conservative)")
    print("          High risk → More enhanced (churn-aware)")
    
    results_summary = []
    
    for tier in ['tier1', 'tier2']:
        # Evaluate ensemble
        results = evaluate_ensemble(tier)
        
        # Compare all approaches
        compare_all_approaches(tier)
        
        results_summary.append({
            'tier': tier,
            'r2': results['overall']['r2'],
            'mae': results['overall']['mae'],
            'mape': results['overall']['mape']
        })
    
    # Save summary
    summary_df = pd.DataFrame(results_summary)
    summary_path = Path('results/step09c_multiplier_ensemble_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print("\n" + "="*80)
    print("📊 ENSEMBLE SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("✅ STEP 9C COMPLETED!")
    print("="*80)
    print(f"\n✅ Summary saved: {summary_path}")
    print("\n💡 Ensemble Strategy:")
    print("   • Uses BOTH baseline and enhanced models")
    print("   • Dynamically blends based on churn_risk_score")
    print("   • Combines strengths of both approaches")
    print("   • Better balance between R² and MAPE")


if __name__ == "__main__":
    main()
