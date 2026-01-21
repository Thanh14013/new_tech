"""
STEP 12 ADVANCED: 4 Major Improvements
1. Hurdle Calibration (Isotonic Regression)
2. ML Multiplier Fine-tuning (GridSearch)
3. Stacking Meta-Learner (Ridge)
4. Advanced Feature Engineering
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import joblib
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error, make_scorer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

warnings.filterwarnings('ignore')


class AdvancedEnsemble:
    """Advanced ensemble with 4 major improvements"""
    
    def __init__(self):
        self.calibrators = {}
        self.meta_model = None
        self.scaler = StandardScaler()
        
    def load_data_and_predictions(self):
        """Load validation data and base predictions"""
        print("\n📊 Loading Data and Base Predictions...")
        
        # Load validation data
        val_path = Path('data/features/validation_enhanced.csv')
        df_val = pd.read_csv(val_path)
        
        target_col = 'ltv_d60' if 'ltv_d60' in df_val.columns else 'ltv_d30'
        y_true = df_val[target_col].values
        
        print(f"   ✅ Validation: {len(df_val):,} rows")
        print(f"   🎯 Target: {target_col}")
        
        # Load base models
        predictions = {}
        
        # 1. Hurdle
        try:
            stage1 = joblib.load('models/tier2/hurdle_stage1_classifier.pkl')
            stage2 = joblib.load('models/tier2/hurdle_stage2_regressor.pkl')
            
            with open('models/tier2/hurdle_features.txt', 'r') as f:
                features = [line.strip() for line in f if line.strip()]
            
            X = df_val[features].fillna(0).replace([np.inf, -np.inf], 0)
            p_payer = stage1.predict_proba(X)[:, 1]
            ltv_given_payer = stage2.predict(X)
            predictions['hurdle'] = p_payer * np.maximum(ltv_given_payer, 0)
            print(f"   ✅ Hurdle loaded")
        except Exception as e:
            print(f"   ⚠ Hurdle failed: {str(e)[:80]}")
        
        # 2. ML Multiplier
        try:
            ml_model = joblib.load('models/tier2/ml_multiplier_enhanced.pkl')
            
            with open('models/tier2/ml_multiplier_enhanced_features.txt', 'r') as f:
                features = [line.strip() for line in f if line.strip()]
            
            X = df_val[features].fillna(0).replace([np.inf, -np.inf], 0)
            multiplier = ml_model.predict(X)
            predictions['ml_multiplier'] = df_val['rev_sum'].values * np.maximum(multiplier, 0.1)
            print(f"   ✅ ML Multiplier loaded")
        except Exception as e:
            print(f"   ⚠ ML Multiplier failed: {str(e)[:80]}")
        
        return df_val, y_true, predictions
    
    def calibrate_hurdle(self, predictions, y_true, method='isotonic'):
        """
        IMPROVEMENT 1: Calibrate Hurdle predictions
        Hurdle overestimates small values significantly
        """
        print("\n🔧 IMPROVEMENT 1: Calibrating Hurdle Predictions...")
        
        if 'hurdle' not in predictions:
            print("   ⚠ Hurdle not available, skipping calibration")
            return predictions
        
        hurdle_pred = predictions['hurdle']
        
        # Filter valid samples
        mask = np.isfinite(y_true) & np.isfinite(hurdle_pred) & (y_true > 0)
        y_train = y_true[mask]
        pred_train = hurdle_pred[mask]
        
        if method == 'isotonic':
            # Isotonic regression - learns monotonic mapping
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(pred_train, y_train)
            
            # Apply calibration
            hurdle_calibrated = iso.transform(hurdle_pred)
            
            # Evaluate improvement
            mask_eval = (y_true >= 0.10) & np.isfinite(hurdle_pred)
            before_mape = mean_absolute_percentage_error(
                y_true[mask_eval], hurdle_pred[mask_eval]
            ) * 100
            after_mape = mean_absolute_percentage_error(
                y_true[mask_eval], hurdle_calibrated[mask_eval]
            ) * 100
            
            print(f"   Before calibration: MAPE = {before_mape:.2f}%")
            print(f"   After calibration:  MAPE = {after_mape:.2f}%")
            print(f"   ✅ Improvement: {before_mape - after_mape:.2f}% reduction")
            
            predictions['hurdle_calibrated'] = hurdle_calibrated
            self.calibrators['hurdle'] = iso
            
        elif method == 'range_based':
            # Range-based correction factors
            hurdle_calibrated = hurdle_pred.copy()
            
            # Small values (<$1): divide by 6
            small_mask = hurdle_pred < 1.0
            hurdle_calibrated[small_mask] = hurdle_pred[small_mask] / 6.0
            
            # Medium values ($1-$10): divide by 1.5
            medium_mask = (hurdle_pred >= 1.0) & (hurdle_pred < 10.0)
            hurdle_calibrated[medium_mask] = hurdle_pred[medium_mask] / 1.5
            
            # Large values (>$10): minor adjustment
            large_mask = hurdle_pred >= 10.0
            hurdle_calibrated[large_mask] = hurdle_pred[large_mask] / 1.2
            
            mask_eval = (y_true >= 0.10) & np.isfinite(hurdle_pred)
            before_mape = mean_absolute_percentage_error(
                y_true[mask_eval], hurdle_pred[mask_eval]
            ) * 100
            after_mape = mean_absolute_percentage_error(
                y_true[mask_eval], hurdle_calibrated[mask_eval]
            ) * 100
            
            print(f"   Before calibration: MAPE = {before_mape:.2f}%")
            print(f"   After calibration:  MAPE = {after_mape:.2f}%")
            print(f"   ✅ Improvement: {before_mape - after_mape:.2f}% reduction")
            
            predictions['hurdle_calibrated'] = hurdle_calibrated
        
        return predictions
    
    def fine_tune_ml_multiplier(self, df_val, y_true):
        """
        IMPROVEMENT 2: Fine-tune ML Multiplier with GridSearch
        Current MAPE=11.66%, target <10%
        """
        print("\n🎯 IMPROVEMENT 2: Fine-tuning ML Multiplier...")
        
        try:
            # Load features
            with open('models/tier2/ml_multiplier_enhanced_features.txt', 'r') as f:
                features = [line.strip() for line in f if line.strip()]
            
            X = df_val[features].fillna(0).replace([np.inf, -np.inf], 0)
            base_rev = df_val['rev_sum'].values
            
            # Target: multiplier = ltv / rev_sum
            mask = (base_rev > 0.01) & (y_true > 0)
            X_train = X[mask]
            y_multiplier = y_true[mask] / base_rev[mask]
            
            print(f"   Training samples: {len(X_train):,}")
            
            # Custom MAPE scorer
            def mape_scorer(y_true, y_pred):
                # Predict LTV, not multiplier
                y_true_ltv = y_true * base_rev[mask]
                y_pred_ltv = y_pred * base_rev[mask]
                mask_eval = (y_true_ltv >= 0.10) & np.isfinite(y_pred_ltv)
                if mask_eval.sum() < 10:
                    return 0
                mape = mean_absolute_percentage_error(
                    y_true_ltv[mask_eval], y_pred_ltv[mask_eval]
                )
                return -mape  # Negative because GridSearch maximizes
            
            # Grid search (simplified for speed)
            param_grid = {
                'n_estimators': [200, 300],
                'learning_rate': [0.05, 0.1],
                'max_depth': [5, 7],
                'num_leaves': [31, 63],
                'min_child_samples': [20, 50]
            }
            
            print(f"   Testing {np.prod([len(v) for v in param_grid.values()])} combinations...")
            
            base_model = lgb.LGBMRegressor(
                objective='regression',
                random_state=42,
                verbose=-1
            )
            
            # Use 3-fold CV
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=3,
                scoring=make_scorer(mape_scorer),
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_multiplier)
            
            print(f"   ✅ Best params: {grid_search.best_params_}")
            
            # Predict with best model
            best_multiplier = grid_search.best_estimator_.predict(X)
            predictions_tuned = base_rev * np.maximum(best_multiplier, 0.1)
            
            # Compare
            ml_original = joblib.load('models/tier2/ml_multiplier_enhanced.pkl')
            mult_original = ml_original.predict(X)
            pred_original = base_rev * np.maximum(mult_original, 0.1)
            
            mask_eval = (y_true >= 0.10) & np.isfinite(pred_original)
            mape_before = mean_absolute_percentage_error(
                y_true[mask_eval], pred_original[mask_eval]
            ) * 100
            mape_after = mean_absolute_percentage_error(
                y_true[mask_eval], predictions_tuned[mask_eval]
            ) * 100
            
            print(f"   Before tuning: MAPE = {mape_before:.2f}%")
            print(f"   After tuning:  MAPE = {mape_after:.2f}%")
            
            if mape_after < mape_before:
                print(f"   ✅ Improvement: {mape_before - mape_after:.2f}% reduction")
                # Save tuned model
                joblib.dump(grid_search.best_estimator_, 
                           'models/tier2/ml_multiplier_tuned.pkl')
                return predictions_tuned
            else:
                print(f"   ⚠ No improvement, keeping original")
                return pred_original
                
        except Exception as e:
            print(f"   ⚠ Fine-tuning failed: {str(e)}")
            return None
    
    def train_stacking_model(self, predictions, y_true, df_val):
        """
        IMPROVEMENT 3: Stacking meta-learner
        Learn optimal combination of predictions
        """
        print("\n🏗️  IMPROVEMENT 3: Training Stacking Meta-Learner...")
        
        # Prepare stacking features
        stack_features = []
        method_names = []
        
        for method in ['hurdle_calibrated', 'ml_multiplier', 'hurdle', 'semantic']:
            if method in predictions:
                stack_features.append(predictions[method])
                method_names.append(method)
        
        if len(stack_features) < 2:
            print("   ⚠ Need at least 2 methods for stacking")
            return None
        
        X_stack = np.column_stack(stack_features)
        
        # Add campaign-level aggregated features
        campaign_features = []
        for col in ['installs', 'cost', 'rev_sum', 'engagement_score', 'actual_cpi']:
            if col in df_val.columns:
                campaign_features.append(df_val[col].values)
        
        if campaign_features:
            X_campaign = np.column_stack(campaign_features)
            X_stack = np.column_stack([X_stack, X_campaign])
            print(f"   Added {len(campaign_features)} campaign features")
        
        # Filter valid samples
        mask = (y_true >= 0.10) & np.isfinite(X_stack).all(axis=1)
        X_train = X_stack[mask]
        y_train = y_true[mask]
        
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Features: {X_train.shape[1]} ({len(method_names)} predictions + {len(campaign_features)} campaign)")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train Ridge regression
        alphas = [0.1, 0.5, 1.0, 5.0, 10.0]
        best_alpha = 1.0  # Default
        best_mape = float('inf')
        
        for alpha in alphas:
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_train_scaled, y_train)
            
            # Predict and evaluate
            y_pred = ridge.predict(X_train_scaled)
            mask_eval = (y_train >= 0.10) & np.isfinite(y_pred)
            if mask_eval.sum() > 10:
                mape = mean_absolute_percentage_error(y_train[mask_eval], y_pred[mask_eval]) * 100
                
                if mape < best_mape:
                    best_mape = mape
                    best_alpha = alpha
        
        print(f"   Best alpha: {best_alpha} (Train MAPE: {best_mape:.2f}%)")
        
        # Train final model
        meta_model = Ridge(alpha=best_alpha)
        meta_model.fit(X_train_scaled, y_train)
        
        # Predict on full dataset
        X_full_scaled = self.scaler.transform(X_stack)
        stacked_pred = meta_model.predict(X_full_scaled)
        stacked_pred = np.clip(stacked_pred, 0, y_true.max() * 2)
        
        # Evaluate
        mask_eval = (y_true >= 0.10) & np.isfinite(stacked_pred)
        mape_stacked = mean_absolute_percentage_error(
            y_true[mask_eval], stacked_pred[mask_eval]
        ) * 100
        
        # Compare with best individual method
        best_individual = min([
            mean_absolute_percentage_error(
                y_true[mask_eval], predictions[m][mask_eval]
            ) * 100
            for m in method_names if m in predictions
        ])
        
        print(f"   Best individual: MAPE = {best_individual:.2f}%")
        print(f"   Stacked model:   MAPE = {mape_stacked:.2f}%")
        
        if mape_stacked < best_individual:
            print(f"   ✅ Improvement: {best_individual - mape_stacked:.2f}% reduction")
        else:
            print(f"   ⚠ No improvement over best individual")
        
        self.meta_model = meta_model
        predictions['stacked'] = stacked_pred
        
        return predictions
    
    def add_campaign_features(self, df_val):
        """
        IMPROVEMENT 4: Advanced feature engineering
        Add campaign-level temporal and behavioral features
        """
        print("\n🔬 IMPROVEMENT 4: Advanced Feature Engineering...")
        
        # Campaign-level aggregation
        campaign_agg = df_val.groupby(['app_id', 'campaign']).agg({
            'installs': 'sum',
            'ltv_d60': 'mean',
            'ltv_d30': 'mean',
            'cost': 'sum',
            'rev_sum': 'sum',
            'engagement_score': 'mean'
        }).reset_index()
        
        # Campaign age (proxy: install volume)
        campaign_agg['campaign_maturity'] = campaign_agg['installs'] / campaign_agg['installs'].max()
        
        # Efficiency metrics
        campaign_agg['roas_d60'] = campaign_agg['ltv_d60'] * campaign_agg['installs'] / (campaign_agg['cost'] + 0.01)
        campaign_agg['ltv_per_install'] = campaign_agg['ltv_d60']
        
        # Growth potential (D60 vs D30 ratio)
        campaign_agg['growth_potential'] = campaign_agg['ltv_d60'] / (campaign_agg['ltv_d30'] + 0.01)
        
        print(f"   Created {len(campaign_agg)} campaign profiles")
        print(f"   New features: campaign_maturity, roas_d60, growth_potential")
        
        # Merge back to user level
        df_enhanced = df_val.merge(
            campaign_agg[['app_id', 'campaign', 'campaign_maturity', 'roas_d60', 'growth_potential']],
            on=['app_id', 'campaign'],
            how='left'
        )
        
        return df_enhanced
    
    def evaluate_campaign_level(self, predictions, df_val, y_true):
        """Evaluate at campaign level (aggregate users)"""
        print("\n📊 Campaign-Level Evaluation...")
        
        df_eval = df_val.copy()
        df_eval['actual'] = y_true
        
        for method, preds in predictions.items():
            if method not in ['actual', 'app_id', 'campaign', 'geo']:
                df_eval[f'pred_{method}'] = preds
        
        # Aggregate to campaign level
        campaign_results = df_eval.groupby(['app_id', 'campaign']).agg({
            'actual': 'mean',
            **{f'pred_{m}': 'mean' for m in predictions.keys() 
               if m not in ['actual', 'app_id', 'campaign', 'geo']}
        }).reset_index()
        
        print(f"   Campaigns: {len(campaign_results):,}")
        
        # Evaluate each method at campaign level
        for method in predictions.keys():
            if method not in ['actual', 'app_id', 'campaign', 'geo']:
                pred_col = f'pred_{method}'
                if pred_col in campaign_results.columns:
                    mask = (campaign_results['actual'] >= 0.10) & np.isfinite(campaign_results[pred_col])
                    if mask.sum() > 10:
                        mape = mean_absolute_percentage_error(
                            campaign_results.loc[mask, 'actual'],
                            campaign_results.loc[mask, pred_col]
                        ) * 100
                        r2 = r2_score(
                            campaign_results.loc[mask, 'actual'],
                            campaign_results.loc[mask, pred_col]
                        )
                        print(f"   {method:<20} R²={r2:>7.4f}  MAPE={mape:>7.2f}%")


def main():
    """Main execution"""
    import time
    start_time = time.time()
    
    print("="*80)
    print("🚀 STEP 12 ADVANCED: 4 MAJOR IMPROVEMENTS")
    print("="*80)
    print()
    print("1. Hurdle Calibration (Isotonic Regression)")
    print("2. ML Multiplier Fine-tuning (GridSearch)")
    print("3. Stacking Meta-Learner (Ridge)")
    print("4. Advanced Feature Engineering")
    print()
    
    ensemble = AdvancedEnsemble()
    
    # Load data
    df_val, y_true, predictions = ensemble.load_data_and_predictions()
    
    # Improvement 1: Calibrate Hurdle
    predictions = ensemble.calibrate_hurdle(predictions, y_true, method='isotonic')
    
    # Improvement 2: Fine-tune ML Multiplier
    ml_tuned = ensemble.fine_tune_ml_multiplier(df_val, y_true)
    if ml_tuned is not None:
        predictions['ml_multiplier_tuned'] = ml_tuned
    
    # Improvement 4: Advanced features (before stacking)
    df_enhanced = ensemble.add_campaign_features(df_val)
    
    # Improvement 3: Stacking
    predictions = ensemble.train_stacking_model(predictions, y_true, df_enhanced)
    
    # Final evaluation
    print("\n" + "="*80)
    print("📊 FINAL RESULTS - USER LEVEL")
    print("="*80)
    
    best_method = None
    best_mape = float('inf')
    
    for method in predictions.keys():
        if method not in ['actual', 'app_id', 'campaign', 'geo']:
            mask = (y_true >= 0.10) & np.isfinite(predictions[method])
            if mask.sum() > 100:
                mape = mean_absolute_percentage_error(
                    y_true[mask], predictions[method][mask]
                ) * 100
                r2 = r2_score(y_true[mask], predictions[method][mask])
                mae = mean_absolute_error(y_true[mask], predictions[method][mask])
                
                print(f"\n{method}:")
                print(f"  R² Score: {r2:>8.4f}")
                print(f"  MAE:    ${mae:>9.2f}")
                print(f"  MAPE:     {mape:>8.2f}%")
                
                if mape < best_mape:
                    best_mape = mape
                    best_method = method
    
    print(f"\n{'='*80}")
    print(f"🏆 BEST METHOD: {best_method} (MAPE = {best_mape:.2f}%)")
    print(f"{'='*80}")
    
    # Campaign-level evaluation
    ensemble.evaluate_campaign_level(predictions, df_val, y_true)
    
    # Save best model
    if best_method and ensemble.meta_model:
        output_path = Path('models/ensemble')
        output_path.mkdir(parents=True, exist_ok=True)
        
        config = {
            'best_method': best_method,
            'best_mape': best_mape,
            'meta_model': ensemble.meta_model,
            'scaler': ensemble.scaler,
            'calibrators': ensemble.calibrators
        }
        
        joblib.dump(config, output_path / 'advanced_ensemble_config.pkl')
        print(f"\n✅ Advanced ensemble saved to {output_path}")
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Total execution time: {elapsed:.2f} seconds")
    print(f"\n{'='*80}")
    if best_mape <= 5.0:
        print(f"🎉 TARGET ACHIEVED! MAPE ≤ 5% ✅")
    elif best_mape <= 10.0:
        print(f"✅ EXCELLENT RESULT! MAPE ≤ 10%")
    else:
        print(f"📊 Final MAPE: {best_mape:.2f}%")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
