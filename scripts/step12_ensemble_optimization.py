"""
STEP 12: CALIBRATION & OPTIMIZATION
Ensemble selection and final calibration
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import joblib
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error

# Suppress warnings
warnings.filterwarnings('ignore')


class EnsembleOptimizer:
    """Ensemble predictions from all models with optimal method selection"""
    
    def __init__(self):
        self.models = {}
        self.method_performance = {}
        
    def load_features_from_txt(self, txt_path):
        """Load feature names from .txt file"""
        with open(txt_path, 'r') as f:
            features = [line.strip() for line in f if line.strip()]
        return features
    
    def load_validation_predictions(self):
        """Load or generate predictions from all methods"""
        print("\n📊 Loading Validation Predictions from All Methods...")
        
        # Load validation data (enhanced)
        val_path = Path('data/features/validation_enhanced.csv')
        if not val_path.exists():
            val_path = Path('data/features/validation.csv')
        
        df_val = pd.read_csv(val_path)
        print(f"   ✅ Validation data: {len(df_val):,} rows")
        
        # Use ltv_d60 as target (requirement: D1 -> D60 prediction)
        target_col = 'ltv_d60' if 'ltv_d60' in df_val.columns else 'ltv_d30'
        print(f"   🎯 Target: {target_col}")
        
        predictions = {
            'actual': df_val[target_col].values,
            'app_id': df_val['app_id'].values,
            'campaign': df_val['campaign'].values,
            'geo': df_val['geo'].values if 'geo' in df_val.columns else None
        }
        
        print("\n   📦 Loading model predictions...")
        
        # 1. Hurdle Model (Tier2)
        hurdle_path = Path('models/tier2/hurdle_stage2_regressor.pkl')
        if hurdle_path.exists():
            try:
                # Load models
                stage1_model = joblib.load(Path('models/tier2/hurdle_stage1_classifier.pkl'))
                stage2_model = joblib.load(hurdle_path)
                
                # Load features from txt
                feature_path = Path('models/tier2/hurdle_features.txt')
                if feature_path.exists():
                    feature_cols = self.load_features_from_txt(feature_path)
                else:
                    # Default features
                    feature_cols = ['rev_sum', 'rev_max', 'rev_last', 'engagement_score', 
                                   'actual_cpi', 'cpi_quality_score']
                
                # Prepare features
                X = df_val[feature_cols].copy()
                X = X.fillna(0).replace([np.inf, -np.inf], 0)
                
                # Predict
                p_payer = stage1_model.predict_proba(X)[:, 1]
                ltv_given_payer = stage2_model.predict(X)
                predictions['hurdle'] = p_payer * np.maximum(ltv_given_payer, 0)
                
                print(f"      ✅ Hurdle predictions: {len(predictions['hurdle']):,} samples")
            except Exception as e:
                print(f"      ⚠ Hurdle failed: {str(e)[:100]}")
        
        # 2. ML Multiplier Enhanced
        ml_mult_path = Path('models/tier2/ml_multiplier_enhanced.pkl')
        if ml_mult_path.exists():
            try:
                ml_model = joblib.load(ml_mult_path)
                
                # Load features
                feature_path = Path('models/tier2/ml_multiplier_enhanced_features.txt')
                if feature_path.exists():
                    feature_cols = self.load_features_from_txt(feature_path)
                else:
                    feature_path = Path('models/tier2/ml_multiplier_features.txt')
                    feature_cols = self.load_features_from_txt(feature_path)
                
                # Prepare features
                X = df_val[feature_cols].copy()
                X = X.fillna(0).replace([np.inf, -np.inf], 0)
                
                # Predict multiplier
                multiplier = ml_model.predict(X)
                base_rev = df_val['rev_sum'].values
                predictions['ml_multiplier'] = base_rev * np.maximum(multiplier, 0.1)
                
                print(f"      ✅ ML Multiplier predictions: {len(predictions['ml_multiplier']):,} samples")
            except Exception as e:
                print(f"      ⚠ ML Multiplier failed: {str(e)[:100]}")
        
        # 3. Lookalike - Load actual model
        lookalike_kmeans = Path('models/fallback/lookalike_kmeans.pkl')
        if lookalike_kmeans.exists():
            try:
                # Add parent directory to sys.path
                import sys
                parent_dir = str(Path(__file__).parent.parent)
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
                
                from scripts.step10_train_lookalike import LookalikeModel
                
                # Initialize model
                lookalike = LookalikeModel(n_clusters=20)
                lookalike.scaler = joblib.load(Path('models/fallback/lookalike_scaler.pkl'))
                lookalike.kmeans = joblib.load(lookalike_kmeans)
                lookalike.cluster_stats = joblib.load(Path('models/fallback/lookalike_cluster_stats.pkl'))
                lookalike.feature_cols = joblib.load(Path('models/fallback/lookalike_features.pkl'))
                lookalike.global_mean_ltv = joblib.load(Path('models/fallback/lookalike_global_mean.pkl'))
                
                # Aggregate to campaign-geo level with dynamic column checking
                from scripts.step10_train_lookalike import LookalikeModel
                
                # Build aggregation dict dynamically based on available columns and feature needs
                agg_dict = {}
                
                # Define aggregation rules for each feature
                agg_rules = {
                    'installs': 'sum',
                    'cost': 'sum',
                    'rev_sum': 'sum',
                    'rev_max': 'max',
                    'rev_last': 'mean',
                    'rev_volatility': 'mean',
                    'rev_growth_rate': 'mean',
                    'engagement_score': 'mean',
                    'actual_cpi': 'mean',
                    'cpi_quality_score': 'mean',
                    'campaign_ltv_avg': 'mean',
                    'campaign_engagement_avg': 'mean',
                    'campaign_total_installs': 'sum',
                    'ltv_d30': 'mean'
                }
                
                # Only add columns that exist in df_val
                for col, agg_func in agg_rules.items():
                    if col in df_val.columns:
                        agg_dict[col] = agg_func
                
                campaign_features = df_val.groupby(['app_id', 'campaign', 'geo']).agg(agg_dict).reset_index()
                
                # OPTIMIZED: Predict for unique campaigns only, then map to all rows
                print(f"      🔄 Predicting for {len(campaign_features):,} unique campaigns...")
                
                # Step 1: Create predictions for unique campaigns
                campaign_preds = {}
                for idx, camp_row in campaign_features.iterrows():
                    camp_key = (camp_row['app_id'], camp_row['campaign'], camp_row['geo'])
                    try:
                        pred, _, _ = lookalike.predict_campaign(camp_row)
                        campaign_preds[camp_key] = pred
                    except:
                        campaign_preds[camp_key] = lookalike.global_mean_ltv
                
                # Step 2: Vectorized mapping to all validation rows
                df_val['_temp_camp_key'] = list(zip(df_val['app_id'], df_val['campaign'], df_val['geo']))
                predictions['lookalike'] = df_val['_temp_camp_key'].map(campaign_preds).fillna(lookalike.global_mean_ltv).values
                df_val.drop('_temp_camp_key', axis=1, inplace=True)
                
                print(f"      ✅ Lookalike predictions: {len(predictions['lookalike']):,} samples")
            except Exception as e:
                import traceback
                print(f"      ⚠ Lookalike failed: {str(e)}")
                # traceback.print_exc()  # Uncomment for full traceback
        
        # 4. Semantic
        semantic_ltv_map = Path('models/semantic/campaign_ltv_map.pkl')
        if semantic_ltv_map.exists():
            try:
                campaign_ltv = joblib.load(semantic_ltv_map)
                global_mean = joblib.load(Path('models/semantic/global_mean_ltv.pkl'))
                
                # OPTIMIZED: Vectorized campaign ID creation and mapping
                df_val['_temp_campaign_id'] = df_val['app_id'].astype(str) + '::' + df_val['campaign'].astype(str)
                predictions['semantic'] = df_val['_temp_campaign_id'].map(campaign_ltv).fillna(global_mean).values
                df_val.drop('_temp_campaign_id', axis=1, inplace=True)
                
                print(f"      ✅ Semantic predictions: {len(predictions['semantic']):,} samples")
            except Exception as e:
                print(f"      ⚠ Semantic failed: {str(e)[:100]}")
        
        return predictions, df_val
    
    def evaluate_method_performance(self, predictions, df_val):
        """Evaluate each method's performance"""
        print("\n📈 Evaluating Method Performance...")
        
        y_true = predictions['actual']
        
        methods = [k for k in predictions.keys() if k not in ['actual', 'app_id', 'campaign', 'geo']]
        
        results = []
        
        print(f"\n   {'Method':<20} {'R²':>8} {'MAE':>10} {'MAPE':>10} {'Samples':>10}")
        print(f"   {'-'*60}")
        
        for method in methods:
            if method not in predictions:
                continue
                
            y_pred = predictions[method]
            
            # Remove any NaN or inf AND filter out very small y_true (< $0.10) to avoid extreme MAPE
            mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true >= 0.10)
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            
            if len(y_true_clean) > 10:
                r2 = r2_score(y_true_clean, y_pred_clean)
                mae = mean_absolute_error(y_true_clean, y_pred_clean)
                
                # MAPE: proper calculation (already filtered small y_true)
                mape = mean_absolute_percentage_error(y_true_clean, y_pred_clean) * 100
                
                results.append({
                    'method': method,
                    'r2': r2,
                    'mae': mae,
                    'mape': mape,
                    'samples': len(y_true_clean)
                })
                
                print(f"   {method:<20} {r2:>8.4f} ${mae:>9.2f} {mape:>9.2f}% {len(y_true_clean):>10,}")
        
        results_df = pd.DataFrame(results)
        self.method_performance = results_df
        
        # Filter out poor performing methods (R² < -1 or MAPE > 500%)
        # Also filter if MAPE > 50% when a method with MAPE < 20% exists
        if len(results_df) > 0:
            # First, check if there's an excellent method (MAPE < 20%)
            excellent_methods = results_df[results_df['mape'] < 20]['method'].tolist()
            
            if excellent_methods:
                # If we have excellent methods, be strict with others
                poor_methods = results_df[
                    (results_df['r2'] < -1) | 
                    (results_df['mape'] > 50) |
                    ((results_df['mape'] > 100) & (~results_df['method'].isin(excellent_methods)))
                ]['method'].tolist()
            else:
                # No excellent methods, use original filter
                poor_methods = results_df[(results_df['r2'] < -1) | (results_df['mape'] > 500)]['method'].tolist()
            
            if poor_methods:
                print(f"\n   ⚠️  Excluding poor methods: {', '.join(poor_methods)}")
                for method in poor_methods:
                    if method in predictions:
                        del predictions[method]
        
        return results_df
    
    def create_weighted_ensemble(self, predictions, weights=None):
        """Create weighted ensemble of predictions"""
        print("\n🎯 Creating Weighted Ensemble...")
        
        methods = [k for k in predictions.keys() if k not in ['actual', 'app_id', 'campaign', 'geo']]
        
        if weights is None:
            # Default: equal weights, but prioritize hurdle if available
            if 'hurdle' in methods:
                weights = {'hurdle': 0.4}
                other_weight = 0.6 / (len(methods) - 1) if len(methods) > 1 else 0.6
                for m in methods:
                    if m != 'hurdle':
                        weights[m] = other_weight
            else:
                weights = {m: 1.0 / len(methods) for m in methods}
        
        print(f"   Weights: {weights}")
        
        # Weighted average
        ensemble_pred = np.zeros(len(predictions['actual']))
        
        for method, weight in weights.items():
            if method in predictions:
                ensemble_pred += weight * predictions[method]
        
        return ensemble_pred
    
    def optimize_ensemble_weights(self, predictions, df_val):
        """Find optimal weights for ensemble (simplified grid search)"""
        print("\n⚙️ Optimizing Ensemble Weights...")
        
        y_true = predictions['actual']
        methods = [k for k in predictions.keys() if k not in ['actual', 'app_id', 'campaign', 'geo']]
        
        if len(methods) <= 1:
            print("   ⚠ Only one method available, using default weights")
            return None
        
        best_mape = float('inf')
        best_weights = None
        
        # Enhanced grid search - prioritize best performing method (typically ML Multiplier)
        print(f"   Testing weight combinations for {len(methods)} methods...")
        
        # Strategy 1: If ML Multiplier available, prioritize it (lowest MAPE)
        if 'ml_multiplier' in methods:
            for ml_w in [0.5, 0.6, 0.7, 0.8]:
                remaining = 1.0 - ml_w
                other_methods = [m for m in methods if m != 'ml_multiplier']
                
                if len(other_methods) > 0:
                    # Distribute remaining weight
                    if 'hurdle' in other_methods:
                        hurdle_w = remaining * 0.7  # Give more to hurdle
                        semantic_w = remaining * 0.3
                        test_weights = {
                            'ml_multiplier': ml_w,
                            'hurdle': hurdle_w,
                            'semantic': semantic_w
                        }
                    else:
                        other_w = remaining / len(other_methods)
                        test_weights = {'ml_multiplier': ml_w}
                        for m in other_methods:
                            test_weights[m] = other_w
                    
                    # Calculate ensemble
                    ensemble_pred = self.create_weighted_ensemble(predictions, test_weights)
                    
                    # MAPE with proper filtering (>=$0.10)
                    mask = (y_true >= 0.10) & np.isfinite(ensemble_pred)
                    if mask.sum() > 0:
                        mape = mean_absolute_percentage_error(y_true[mask], ensemble_pred[mask])
                        
                        if mape < best_mape:
                            best_mape = mape
                            best_weights = test_weights.copy()
        
        # Strategy 2: If no ML Multiplier, prioritize hurdle
        elif 'hurdle' in methods:
            for hurdle_w in [0.5, 0.6, 0.7, 0.8]:
                remaining = 1.0 - hurdle_w
                other_methods = [m for m in methods if m != 'hurdle']
                
                if len(other_methods) > 0:
                    other_w = remaining / len(other_methods)
                    
                    test_weights = {'hurdle': hurdle_w}
                    for m in other_methods:
                        test_weights[m] = other_w
                    
                    # Calculate ensemble
                    ensemble_pred = self.create_weighted_ensemble(predictions, test_weights)
                    
                    # MAPE with proper filtering
                    mask = (y_true >= 0.10) & np.isfinite(ensemble_pred)
                    if mask.sum() > 0:
                        mape = mean_absolute_percentage_error(y_true[mask], ensemble_pred[mask])
                        
                        if mape < best_mape:
                            best_mape = mape
                            best_weights = test_weights.copy()
        
        if best_weights is not None:
            print(f"   ✅ Best MAPE: {best_mape*100:.2f}%")
            print(f"   ✅ Optimal weights: {best_weights}")
        else:
            print("   ⚠ Using default equal weights")
            best_weights = {m: 1.0/len(methods) for m in methods}
        
        return best_weights
    
    def evaluate_final_ensemble(self, ensemble_pred, y_true):
        """Evaluate final ensemble performance"""
        print("\n🎊 Final Ensemble Evaluation...")
        
        # Calibration: clip predictions to reasonable range
        ensemble_pred = np.clip(ensemble_pred, 0, y_true.max() * 2)
        
        # Remove any NaN or inf and filter small values for MAPE
        mask = np.isfinite(y_true) & np.isfinite(ensemble_pred) & (y_true >= 0.10)
        y_true_clean = y_true[mask]
        y_pred_clean = ensemble_pred[mask]
        
        if len(y_true_clean) > 0:
            r2 = r2_score(y_true_clean, y_pred_clean)
            mae = mean_absolute_error(y_true_clean, y_pred_clean)
            mape = mean_absolute_percentage_error(y_true_clean, y_pred_clean) * 100
            
            print(f"\n   📊 Final Metrics:")
            print(f"      R² Score:        {r2:>8.4f}")
            print(f"      MAE:           ${mae:>9.2f}")
            print(f"      MAPE:           {mape:>8.2f}%")
            print(f"      Samples:        {len(y_true_clean):>9,}")
            
            return {
                'r2': r2,
                'mae': mae,
                'mape': mape,
                'samples': len(y_true_clean)
            }
        else:
            return None
    
    def save_ensemble_config(self, weights, performance):
        """Save ensemble configuration"""
        output_path = Path('models/ensemble')
        output_path.mkdir(parents=True, exist_ok=True)
        
        config = {
            'weights': weights,
            'performance': performance,
            'methods_used': list(weights.keys())
        }
        
        joblib.dump(config, output_path / 'ensemble_config.pkl')
        
        # Save as CSV too
        weights_df = pd.DataFrame([
            {'method': k, 'weight': v} for k, v in weights.items()
        ])
        weights_df.to_csv(output_path / 'ensemble_weights.csv', index=False)
        
        print(f"\n   ✅ Ensemble config saved to: {output_path}")


def main():
    """Main execution"""
    import time
    start_time = time.time()
    
    print("="*80)
    print("🚀 STEP 12: CALIBRATION & OPTIMIZATION")
    print("="*80)
    print()
    print("Goal: Ensemble all methods with optimal weights")
    print("      Methods: Hurdle, ML Multiplier, Lookalike, Semantic")
    print()
    
    # Initialize
    optimizer = EnsembleOptimizer()
    
    # Load all predictions
    predictions, df_val = optimizer.load_validation_predictions()
    
    # Evaluate individual methods
    method_results = optimizer.evaluate_method_performance(predictions, df_val)
    
    # Optimize ensemble weights
    best_weights = optimizer.optimize_ensemble_weights(predictions, df_val)
    
    # If no weights found (single method), create default
    if best_weights is None:
        methods = [k for k in predictions.keys() if k not in ['actual', 'app_id', 'campaign', 'geo']]
        best_weights = {m: 1.0/len(methods) for m in methods}
    
    # Create final ensemble
    ensemble_pred = optimizer.create_weighted_ensemble(predictions, best_weights)
    
    # Evaluate ensemble
    ensemble_results = optimizer.evaluate_final_ensemble(ensemble_pred, predictions['actual'])
    
    # Save configuration
    if ensemble_results is not None:
        optimizer.save_ensemble_config(best_weights, ensemble_results)
    
    # Save summaries
    results_path = Path('results')
    results_path.mkdir(exist_ok=True)
    
    # Individual method performance
    method_results.to_csv('results/step12_individual_methods.csv', index=False)
    
    # Ensemble performance
    if ensemble_results is not None:
        ensemble_df = pd.DataFrame([ensemble_results])
        ensemble_df.to_csv('results/step12_ensemble_final.csv', index=False)
    
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    
    print("\n1️⃣ Individual Methods:")
    print(method_results.to_string(index=False))
    
    if ensemble_results is not None:
        print("\n2️⃣ Final Ensemble:")
        print(f"   R² = {ensemble_results['r2']:.4f}")
        print(f"   MAE = ${ensemble_results['mae']:.2f}")
        print(f"   MAPE = {ensemble_results['mape']:.2f}%")
    
    print("\n" + "="*80)
    print("✅ STEP 12 COMPLETED!")
    print("="*80)
    print()
    print("✅ Summaries saved:")
    print("   - results/step12_individual_methods.csv")
    print("   - results/step12_ensemble_final.csv")
    print()
    print("✅ Ensemble config saved:")
    print("   - models/ensemble/ensemble_config.pkl")
    print("   - models/ensemble/ensemble_weights.csv")
    print()
    
    elapsed = time.time() - start_time
    print(f"⏱️  Execution time: {elapsed:.2f} seconds")
    print()
    print("➡️  Next Step: Step 13 - Production Pipeline")


if __name__ == "__main__":
    main()
