"""
STEP 9: ML MULTIPLIER METHOD
Train LightGBM to predict D7->D60 multiplier for early prediction
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import joblib
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error

class MLMultiplierModel:
    """ML-based multiplier prediction using early signals"""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
    
    def prepare_features(self, df):
        """Prepare features for multiplier prediction"""
        # Available features (excluding target and meta columns)
        exclude_cols = ['ltv_d60', 'ltv_d30', 'tier', 'campaign', 'geo', 
                       'platform', 'install_date', 'month', 'month_num',
                       'app_id', 'cpi_quality_tier', 'country']
        
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        X = df[feature_cols].copy()
        
        # Keep only numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        # Fill missing values
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        # Store feature names for later use
        if self.feature_names is None:
            self.feature_names = list(X.columns)
        
        return X
    
    def calculate_target_multiplier(self, df):
        """Calculate target multiplier = ltv_d60 / rev_sum
        
        This multiplier represents how much the early revenue (rev_sum)
        will grow to reach D60 LTV.
        """
        df = df.copy()
        
        # Calculate multiplier (with small epsilon to avoid division by zero)
        df['target_multiplier'] = df['ltv_d60'] / (df['rev_sum'] + 1e-6)
        
        # Clip extreme values (multiplier typically 1-50x)
        df['target_multiplier'] = df['target_multiplier'].clip(0, 100)
        
        return df['target_multiplier']
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train LightGBM model"""
        print("\n📊 Training LightGBM Multiplier Model...")
        print(f"   Features: {X_train.shape[1]}")
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Validation samples: {len(X_val):,}")
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Parameters optimized for multiplier prediction
        params = {
            'objective': 'regression',
            'metric': 'mae',  # Mean absolute error
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': 6,
            'min_data_in_leaf': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'verbosity': -1,
            'force_col_wise': True
        }
        
        # Train with early stopping
        callbacks = [
            lgb.early_stopping(stopping_rounds=20),
            lgb.log_evaluation(50)
        ]
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=callbacks
        )
        
        print(f"\n   ✅ Training complete!")
        print(f"   Best iteration: {self.model.best_iteration}")
        print(f"   Best score: {self.model.best_score['val']['l1']:.4f}")
    
    def predict_multiplier(self, X):
        """Predict multiplier"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        multiplier_pred = self.model.predict(X, num_iteration=self.model.best_iteration)
        
        # Ensure non-negative and reasonable range
        multiplier_pred = np.clip(multiplier_pred, 0, 100)
        
        return multiplier_pred
    
    def predict_ltv(self, X, rev_sum):
        """Predict LTV using multiplier approach
        
        ltv_pred = rev_sum × multiplier_pred
        """
        multiplier_pred = self.predict_multiplier(X)
        ltv_pred = rev_sum * multiplier_pred
        
        return ltv_pred
    
    def evaluate(self, X_val, y_actual_ltv, rev_sum_val):
        """Evaluate model on validation set"""
        print("\n📈 Evaluating Model...")
        
        # Predict LTV
        ltv_pred = self.predict_ltv(X_val, rev_sum_val)
        
        # Filter out very small values for MAPE calculation
        mask = y_actual_ltv > 0.01
        y_filtered = y_actual_ltv[mask]
        pred_filtered = ltv_pred[mask]
        
        # Calculate metrics
        r2 = r2_score(y_actual_ltv, ltv_pred)
        mae = mean_absolute_error(y_actual_ltv, ltv_pred)
        
        # MAPE on filtered values
        if len(y_filtered) > 0:
            mape = mean_absolute_percentage_error(y_filtered, pred_filtered)
        else:
            mape = np.nan
        
        print(f"   R²:   {r2:.4f}")
        print(f"   MAE:  ${mae:.2f}")
        print(f"   MAPE: {mape:.2%} (on {len(y_filtered):,} samples with LTV>$0.01)")
        
        return {
            'r2': r2,
            'mae': mae,
            'mape': mape,
            'samples_evaluated': len(y_filtered)
        }
    
    def get_feature_importance(self, top_n=20):
        """Get top feature importances"""
        if self.model is None:
            return None
        
        importance = self.model.feature_importance(importance_type='gain')
        feature_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_imp.head(top_n)
    
    def save_model(self, tier, models_path='models'):
        """Save model"""
        tier_path = Path(models_path) / tier
        tier_path.mkdir(parents=True, exist_ok=True)
        
        model_file = tier_path / 'ml_multiplier.pkl'
        joblib.dump(self.model, model_file)
        
        # Save feature names
        feature_file = tier_path / 'ml_multiplier_features.txt'
        with open(feature_file, 'w') as f:
            f.write('\n'.join(self.feature_names))
        
        print(f"\n   ✅ Model saved to: {model_file}")
        print(f"   ✅ Features saved to: {feature_file}")


def train_multiplier_for_tier(tier, df_train, df_val):
    """Train multiplier model for a specific tier"""
    print("\n" + "="*80)
    print(f"🎯 TRAINING ML MULTIPLIER: {tier.upper()}")
    print("="*80)
    
    # Filter by tier
    df_train_tier = df_train[df_train['tier'] == tier].copy()
    df_val_tier = df_val[df_val['tier'] == tier].copy()
    
    print(f"\n📊 Dataset:")
    print(f"   Training:   {len(df_train_tier):,} rows")
    print(f"   Validation: {len(df_val_tier):,} rows")
    
    if len(df_train_tier) < 100:
        print(f"   ⚠️  Insufficient training data for {tier}")
        return None
    
    # Initialize model
    ml_mult = MLMultiplierModel()
    
    # Prepare features
    X_train = ml_mult.prepare_features(df_train_tier)
    X_val = ml_mult.prepare_features(df_val_tier)
    
    # Calculate target multiplier
    y_train_mult = ml_mult.calculate_target_multiplier(df_train_tier)
    y_val_mult = ml_mult.calculate_target_multiplier(df_val_tier)
    
    print(f"\n📊 Target Multiplier Statistics:")
    print(f"   Training:   Mean={y_train_mult.mean():.2f}, Median={y_train_mult.median():.2f}")
    print(f"   Validation: Mean={y_val_mult.mean():.2f}, Median={y_val_mult.median():.2f}")
    
    # Train
    ml_mult.train(X_train, y_train_mult, X_val, y_val_mult)
    
    # Evaluate (using actual LTV, not multiplier)
    results = ml_mult.evaluate(
        X_val, 
        df_val_tier['ltv_d60'].values,
        df_val_tier['rev_sum'].values
    )
    
    # Feature importance
    print("\n📊 Top 20 Features by Importance:")
    feat_imp = ml_mult.get_feature_importance(top_n=20)
    for idx, row in feat_imp.iterrows():
        print(f"   {row['feature']:<35} {row['importance']:>10.0f}")
    
    # Save model
    ml_mult.save_model(tier)
    
    return results


def main():
    """Main execution"""
    print("="*80)
    print("🚀 STEP 9: ML MULTIPLIER METHOD")
    print("="*80)
    print("\nGoal: Train LightGBM to predict D7->D60 multiplier for early LTV prediction")
    
    # Load data
    print("\n📂 Loading data...")
    df_train = pd.read_csv('data/features/train.csv')
    df_val = pd.read_csv('data/features/validation.csv')
    
    print(f"   Training: {len(df_train):,} rows")
    print(f"   Validation: {len(df_val):,} rows")
    
    # Train for each tier
    results_summary = []
    
    for tier in ['tier1', 'tier2']:
        results = train_multiplier_for_tier(tier, df_train, df_val)
        
        if results is not None:
            results_summary.append({
                'tier': tier,
                'r2': results['r2'],
                'mae': results['mae'],
                'mape': results['mape'],
                'samples_evaluated': results['samples_evaluated']
            })
    
    # Save summary
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        summary_path = Path('results/step09_multiplier_summary.csv')
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
        
        print("\n" + "="*80)
        print("📊 SUMMARY")
        print("="*80)
        print(summary_df.to_string(index=False))
        
        print("\n" + "="*80)
        print("✅ STEP 9 COMPLETED!")
        print("="*80)
        print(f"\n✅ Models saved:")
        for tier in ['tier1', 'tier2']:
            print(f"   - models/{tier}/ml_multiplier.pkl")
        print(f"\n✅ Summary saved: {summary_path}")
        print("\n➡️  Next Step: Step 10 - Lookalike Model")
    else:
        print("\n⚠️  No models were trained successfully")


if __name__ == "__main__":
    main()
