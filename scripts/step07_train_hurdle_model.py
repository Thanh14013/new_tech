"""
Step 7: Two-Stage Hurdle Model
================================
Train XGBClassifier (Stage 1) + XGBRegressor (Stage 2) for zero-inflated LTV
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import joblib
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import roc_auc_score, mean_absolute_percentage_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class HurdleModel:
    """Two-Stage Hurdle Model for Zero-Inflated LTV Data"""
    
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.stage1_model = None  # Classifier: Will pay?
        self.stage2_model = None  # Regressor: How much?
        self.feature_names = None
    
    def prepare_features(self, df):
        """Prepare feature matrix X from available features"""
        
        # Use all available numeric features except targets
        exclude_cols = ['app_id', 'campaign', 'install_date', 'geo', 'tier', 
                       'ltv_d30', 'ltv_d60', 'roas_d30', 'roas_d60',
                       'month', 'month_num', 'cpi_quality_tier']
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c not in exclude_cols]
        
        X = df[feature_cols].copy()
        
        # Handle missing/inf values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        if self.feature_names is None:
            self.feature_names = X.columns.tolist()
        
        return X
    
    def train_stage1_classifier(self, X_train, y_train, X_val, y_val):
        """Train Stage 1: Will Pay? (Binary Classification)"""
        print("\n" + "─"*70)
        print("[STAGE 1] Training Payer Classifier")
        print("─"*70)
        
        print(f"\n  Training data:")
        print(f"    Total: {len(X_train):,} samples")
        print(f"    Payers: {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
        print(f"    Non-payers: {(~y_train.astype(bool)).sum():,} ({(1-y_train.mean())*100:.1f}%)")
        
        # XGBoost Classifier
        self.stage1_model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            random_state=self.config['modeling']['random_seed'],
            eval_metric='auc',
            verbosity=0
        )
        
        # Train with early stopping
        self.stage1_model.set_params(early_stopping_rounds=20)
        self.stage1_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Predict probabilities
        y_train_proba = self.stage1_model.predict_proba(X_train)[:, 1]
        y_val_proba = self.stage1_model.predict_proba(X_val)[:, 1]
        
        # Metrics
        auc_train = roc_auc_score(y_train, y_train_proba)
        auc_val = roc_auc_score(y_val, y_val_proba)
        
        print(f"\n  ✓ Stage 1 Performance (AUC):")
        print(f"    Train: {auc_train:.4f}")
        print(f"    Val:   {auc_val:.4f}")
        
        # Feature importance (top 10)
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.stage1_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n  Top 5 features for classification:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"    {row['feature']:30s}: {row['importance']:.4f}")
        
        return y_train_proba, y_val_proba
    
    def train_stage2_regressor(self, X_train_payers, y_train_payers, 
                               X_val_payers, y_val_payers):
        """Train Stage 2: How Much? (Regression on Payers Only)"""
        print("\n" + "─"*70)
        print("[STAGE 2] Training Amount Regressor (Payers Only)")
        print("─"*70)
        
        print(f"\n  Training data (payers only):")
        print(f"    Train payers: {len(X_train_payers):,}")
        print(f"    Val payers:   {len(X_val_payers):,}")
        print(f"    Train LTV range: ${y_train_payers.min():.2f} - ${y_train_payers.max():.2f}")
        print(f"    Train LTV mean:  ${y_train_payers.mean():.2f}")
        
        # XGBoost Regressor
        self.stage2_model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            random_state=self.config['modeling']['random_seed'],
            verbosity=0
        )
        
        # Train with early stopping
        self.stage2_model.set_params(early_stopping_rounds=20)
        self.stage2_model.fit(
            X_train_payers, y_train_payers,
            eval_set=[(X_val_payers, y_val_payers)],
            verbose=False
        )
        
        # Predict
        y_train_pred = self.stage2_model.predict(X_train_payers)
        y_val_pred = self.stage2_model.predict(X_val_payers)
        
        # Ensure non-negative
        y_train_pred = np.maximum(y_train_pred, 0)
        y_val_pred = np.maximum(y_val_pred, 0)
        
        # Metrics
        r2_train = r2_score(y_train_payers, y_train_pred)
        r2_val = r2_score(y_val_payers, y_val_pred)
        
        # MAPE (avoid division by zero)
        mape_train = mean_absolute_percentage_error(
            y_train_payers[y_train_payers > 0.01], 
            y_train_pred[y_train_payers > 0.01]
        )
        mape_val = mean_absolute_percentage_error(
            y_val_payers[y_val_payers > 0.01],
            y_val_pred[y_val_payers > 0.01]
        )
        
        print(f"\n  ✓ Stage 2 Performance (Payers Only):")
        print(f"    Train R²:   {r2_train:.4f}  |  MAPE: {mape_train:.4f}")
        print(f"    Val R²:     {r2_val:.4f}  |  MAPE: {mape_val:.4f}")
        
        # Feature importance (top 10)
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.stage2_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n  Top 5 features for regression:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"    {row['feature']:30s}: {row['importance']:.4f}")
        
        return y_train_pred, y_val_pred
    
    def predict(self, X):
        """Two-stage prediction: P(payer) × E[LTV|payer]"""
        # Stage 1: Probability of being payer
        p_payer = self.stage1_model.predict_proba(X)[:, 1]
        
        # Stage 2: Expected LTV given payer
        ltv_given_payer = self.stage2_model.predict(X)
        ltv_given_payer = np.maximum(ltv_given_payer, 0)  # No negative LTV
        
        # Final prediction: Hurdle model
        ltv_pred = p_payer * ltv_given_payer
        
        return ltv_pred, p_payer, ltv_given_payer
    
    def evaluate(self, X_val, y_val):
        """Evaluate full two-stage model"""
        print("\n" + "─"*70)
        print("[EVALUATION] Two-Stage Hurdle Model")
        print("─"*70)
        
        # Predict
        ltv_pred, p_payer, ltv_given_payer = self.predict(X_val)
        
        # Overall MAPE (avoid near-zero values)
        valid_mask = y_val > 0.01
        if valid_mask.sum() > 0:
            mape = mean_absolute_percentage_error(y_val[valid_mask], ltv_pred[valid_mask])
        else:
            mape = float('inf')
        
        # R²
        r2 = r2_score(y_val, ltv_pred)
        
        # MAE
        mae = np.mean(np.abs(y_val - ltv_pred))
        
        print(f"\n  ✓ Overall Metrics:")
        print(f"    MAPE: {mape:.4f}")
        print(f"    R²:   {r2:.4f}")
        print(f"    MAE:  ${mae:.2f}")
        
        # By payer status
        is_payer = y_val > 0
        
        print(f"\n  Breakdown by actual payer status:")
        
        if is_payer.sum() > 0:
            payers_actual = y_val[is_payer]
            payers_pred = ltv_pred[is_payer]
            
            payers_valid = payers_actual > 0.01
            if payers_valid.sum() > 0:
                mape_payers = mean_absolute_percentage_error(
                    payers_actual[payers_valid], 
                    payers_pred[payers_valid]
                )
                r2_payers = r2_score(payers_actual, payers_pred)
                
                print(f"    Payers ({is_payer.sum():,}):")
                print(f"      MAPE: {mape_payers:.4f}")
                print(f"      R²:   {r2_payers:.4f}")
        
        if (~is_payer).sum() > 0:
            mae_non_payers = np.mean(np.abs(y_val[~is_payer] - ltv_pred[~is_payer]))
            print(f"    Non-payers ({(~is_payer).sum():,}):")
            print(f"      MAE:  ${mae_non_payers:.4f}")
        
        print(f"\n  Average P(payer) predicted: {p_payer.mean():.4f}")
        print(f"  Average E[LTV|payer]: ${ltv_given_payer.mean():.2f}")
        
        return {
            'mape': mape,
            'r2': r2,
            'mae': mae,
            'predictions': ltv_pred,
            'p_payer': p_payer,
            'ltv_given_payer': ltv_given_payer
        }
    
    def save_models(self, tier, models_path='models'):
        """Save both stage models"""
        tier_path = Path(models_path) / tier
        tier_path.mkdir(parents=True, exist_ok=True)
        
        # Save models
        stage1_path = tier_path / 'hurdle_stage1_classifier.pkl'
        stage2_path = tier_path / 'hurdle_stage2_regressor.pkl'
        
        joblib.dump(self.stage1_model, stage1_path)
        joblib.dump(self.stage2_model, stage2_path)
        
        # Save feature names
        feature_path = tier_path / 'hurdle_features.txt'
        with open(feature_path, 'w') as f:
            f.write('\n'.join(self.feature_names))
        
        print(f"\n  ✓ Saved models:")
        print(f"    - {stage1_path}")
        print(f"    - {stage2_path}")
        print(f"    - {feature_path}")


def train_hurdle_for_tier(tier, df_train, df_val):
    """Train hurdle model for specific tier"""
    print("\n" + "="*70)
    print(f"TRAINING HURDLE MODEL: {tier.upper()}")
    print("="*70)
    
    # Filter data for tier
    df_train_tier = df_train[df_train['tier'] == tier].copy()
    df_val_tier = df_val[df_val['tier'] == tier].copy()
    
    print(f"\nDataset size:")
    print(f"  Train: {len(df_train_tier):,} rows")
    print(f"  Val:   {len(df_val_tier):,} rows")
    
    if len(df_train_tier) < 100:
        print(f"\n⚠ Warning: Insufficient data for {tier}. Skipping...")
        return None, None
    
    # Initialize model
    hurdle = HurdleModel()
    
    # Prepare features
    X_train = hurdle.prepare_features(df_train_tier)
    X_val = hurdle.prepare_features(df_val_tier)
    
    print(f"\nFeatures: {len(hurdle.feature_names)} numeric features")
    
    # Stage 1 targets (binary): is_payer
    y_train_is_payer = (df_train_tier['ltv_d60'] > 0).astype(int)
    y_val_is_payer = (df_val_tier['ltv_d60'] > 0).astype(int)
    
    # Train Stage 1: Payer classifier
    p_train_proba, p_val_proba = hurdle.train_stage1_classifier(
        X_train, y_train_is_payer, X_val, y_val_is_payer
    )
    
    # Stage 2: Filter to payers only
    train_payer_mask = y_train_is_payer == 1
    val_payer_mask = y_val_is_payer == 1
    
    if train_payer_mask.sum() < 50:
        print(f"\n⚠ Warning: Too few payers for Stage 2. Skipping...")
        return None, None
    
    X_train_payers = X_train[train_payer_mask]
    y_train_payers = df_train_tier.loc[train_payer_mask, 'ltv_d60'].values
    
    X_val_payers = X_val[val_payer_mask]
    y_val_payers = df_val_tier.loc[val_payer_mask, 'ltv_d60'].values
    
    # Train Stage 2: Amount regressor
    ltv_train_pred, ltv_val_pred = hurdle.train_stage2_regressor(
        X_train_payers, y_train_payers, X_val_payers, y_val_payers
    )
    
    # Evaluate full model
    results = hurdle.evaluate(X_val, df_val_tier['ltv_d60'].values)
    
    # Save models
    hurdle.save_models(tier)
    
    return hurdle, results


def main():
    """Main execution"""
    print("="*70)
    print("STEP 7: TWO-STAGE HURDLE MODEL")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    df_train = pd.read_csv('data/features/train.csv')
    df_val = pd.read_csv('data/features/validation.csv')
    
    print(f"  Train: {len(df_train):,} rows")
    print(f"  Val:   {len(df_val):,} rows")
    
    # Train for Tier 1 and Tier 2
    results_summary = []
    
    for tier in ['tier1', 'tier2']:
        hurdle, results = train_hurdle_for_tier(tier, df_train, df_val)
        
        if results is not None:
            results_summary.append({
                'tier': tier,
                'mape': results['mape'],
                'r2': results['r2'],
                'mae': results['mae']
            })
    
    # Save summary
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        summary_path = Path('results/step07_hurdle_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print("\n" + summary_df.to_string(index=False))
        print(f"\n✓ Summary saved: {summary_path}")
    
    print("\n" + "="*70)
    print("✅ STEP 7 COMPLETED!")
    print("="*70)
    print("\nOutputs:")
    print("  📁 models/tier1/hurdle_stage1_classifier.pkl")
    print("  📁 models/tier1/hurdle_stage2_regressor.pkl")
    print("  📁 models/tier2/hurdle_stage1_classifier.pkl")
    print("  📁 models/tier2/hurdle_stage2_regressor.pkl")
    print("  📊 results/step07_hurdle_summary.csv")
    print("\n➡️  Next Step: step08_train_curve_fitting.py")


if __name__ == "__main__":
    main()
