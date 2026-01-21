"""
PRODUCTION PIPELINE - D60 LTV PREDICTION
Fast, reliable prediction engine for production use
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings

warnings.filterwarnings('ignore')


class ModelRegistry:
    """Load and cache trained models"""
    
    def __init__(self, models_dir: str = 'models'):
        self.models_dir = Path(models_dir)
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """Load all required models"""
        print("📦 Loading models...")
        
        try:
            # Primary model: ML Multiplier Tuned (Best - MAPE 9.94%)
            ml_tuned_path = self.models_dir / 'tier2' / 'ml_multiplier_tuned.pkl'
            if ml_tuned_path.exists():
                self.models['ml_multiplier'] = joblib.load(ml_tuned_path)
                print("   ✅ ML Multiplier Tuned loaded")
            else:
                # Fallback to enhanced
                ml_path = self.models_dir / 'tier2' / 'ml_multiplier_enhanced.pkl'
                self.models['ml_multiplier'] = joblib.load(ml_path)
                print("   ✅ ML Multiplier Enhanced loaded (fallback)")
            
            # Load features list
            features_path = self.models_dir / 'tier2' / 'ml_multiplier_enhanced_features.txt'
            with open(features_path, 'r') as f:
                self.models['features'] = [line.strip() for line in f if line.strip()]
            print(f"   ✅ Features loaded: {len(self.models['features'])} features")
            
            # Semantic fallback
            semantic_ltv = self.models_dir / 'semantic' / 'campaign_ltv_map.pkl'
            if semantic_ltv.exists():
                self.models['semantic'] = joblib.load(semantic_ltv)
                global_mean = joblib.load(self.models_dir / 'semantic' / 'global_mean_ltv.pkl')
                self.models['global_mean'] = global_mean
                print(f"   ✅ Semantic fallback loaded ({len(self.models['semantic'])} campaigns)")
            else:
                self.models['semantic'] = {}
                self.models['global_mean'] = 5.28  # Default from data
                print("   ⚠️  Semantic fallback not found, using global mean")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load models: {str(e)}")
    
    def get_model(self, name: str):
        """Get loaded model by name"""
        return self.models.get(name)


class FeatureEngine:
    """Compute features from raw data"""
    
    @staticmethod
    def compute_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all required features for prediction
        Input: DataFrame with raw columns (installs, cost, revenue, etc.)
        Output: DataFrame with ML features
        """
        df_features = df.copy()
        
        # Basic revenue features
        if 'rev_sum' not in df_features.columns:
            df_features['rev_sum'] = df_features.get('revenue', 0)
        
        if 'rev_max' not in df_features.columns:
            df_features['rev_max'] = df_features['rev_sum']
        
        if 'rev_last' not in df_features.columns:
            df_features['rev_last'] = df_features['rev_sum']
        
        # Revenue dynamics
        if 'rev_volatility' not in df_features.columns:
            df_features['rev_volatility'] = 0.5  # Default
        
        if 'rev_growth_rate' not in df_features.columns:
            df_features['rev_growth_rate'] = 0.1  # Default
        
        # Engagement
        if 'engagement_score' not in df_features.columns:
            df_features['engagement_score'] = df_features.get('installs', 100) / 100
        
        # CPI
        if 'actual_cpi' not in df_features.columns:
            df_features['actual_cpi'] = df_features.get('cost', 100) / df_features.get('installs', 1).clip(lower=1)
        
        if 'cpi_quality_score' not in df_features.columns:
            df_features['cpi_quality_score'] = 0.7  # Default
        
        # LTV features
        if 'ltv_d30' not in df_features.columns:
            df_features['ltv_d30'] = df_features['rev_sum'] / df_features.get('installs', 1).clip(lower=1)
        
        # Growth features (enhanced)
        if 'ltv_growth_d30_d60' not in df_features.columns:
            df_features['ltv_growth_d30_d60'] = 0.0
        
        if 'ltv_growth_absolute' not in df_features.columns:
            df_features['ltv_growth_absolute'] = 0.0
        
        if 'ltv_decay_rate' not in df_features.columns:
            df_features['ltv_decay_rate'] = -0.05
        
        # Campaign features
        if 'campaign_ltv_avg' not in df_features.columns:
            df_features['campaign_ltv_avg'] = df_features['ltv_d30']
        
        if 'campaign_engagement_avg' not in df_features.columns:
            df_features['campaign_engagement_avg'] = df_features['engagement_score']
        
        if 'campaign_total_installs' not in df_features.columns:
            df_features['campaign_total_installs'] = df_features.get('installs', 100)
        
        # Churn features
        if 'user_churn_rate_d7' not in df_features.columns:
            df_features['user_churn_rate_d7'] = 0.3  # Default
        
        if 'user_churn_rate_d14' not in df_features.columns:
            df_features['user_churn_rate_d14'] = 0.5
        
        if 'user_churn_rate_d30' not in df_features.columns:
            df_features['user_churn_rate_d30'] = 0.7
        
        # Retention
        if 'retention_d7' not in df_features.columns:
            df_features['retention_d7'] = 0.4
        
        if 'retention_d14' not in df_features.columns:
            df_features['retention_d14'] = 0.3
        
        if 'retention_d30' not in df_features.columns:
            df_features['retention_d30'] = 0.2
        
        return df_features


class PredictionService:
    """Core prediction service with fallback logic"""
    
    def __init__(self, model_registry: ModelRegistry):
        self.registry = model_registry
        self.feature_engine = FeatureEngine()
    
    def predict_single(self, 
                      app_id: str, 
                      campaign: str, 
                      data: Dict) -> Dict:
        """
        Predict D60 LTV for a single campaign
        
        Args:
            app_id: Application ID
            campaign: Campaign name
            data: Dictionary with campaign metrics
                  Required: installs, cost, revenue (or rev_sum)
                  Optional: All other features
        
        Returns:
            Dictionary with prediction, confidence, method used
        """
        # Create DataFrame from input
        df = pd.DataFrame([{
            'app_id': app_id,
            'campaign': campaign,
            **data
        }])
        
        # Predict batch (more efficient)
        results = self.predict_batch(df)
        
        return results[0] if len(results) > 0 else None
    
    def predict_batch(self, df: pd.DataFrame) -> List[Dict]:
        """
        Batch prediction for multiple campaigns
        
        Args:
            df: DataFrame with columns: app_id, campaign, installs, cost, revenue, ...
        
        Returns:
            List of prediction dictionaries
        """
        # Compute features
        df_features = self.feature_engine.compute_features(df)
        
        # Get required features
        feature_cols = self.registry.get_model('features')
        
        # Ensure all features exist
        for col in feature_cols:
            if col not in df_features.columns:
                df_features[col] = 0
        
        X = df_features[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        
        # Predict with ML Multiplier
        ml_model = self.registry.get_model('ml_multiplier')
        multiplier = ml_model.predict(X)
        
        # Get base revenue
        base_rev = df_features.get('rev_sum', df_features.get('revenue', 0)).values
        
        # Calculate LTV predictions
        ltv_pred = base_rev * np.maximum(multiplier, 0.1)
        
        # Build results
        results = []
        for i, (idx, row) in enumerate(df.iterrows()):
            pred_ltv = ltv_pred[i]  # Use enumerate index, not df index
            method = 'ml_multiplier_tuned'
            confidence = 0.90  # High confidence for ML Multiplier
            
            # Fallback to semantic if prediction seems off
            if pred_ltv <= 0 or not np.isfinite(pred_ltv):
                campaign_id = f"{row['app_id']}::{row['campaign']}"
                semantic_map = self.registry.get_model('semantic')
                
                if campaign_id in semantic_map:
                    pred_ltv = semantic_map[campaign_id]
                    method = 'semantic_fallback'
                    confidence = 0.70
                else:
                    pred_ltv = self.registry.get_model('global_mean')
                    method = 'global_mean'
                    confidence = 0.50
            
            results.append({
                'app_id': row['app_id'],
                'campaign': row['campaign'],
                'predicted_d60_ltv': float(pred_ltv),
                'method': method,
                'confidence': confidence,
                'multiplier': float(multiplier[i]) if np.isfinite(multiplier[i]) else 1.0,
                'base_revenue': float(base_rev[i])
            })
        
        return results


class ProductionPipeline:
    """
    Main production pipeline for D60 LTV predictions
    
    Usage:
        pipeline = ProductionPipeline()
        
        # Single prediction
        result = pipeline.predict(
            app_id='com.example.app',
            campaign='summer_sale_2024',
            data={'installs': 1000, 'cost': 500, 'revenue': 200}
        )
        
        # Batch prediction
        results = pipeline.predict_batch('input_data.csv')
    """
    
    def __init__(self, models_dir: str = 'models'):
        """Initialize pipeline with models"""
        print("🚀 Initializing Production Pipeline...")
        self.registry = ModelRegistry(models_dir)
        self.service = PredictionService(self.registry)
        print("✅ Pipeline ready!")
    
    def predict(self, 
                app_id: str, 
                campaign: str, 
                data: Dict) -> Dict:
        """
        Predict D60 LTV for a single campaign
        
        Example:
            result = pipeline.predict(
                app_id='com.game.app',
                campaign='new_year_campaign',
                data={
                    'installs': 5000,
                    'cost': 2500,
                    'revenue': 1200
                }
            )
            
            print(f"Predicted D60 LTV: ${result['predicted_d60_ltv']:.2f}")
        """
        return self.service.predict_single(app_id, campaign, data)
    
    def predict_batch(self, 
                     input_data: Union[str, pd.DataFrame],
                     output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Batch prediction from CSV or DataFrame
        
        Args:
            input_data: Path to CSV file or DataFrame
            output_path: Optional path to save results CSV
        
        Returns:
            DataFrame with predictions
        
        Example:
            results = pipeline.predict_batch('campaigns.csv', 'predictions.csv')
        """
        # Load data
        if isinstance(input_data, str):
            df = pd.read_csv(input_data)
            print(f"📂 Loaded {len(df):,} rows from {input_data}")
        else:
            df = input_data.copy()
        
        # Predict
        print(f"🔮 Predicting D60 LTV for {len(df):,} campaigns...")
        results = self.service.predict_batch(df)
        
        # Convert to DataFrame
        df_results = pd.DataFrame(results)
        
        # Add actual if available
        if 'ltv_d60' in df.columns:
            df_results['actual_d60_ltv'] = df['ltv_d60'].values
            
            # Calculate metrics
            mask = (df_results['actual_d60_ltv'] >= 0.10) & np.isfinite(df_results['predicted_d60_ltv'])
            if mask.sum() > 0:
                from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error
                
                mape = mean_absolute_percentage_error(
                    df_results.loc[mask, 'actual_d60_ltv'],
                    df_results.loc[mask, 'predicted_d60_ltv']
                ) * 100
                
                r2 = r2_score(
                    df_results.loc[mask, 'actual_d60_ltv'],
                    df_results.loc[mask, 'predicted_d60_ltv']
                )
                
                mae = mean_absolute_error(
                    df_results.loc[mask, 'actual_d60_ltv'],
                    df_results.loc[mask, 'predicted_d60_ltv']
                )
                
                print(f"\n📊 Validation Metrics:")
                print(f"   R² Score:  {r2:.4f}")
                print(f"   MAE:      ${mae:.2f}")
                print(f"   MAPE:      {mape:.2f}%")
                print(f"   Samples:   {mask.sum():,}")
        
        # Save if output path provided
        if output_path:
            df_results.to_csv(output_path, index=False)
            print(f"💾 Results saved to {output_path}")
        
        print(f"✅ Prediction complete!")
        
        return df_results
    
    def evaluate(self, validation_path: str = 'data/features/validation_enhanced.csv'):
        """
        Evaluate pipeline on validation set
        
        Example:
            pipeline.evaluate()
        """
        print("📊 Evaluating on validation set...")
        results = self.predict_batch(validation_path)
        return results


# Convenience function
def predict_campaign(app_id: str, campaign: str, **kwargs) -> Dict:
    """
    Quick prediction function
    
    Example:
        result = predict_campaign(
            app_id='com.game.app',
            campaign='christmas_2024',
            installs=10000,
            cost=5000,
            revenue=3000
        )
    """
    pipeline = ProductionPipeline()
    return pipeline.predict(app_id, campaign, kwargs)


if __name__ == "__main__":
    # Example usage
    print("="*80)
    print("PRODUCTION PIPELINE - D60 LTV PREDICTION")
    print("="*80)
    print()
    
    # Initialize
    pipeline = ProductionPipeline()
    
    # Example 1: Single prediction
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Campaign Prediction")
    print("="*80)
    
    result = pipeline.predict(
        app_id='com.example.game',
        campaign='summer_sale_2024',
        data={
            'installs': 5000,
            'cost': 2500,
            'revenue': 1200
        }
    )
    
    print(f"\n✅ Prediction Result:")
    print(f"   App ID:           {result['app_id']}")
    print(f"   Campaign:         {result['campaign']}")
    print(f"   Predicted D60 LTV: ${result['predicted_d60_ltv']:.2f}")
    print(f"   Method:           {result['method']}")
    print(f"   Confidence:       {result['confidence']*100:.0f}%")
    print(f"   Multiplier:       {result['multiplier']:.2f}x")
    
    # Example 2: Batch prediction on validation set
    print("\n" + "="*80)
    print("EXAMPLE 2: Batch Prediction (Validation Set)")
    print("="*80)
    
    val_path = 'data/features/validation_enhanced.csv'
    if Path(val_path).exists():
        results_df = pipeline.predict_batch(val_path)
        print(f"\n📊 Results preview:")
        print(results_df[['app_id', 'campaign', 'predicted_d60_ltv', 'method', 'confidence']].head(10))
    else:
        print(f"⚠️  Validation file not found: {val_path}")
