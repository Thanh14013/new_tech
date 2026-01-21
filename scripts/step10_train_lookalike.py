"""
STEP 10: LOOKALIKE METHOD
Clustering-based similarity matching for new/sparse campaigns
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error

# Suppress FutureWarnings from pandas downcasting
warnings.filterwarnings('ignore', category=FutureWarning)


class LookalikeModel:
    """Clustering-based look-alike matching for cold-start campaigns"""
    
    def __init__(self, n_clusters=20, random_seed=42):
        self.n_clusters = n_clusters
        self.random_seed = random_seed
        self.scaler = StandardScaler()
        self.kmeans = None
        self.cluster_stats = {}
        self.feature_cols = None
    
    def prepare_campaign_features(self, df):
        """Aggregate features at campaign-geo level for clustering"""
        print("\n📊 Aggregating campaign-geo level features...")
        
        # Check which columns exist
        available_cols = df.columns.tolist()
        
        # Define aggregation dict with only available columns
        agg_dict = {}
        
        # Always available
        if 'installs' in available_cols:
            agg_dict['installs'] = 'sum'
        if 'cost' in available_cols:
            agg_dict['cost'] = 'sum'
        if 'rev_sum' in available_cols:
            agg_dict['rev_sum'] = 'mean'
        if 'rev_max' in available_cols:
            agg_dict['rev_max'] = 'mean'
        if 'rev_last' in available_cols:
            agg_dict['rev_last'] = 'mean'
        if 'rev_volatility' in available_cols:
            agg_dict['rev_volatility'] = 'mean'
        if 'rev_growth_rate' in available_cols:
            agg_dict['rev_growth_rate'] = 'mean'
        if 'engagement_score' in available_cols:
            agg_dict['engagement_score'] = 'mean'
        if 'actual_cpi' in available_cols:
            agg_dict['actual_cpi'] = 'mean'
        if 'cpi_quality_score' in available_cols:
            agg_dict['cpi_quality_score'] = 'mean'
        if 'campaign_ltv_avg' in available_cols:
            agg_dict['campaign_ltv_avg'] = 'first'
        if 'campaign_engagement_avg' in available_cols:
            agg_dict['campaign_engagement_avg'] = 'first'
        if 'campaign_total_installs' in available_cols:
            agg_dict['campaign_total_installs'] = 'first'
        if 'ltv_d60' in available_cols:
            agg_dict['ltv_d60'] = 'mean'
        if 'tier' in available_cols:
            agg_dict['tier'] = 'first'
        
        # Group by campaign + geo
        campaign_agg = df.groupby(['campaign', 'geo']).agg(agg_dict).reset_index()
        
        print(f"   ✅ Aggregated to {len(campaign_agg):,} campaign-geo combinations")
        
        return campaign_agg
    
    def train(self, df_train):
        """Train KMeans clustering on campaign features"""
        print("\n🎯 Training KMeans Clustering...")
        
        # Prepare campaign-level data
        campaign_df = self.prepare_campaign_features(df_train)
        
        # Feature columns for clustering
        self.feature_cols = [
            'installs',
            'cost',
            'rev_sum',
            'rev_max',
            'rev_last',
            'rev_volatility',
            'rev_growth_rate',
            'engagement_score',
            'actual_cpi',
            'cpi_quality_score',
            'campaign_ltv_avg',
            'campaign_engagement_avg',
            'campaign_total_installs'
        ]
        
        X = campaign_df[self.feature_cols].copy()
        X = X.fillna(0).infer_objects(copy=False).replace([np.inf, -np.inf], 0)
        
        print(f"   Features for clustering: {len(self.feature_cols)}")
        print(f"   Training samples: {len(X):,}")
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train KMeans
        print(f"\n   Training KMeans with {self.n_clusters} clusters...")
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_seed,
            n_init=10,
            max_iter=300
        )
        
        campaign_df['cluster'] = self.kmeans.fit_predict(X_scaled)
        
        print(f"   ✅ KMeans fitted successfully")
        
        # Calculate cluster statistics
        print(f"\n📊 Cluster Statistics:")
        print(f"   {'Cluster':<10} {'Count':<10} {'Avg LTV':<12} {'Median LTV':<12} {'Std LTV':<12}")
        print(f"   {'-'*60}")
        
        for cluster_id in range(self.n_clusters):
            cluster_data = campaign_df[campaign_df['cluster'] == cluster_id]
            
            if len(cluster_data) > 0:
                self.cluster_stats[cluster_id] = {
                    'count': len(cluster_data),
                    'mean_ltv': cluster_data['ltv_d60'].mean(),
                    'median_ltv': cluster_data['ltv_d60'].median(),
                    'std_ltv': cluster_data['ltv_d60'].std(),
                    'tier_distribution': cluster_data['tier'].value_counts().to_dict()
                }
                
                stats = self.cluster_stats[cluster_id]
                print(f"   {cluster_id:<10} {stats['count']:<10,} ${stats['mean_ltv']:<11.2f} "
                      f"${stats['median_ltv']:<11.2f} ${stats['std_ltv']:<11.2f}")
        
        # Use mean as fallback prediction
        self.global_mean_ltv = campaign_df['ltv_d60'].mean()
        print(f"\n   Global mean LTV: ${self.global_mean_ltv:.2f} (used as fallback)")
        
        return campaign_df
    
    def predict_campaign(self, campaign_features):
        """Predict LTV for a single campaign using lookalike matching
        
        Args:
            campaign_features: DataFrame row or dict with campaign features
            
        Returns:
            ltv_pred: Predicted LTV
            cluster_id: Assigned cluster
            confidence: Confidence score (based on cluster size and std)
        """
        if isinstance(campaign_features, dict):
            campaign_features = pd.DataFrame([campaign_features])
        elif isinstance(campaign_features, pd.Series):
            campaign_features = campaign_features.to_frame().T
        
        # Extract features
        X = campaign_features[self.feature_cols].copy()
        X = X.replace([np.inf, -np.inf], 0).fillna(0)
        X = X.infer_objects(copy=False)
        
        # Standardize
        X_scaled = self.scaler.transform(X)
        
        # Find closest cluster
        cluster_id = self.kmeans.predict(X_scaled)[0]
        
        # Get cluster stats
        if cluster_id in self.cluster_stats:
            stats = self.cluster_stats[cluster_id]
            
            # Use mean as prediction
            ltv_pred = stats['mean_ltv']
            
            # Confidence based on cluster size and stability
            # Larger clusters with lower std = higher confidence
            confidence = min(1.0, stats['count'] / 100) * (1 - min(1.0, stats['std_ltv'] / stats['mean_ltv']))
        else:
            # Fallback to global mean
            ltv_pred = self.global_mean_ltv
            confidence = 0.1
        
        return ltv_pred, cluster_id, confidence
    
    def predict_batch(self, df):
        """Predict LTV for multiple campaigns"""
        campaign_df = self.prepare_campaign_features(df)
        
        predictions = []
        clusters = []
        confidences = []
        
        for idx, row in campaign_df.iterrows():
            ltv_pred, cluster_id, confidence = self.predict_campaign(row)
            predictions.append(ltv_pred)
            clusters.append(cluster_id)
            confidences.append(confidence)
        
        campaign_df['ltv_pred'] = predictions
        campaign_df['cluster'] = clusters
        campaign_df['confidence'] = confidences
        
        return campaign_df
    
    def evaluate(self, df_val):
        """Evaluate lookalike model on validation set"""
        print("\n📈 Evaluating Lookalike Model...")
        
        # Predict
        results_df = self.predict_batch(df_val)
        
        # Calculate metrics
        ltv_actual = results_df['ltv_d60'].values
        ltv_pred = results_df['ltv_pred'].values
        
        # Overall metrics
        mask = ltv_actual > 0.01
        r2 = r2_score(ltv_actual, ltv_pred)
        mae = mean_absolute_error(ltv_actual, ltv_pred)
        mape = mean_absolute_percentage_error(ltv_actual[mask], ltv_pred[mask]) if mask.sum() > 0 else np.nan
        
        print(f"\n   Overall Metrics:")
        print(f"      R²:   {r2:.4f}")
        print(f"      MAE:  ${mae:.2f}")
        print(f"      MAPE: {mape:.2%}")
        print(f"      Samples: {len(results_df):,}")
        
        # By confidence level
        print(f"\n   By Confidence Level:")
        print(f"      {'Level':<15} {'Count':<10} {'R²':<10} {'MAE':<12}")
        print(f"      {'-'*50}")
        
        confidence_levels = [
            ('Low (0-0.3)', (results_df['confidence'] < 0.3)),
            ('Medium (0.3-0.6)', (results_df['confidence'] >= 0.3) & (results_df['confidence'] < 0.6)),
            ('High (0.6+)', (results_df['confidence'] >= 0.6))
        ]
        
        for level_name, mask_level in confidence_levels:
            if mask_level.sum() > 0:
                seg_actual = ltv_actual[mask_level]
                seg_pred = ltv_pred[mask_level]
                
                seg_r2 = r2_score(seg_actual, seg_pred)
                seg_mae = mean_absolute_error(seg_actual, seg_pred)
                
                print(f"      {level_name:<15} {mask_level.sum():<10,} {seg_r2:<10.4f} ${seg_mae:<11.2f}")
        
        return {
            'r2': r2,
            'mae': mae,
            'mape': mape,
            'samples': len(results_df),
            'results_df': results_df
        }
    
    def save_models(self, models_path='models/fallback'):
        """Save lookalike models"""
        models_path = Path(models_path)
        models_path.mkdir(parents=True, exist_ok=True)
        
        # Save models
        joblib.dump(self.scaler, models_path / 'lookalike_scaler.pkl')
        joblib.dump(self.kmeans, models_path / 'lookalike_kmeans.pkl')
        joblib.dump(self.cluster_stats, models_path / 'lookalike_cluster_stats.pkl')
        joblib.dump(self.feature_cols, models_path / 'lookalike_features.pkl')
        joblib.dump(self.global_mean_ltv, models_path / 'lookalike_global_mean.pkl')
        
        print(f"\n   ✅ Models saved to: {models_path}")


def main():
    """Main execution"""
    print("="*80)
    print("🚀 STEP 10: LOOKALIKE METHOD")
    print("="*80)
    print("\nGoal: Clustering-based similarity matching for cold-start campaigns")
    print("      Use case: New campaigns with no historical data")
    
    # Load data - use enhanced features if available
    print("\n📂 Loading data...")
    
    # Try enhanced first, fallback to regular
    try:
        df_train = pd.read_csv('data/features/train_enhanced.csv')
        df_val = pd.read_csv('data/features/validation_enhanced.csv')
        print("   ✅ Using enhanced features")
    except:
        df_train = pd.read_csv('data/features/train.csv')
        df_val = pd.read_csv('data/features/validation.csv')
        print("   ✅ Using regular features")
    
    print(f"   Training: {len(df_train):,} rows")
    print(f"   Validation: {len(df_val):,} rows")
    
    # Initialize lookalike model
    lookalike = LookalikeModel(n_clusters=20, random_seed=42)
    
    # Train
    campaign_df = lookalike.train(df_train)
    
    # Evaluate
    results = lookalike.evaluate(df_val)
    
    # Save models
    lookalike.save_models()
    
    # Save summary
    summary_df = pd.DataFrame([{
        'method': 'lookalike_kmeans',
        'n_clusters': lookalike.n_clusters,
        'r2': results['r2'],
        'mae': results['mae'],
        'mape': results['mape'],
        'samples': results['samples']
    }])
    
    summary_path = Path('results/step10_lookalike_summary.csv')
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("✅ STEP 10 COMPLETED!")
    print("="*80)
    print(f"\n✅ Models saved:")
    print(f"   - models/fallback/lookalike_scaler.pkl")
    print(f"   - models/fallback/lookalike_kmeans.pkl")
    print(f"   - models/fallback/lookalike_cluster_stats.pkl")
    print(f"\n✅ Summary saved: {summary_path}")
    print("\n➡️  Next Step: Step 11 - Semantic Fallback")


if __name__ == "__main__":
    main()
