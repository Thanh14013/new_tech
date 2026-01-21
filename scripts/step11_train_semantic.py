"""
STEP 11: SEMANTIC SIMILARITY FALLBACK
TF-IDF matching for new campaigns with no history
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import yaml
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score

# Suppress warnings
warnings.filterwarnings('ignore')


class SemanticFallback:
    """TF-IDF based semantic matching for new campaigns"""
    
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.vectorizer = None
        self.campaign_vectors = None
        self.campaign_ltv_map = {}
        self.campaign_names = []
        self.global_mean_ltv = None
    
    def build_vocabulary(self, df_train):
        """Build TF-IDF vocabulary from training campaigns"""
        print("\n🔤 Building TF-IDF Vocabulary...")
        
        # Get unique campaigns with their avg LTV
        campaign_agg = df_train.groupby(['app_id', 'campaign']).agg({
            'ltv_d30': 'mean'
        }).reset_index()
        
        # Campaign identifier: app_id::campaign
        campaign_agg['campaign_id'] = campaign_agg['app_id'] + '::' + campaign_agg['campaign']
        
        # Store campaign names and LTV
        self.campaign_names = campaign_agg['campaign_id'].tolist()
        self.campaign_ltv_map = dict(zip(campaign_agg['campaign_id'], campaign_agg['ltv_d30']))
        self.global_mean_ltv = campaign_agg['ltv_d30'].mean()
        
        # TF-IDF vectorizer
        ngram_min, ngram_max = self.config['semantic_matching']['ngram_range']
        max_features = self.config['semantic_matching']['max_features']
        
        self.vectorizer = TfidfVectorizer(
            ngram_range=(ngram_min, ngram_max),
            max_features=max_features,
            analyzer='char_wb',  # Character n-grams (better for campaign names)
            lowercase=True
        )
        
        # Fit on campaign names only (not full identifier)
        campaign_texts = campaign_agg['campaign'].tolist()
        self.campaign_vectors = self.vectorizer.fit_transform(campaign_texts)
        
        print(f"   ✅ Vocabulary size: {len(self.vectorizer.vocabulary_):,}")
        print(f"   ✅ Campaigns indexed: {len(self.campaign_names):,}")
        print(f"   ✅ Global mean LTV: ${self.global_mean_ltv:.2f}")
    
    def find_similar_campaigns(self, query_campaign, app_id=None, top_k=3, similarity_threshold=0.6):
        """Find top-k most similar campaigns"""
        
        # Vectorize query
        query_vec = self.vectorizer.transform([query_campaign])
        
        # Cosine similarity
        similarities = cosine_similarity(query_vec, self.campaign_vectors).flatten()
        
        # Top-k indices (sorted by similarity descending)
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Filter by threshold
        results = []
        for idx in top_k_indices:
            if similarities[idx] >= similarity_threshold:
                campaign_id = self.campaign_names[idx]
                results.append({
                    'campaign_id': campaign_id,
                    'similarity': similarities[idx],
                    'ltv': self.campaign_ltv_map[campaign_id]
                })
        
        return results
    
    def predict_ltv(self, app_id, campaign, top_k=3):
        """Predict LTV using semantic matching"""
        
        # Find similar campaigns
        similar = self.find_similar_campaigns(campaign, app_id, top_k=top_k)
        
        if not similar:
            # No match found, use global mean
            return self.global_mean_ltv, 0.0  # confidence = 0
        
        # Weighted average (by similarity)
        weights = np.array([s['similarity'] for s in similar])
        weights = weights / weights.sum()  # Normalize
        
        ltv_values = np.array([s['ltv'] for s in similar])
        ltv_pred = np.sum(weights * ltv_values)
        
        # Confidence: avg similarity of matched campaigns
        confidence = np.mean([s['similarity'] for s in similar])
        
        return ltv_pred, confidence
    
    def identify_new_campaigns(self, df_test, df_train):
        """Identify campaigns in test that are NOT in train"""
        print("\n🔍 Identifying New Campaigns...")
        
        # Training combos
        train_combos = set(zip(df_train['app_id'], df_train['campaign']))
        
        # Test combos
        test_combos = set(zip(df_test['app_id'], df_test['campaign']))
        
        # New combos
        new_combos = test_combos - train_combos
        
        print(f"   - Train combos: {len(train_combos):,}")
        print(f"   - Test combos: {len(test_combos):,}")
        print(f"   - New combos: {len(new_combos):,} ({len(new_combos)/len(test_combos)*100:.1f}%)")
        
        return new_combos
    
    def evaluate_on_new_campaigns(self, df_test, df_train):
        """Evaluate semantic matching on new campaigns only"""
        print("\n📊 Evaluating Semantic Fallback on New Campaigns...")
        
        # Find new campaigns
        new_combos = self.identify_new_campaigns(df_test, df_train)
        
        if len(new_combos) == 0:
            print("   ⚠ No new campaigns found in test set")
            return {
                'mape': None, 
                'mae': None,
                'r2': None,
                'coverage': 0,
                'n_new_campaigns': 0,
                'n_matched': 0
            }
        
        # Filter test data to new campaigns only
        df_new = df_test[
            df_test.apply(lambda row: (row['app_id'], row['campaign']) in new_combos, axis=1)
        ].copy()
        
        print(f"   - Evaluating on {len(df_new):,} rows from new campaigns")
        
        # Aggregate by campaign
        campaign_new = df_new.groupby(['app_id', 'campaign']).agg({
            'ltv_d30': 'mean'
        }).reset_index()
        
        print(f"   - {len(campaign_new):,} unique new campaigns")
        
        # Predict for each new campaign
        predictions = []
        confidences = []
        coverage_count = 0
        
        similarity_threshold = self.config['semantic_matching']['similarity_threshold']
        
        for idx, row in campaign_new.iterrows():
            ltv_pred, confidence = self.predict_ltv(row['app_id'], row['campaign'], top_k=3)
            
            if confidence >= similarity_threshold:
                coverage_count += 1
            
            predictions.append(ltv_pred)
            confidences.append(confidence)
        
        campaign_new['ltv_pred'] = predictions
        campaign_new['confidence'] = confidences
        
        # Metrics
        y_true = campaign_new['ltv_d30'].values
        y_pred = campaign_new['ltv_pred'].values
        
        # Remove any potential inf/nan
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        # MAPE: handle zero values
        mape_mask = y_true != 0
        if mape_mask.sum() > 0:
            mape = mean_absolute_percentage_error(y_true[mape_mask], y_pred[mape_mask])
        else:
            mape = np.nan
        
        coverage = coverage_count / len(campaign_new) * 100
        
        print(f"\n   📈 Overall Metrics:")
        print(f"      R²:   {r2:.4f}")
        print(f"      MAE:  ${mae:.2f}")
        print(f"      MAPE: {mape*100:.2f}%")
        print(f"      Samples: {len(campaign_new):,}")
        print(f"\n   🎯 Coverage:")
        print(f"      Matched (similarity ≥ {similarity_threshold}): {coverage:.1f}%")
        print(f"      Campaigns: {coverage_count}/{len(campaign_new)}")
        
        # By confidence level
        print(f"\n   📊 By Confidence Level:")
        for conf_min, conf_max, label in [(0.6, 0.7, 'Low'), (0.7, 0.85, 'Medium'), (0.85, 1.0, 'High')]:
            mask_conf = (campaign_new['confidence'] >= conf_min) & (campaign_new['confidence'] < conf_max)
            if mask_conf.sum() > 0:
                y_true_conf = campaign_new.loc[mask_conf, 'ltv_d30'].values
                y_pred_conf = campaign_new.loc[mask_conf, 'ltv_pred'].values
                
                mask_valid = np.isfinite(y_true_conf) & np.isfinite(y_pred_conf)
                if mask_valid.sum() > 0:
                    r2_conf = r2_score(y_true_conf[mask_valid], y_pred_conf[mask_valid])
                    mae_conf = mean_absolute_error(y_true_conf[mask_valid], y_pred_conf[mask_valid])
                    print(f"      {label} ({conf_min}-{conf_max}): {mask_conf.sum():,} campaigns, R²={r2_conf:.4f}, MAE=${mae_conf:.2f}")
        
        return {
            'mape': mape,
            'mae': mae,
            'r2': r2,
            'coverage': coverage,
            'n_new_campaigns': len(campaign_new),
            'n_matched': coverage_count,
            'avg_confidence': np.mean(confidences)
        }
    
    def save_models(self, models_path='models/semantic'):
        """Save semantic models"""
        models_path = Path(models_path)
        models_path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.vectorizer, models_path / 'tfidf_vectorizer.pkl')
        joblib.dump(self.campaign_vectors, models_path / 'campaign_vectors.pkl')
        joblib.dump(self.campaign_ltv_map, models_path / 'campaign_ltv_map.pkl')
        joblib.dump(self.campaign_names, models_path / 'campaign_names.pkl')
        joblib.dump(self.global_mean_ltv, models_path / 'global_mean_ltv.pkl')
        
        print(f"\n   ✅ Models saved to: {models_path}")


def main():
    """Main execution"""
    print("="*80)
    print("🚀 STEP 11: SEMANTIC SIMILARITY FALLBACK")
    print("="*80)
    print()
    print("Goal: TF-IDF matching for new campaigns with no training history")
    print("      Use case: Cold-start campaigns using campaign name similarity")
    print()
    
    # Load data
    print("📂 Loading data...")
    
    # Try enhanced features first
    train_path = Path('data/features/train_enhanced.csv')
    test_path = Path('data/features/validation_enhanced.csv')
    
    if not train_path.exists():
        train_path = Path('data/features/train.csv')
        test_path = Path('data/features/validation.csv')
        print("   ⚠ Enhanced features not found, using regular features")
    else:
        print("   ✅ Using enhanced features")
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    print(f"   Training: {len(df_train):,} rows")
    print(f"   Test: {len(df_test):,} rows")
    
    # Initialize
    semantic = SemanticFallback()
    
    # Build vocabulary
    semantic.build_vocabulary(df_train)
    
    # Evaluate on new campaigns
    results = semantic.evaluate_on_new_campaigns(df_test, df_train)
    
    # Save models
    semantic.save_models()
    
    # Save summary
    results_path = Path('results')
    results_path.mkdir(exist_ok=True)
    
    summary_df = pd.DataFrame([{
        'method': 'semantic_fallback',
        'n_new_campaigns': results.get('n_new_campaigns', 0),
        'n_matched': results.get('n_matched', 0),
        'coverage_pct': results.get('coverage', 0),
        'r2': results.get('r2', None),
        'mae': results.get('mae', None),
        'mape': results.get('mape', None),
        'avg_confidence': results.get('avg_confidence', None)
    }])
    summary_df.to_csv('results/step11_semantic_summary.csv', index=False)
    
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("✅ STEP 11 COMPLETED!")
    print("="*80)
    print()
    print("✅ Models saved:")
    print("   - models/semantic/tfidf_vectorizer.pkl")
    print("   - models/semantic/campaign_vectors.pkl")
    print("   - models/semantic/campaign_ltv_map.pkl")
    print()
    print("✅ Summary saved: results/step11_semantic_summary.csv")
    print()
    print("➡️  Next Step: Step 12 - Calibration & Optimization")


if __name__ == "__main__":
    main()
