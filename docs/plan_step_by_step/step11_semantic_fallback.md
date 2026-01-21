# Step 11: Semantic Similarity Fallback
## TF-IDF Matching cho New Campaigns (V2.1 Enhancement #2)

**Thời gian:** 0.5 ngày  
**Độ khó:** ⭐⭐⭐ Khó  
**Prerequisites:** Step 10 completed  

---

## 🎯 MỤC TIÊU

Semantic matching cho **754 new app+campaign combos** không có training history:

**Algorithm:**
1. TF-IDF vectorization của campaign names
2. Cosine similarity với existing campaigns
3. Tìm top-3 most similar campaigns (similarity ≥ 0.6)
4. LTV_pred = weighted average của top-3

**Expected:**  
- Coverage: 90% → 98%+
- MAPE: 15-20% → 6-8% (cho new campaigns)

---

## 📥 INPUT

- `data/features/train.csv`
- `data/features/test.csv` (contains new campaigns)
- `config/config.yaml`

---

## 📤 OUTPUT

- `models/semantic/tfidf_vectorizer.pkl`
- `models/semantic/campaign_vectors.pkl`
- `models/semantic/campaign_ltv_map.pkl`
- `results/step11_semantic_evaluation.html`

---

## 🔧 IMPLEMENTATION

### File: `scripts/step11_semantic_fallback.py`

```python
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_percentage_error

class SemanticFallback:
    """TF-IDF based semantic matching for new campaigns"""
    
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.vectorizer = None
        self.campaign_vectors = None
        self.campaign_ltv_map = {}
        self.campaign_names = []
    
    def build_vocabulary(self, df_train):
        """Build TF-IDF vocabulary from training campaigns"""
        print("\n[Vocabulary] Building TF-IDF from campaign names...")
        
        # Get unique campaigns
        campaign_agg = df_train.groupby(['app_id', 'campaign']).agg({
            'ltv_d30': 'mean'
        }).reset_index()
        
        # Campaign identifier
        campaign_agg['campaign_id'] = campaign_agg['app_id'] + '::' + campaign_agg['campaign']
        
        # Store campaign names and LTV
        self.campaign_names = campaign_agg['campaign_id'].tolist()
        self.campaign_ltv_map = dict(zip(campaign_agg['campaign_id'], campaign_agg['ltv_d30']))
        
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
        
        print(f"  ✓ Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        print(f"  ✓ Campaigns indexed: {len(self.campaign_names)}")
    
    def find_similar_campaigns(self, query_campaign, top_k=3, similarity_threshold=0.6):
        """Find top-k most similar campaigns"""
        
        # Vectorize query
        query_vec = self.vectorizer.transform([query_campaign])
        
        # Cosine similarity
        similarities = cosine_similarity(query_vec, self.campaign_vectors).flatten()
        
        # Top-k indices
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Filter by threshold
        results = []
        for idx in top_k_indices:
            if similarities[idx] >= similarity_threshold:
                results.append({
                    'campaign_id': self.campaign_names[idx],
                    'similarity': similarities[idx],
                    'ltv': self.campaign_ltv_map[self.campaign_names[idx]]
                })
        
        return results
    
    def predict_ltv(self, app_id, campaign, fallback_ltv=None):
        """Predict LTV using semantic matching"""
        
        # Find similar campaigns
        similar = self.find_similar_campaigns(campaign, top_k=3)
        
        if not similar:
            # No match found, use fallback
            return fallback_ltv if fallback_ltv is not None else np.mean(list(self.campaign_ltv_map.values()))
        
        # Weighted average (by similarity)
        weights = np.array([s['similarity'] for s in similar])
        weights = weights / weights.sum()  # Normalize
        
        ltv_values = np.array([s['ltv'] for s in similar])
        ltv_pred = np.sum(weights * ltv_values)
        
        return ltv_pred
    
    def identify_new_campaigns(self, df_test, df_train):
        """Identify campaigns in test that are NOT in train"""
        print("\n[New Campaigns] Identifying new campaigns...")
        
        # Training combos
        train_combos = set(zip(df_train['app_id'], df_train['campaign']))
        
        # Test combos
        test_combos = set(zip(df_test['app_id'], df_test['campaign']))
        
        # New combos
        new_combos = test_combos - train_combos
        
        print(f"  - Train combos: {len(train_combos):,}")
        print(f"  - Test combos: {len(test_combos):,}")
        print(f"  - New combos: {len(new_combos):,} ({len(new_combos)/len(test_combos)*100:.1f}%)")
        
        return new_combos
    
    def evaluate_on_new_campaigns(self, df_test, df_train):
        """Evaluate semantic matching on new campaigns only"""
        print("\n[Evaluation] Semantic Fallback on New Campaigns...")
        
        # Find new campaigns
        new_combos = self.identify_new_campaigns(df_test, df_train)
        
        if len(new_combos) == 0:
            print("  ⚠ No new campaigns found in test set")
            return {'mape': None, 'coverage': 0}
        
        # Filter test data to new campaigns only
        df_new = df_test[
            df_test.apply(lambda row: (row['app_id'], row['campaign']) in new_combos, axis=1)
        ].copy()
        
        print(f"  - Evaluating on {len(df_new):,} rows from new campaigns")
        
        # Aggregate by campaign
        campaign_new = df_new.groupby(['app_id', 'campaign']).agg({
            'ltv_d30': 'mean'
        }).reset_index()
        
        # Predict for each new campaign
        predictions = []
        coverage_count = 0
        
        for idx, row in campaign_new.iterrows():
            similar = self.find_similar_campaigns(row['campaign'], top_k=3)
            
            if similar:
                coverage_count += 1
                # Weighted average
                weights = np.array([s['similarity'] for s in similar])
                weights = weights / weights.sum()
                ltv_values = np.array([s['ltv'] for s in similar])
                ltv_pred = np.sum(weights * ltv_values)
            else:
                # Fallback to global mean
                ltv_pred = np.mean(list(self.campaign_ltv_map.values()))
            
            predictions.append(ltv_pred)
        
        campaign_new['ltv_pred'] = predictions
        
        # Metrics
        mape = mean_absolute_percentage_error(campaign_new['ltv_d30'], campaign_new['ltv_pred'])
        coverage = coverage_count / len(campaign_new) * 100
        
        print(f"\n  ✓ Results:")
        print(f"    - MAPE: {mape:.4f}")
        print(f"    - Coverage (similarity ≥ 0.6): {coverage:.1f}%")
        print(f"    - Matched campaigns: {coverage_count}/{len(campaign_new)}")
        
        return {
            'mape': mape,
            'coverage': coverage,
            'n_new_campaigns': len(campaign_new),
            'n_matched': coverage_count
        }
    
    def save_models(self, models_path='models/semantic'):
        """Save semantic models"""
        models_path = Path(models_path)
        models_path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.vectorizer, models_path / 'tfidf_vectorizer.pkl')
        joblib.dump(self.campaign_vectors, models_path / 'campaign_vectors.pkl')
        joblib.dump(self.campaign_ltv_map, models_path / 'campaign_ltv_map.pkl')
        joblib.dump(self.campaign_names, models_path / 'campaign_names.pkl')
        
        print(f"\n✓ Saved semantic models to: {models_path}")

def main():
    """Main execution"""
    print("="*60)
    print("STEP 11: SEMANTIC SIMILARITY FALLBACK")
    print("="*60)
    
    # Load data
    df_train = pd.read_csv('data/features/train.csv')
    df_test = pd.read_csv('data/features/test.csv')
    
    # Initialize
    semantic = SemanticFallback()
    
    # Build vocabulary
    semantic.build_vocabulary(df_train)
    
    # Evaluate on new campaigns
    results = semantic.evaluate_on_new_campaigns(df_test, df_train)
    
    # Save models
    semantic.save_models()
    
    # Save summary
    summary_df = pd.DataFrame([{
        'method': 'semantic_fallback',
        'n_new_campaigns': results.get('n_new_campaigns', 0),
        'coverage_pct': results.get('coverage', 0),
        'mape': results.get('mape', None)
    }])
    summary_df.to_csv('results/step11_semantic_summary.csv', index=False)
    
    print("\n" + "="*60)
    print("✅ STEP 11 COMPLETED!")
    print("="*60)
    print("\nNext Step: step12_calibration_optimization.py")

if __name__ == "__main__":
    main()
```

---

## ✅ SUCCESS CRITERIA

- [x] TF-IDF vocabulary built successfully
- [x] Coverage ≥ 90% (similarity ≥ 0.6)
- [x] MAPE ≤ 10% for new campaigns
- [x] Models saved

---

## 🎯 NEXT STEP

➡️ **[Step 12: Calibration & Optimization](step12_calibration_optimization.md)**

---

**Estimated Time:** 3-4 hours  
**Difficulty:** ⭐⭐⭐ Hard
