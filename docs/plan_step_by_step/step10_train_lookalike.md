# Step 10: Look-alike Method  
## Clustering-Based Similarity Matching

**Thời gian:** 0.5 ngày  
**Độ khó:** ⭐⭐ Trung bình  
**Prerequisites:** Step 9 completed  

---

## 🎯 MỤC TIÊU

Look-alike method: Tìm "twin campaigns" tương tự, lấy average LTV

**Algorithm:**
1. Cluster campaigns theo features (KMeans, k=20)
2. Với campaign mới/sparse → tìm closest cluster
3. LTV_pred = average LTV của cluster

**Use cases:**
- New campaigns (no history)
- Tier 3 (sparse data)

---

## 📥 INPUT

- `data/features/train.csv`
- `data/features/validation.csv`

---

## 📤 OUTPUT

- `models/fallback/lookalike_kmeans.pkl`
- `models/fallback/lookalike_cluster_avg_ltv.pkl`
- `results/step10_lookalike_evaluation.html`

---

## 🔧 IMPLEMENTATION

### File: `scripts/step10_train_lookalike.py`

```python
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score

class LookalikeModel:
    """Clustering-based look-alike matching"""
    
    def __init__(self, config_path='config/config.yaml', n_clusters=20):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = None
        self.cluster_avg_ltv = {}
    
    def prepare_campaign_features(self, df):
        """Aggregate features at campaign level"""
        print("\n[Aggregation] Creating campaign-level features...")
        
        # Group by app_id + campaign
        campaign_agg = df.groupby(['app_id', 'campaign']).agg({
            'rev_sum': 'mean',
            'rev_max': 'mean',
            'rev_d0_d1_ratio': 'mean',
            'engagement_score': 'mean',
            'user_quality_index': 'mean',
            'actual_cpi': 'mean',
            'cpi_quality_score': 'mean',
            'campaign_ltv_avg': 'first',
            'campaign_total_installs': 'first',
            'ltv_d30': 'mean'  # Target
        }).reset_index()
        
        print(f"  ✓ Aggregated {len(campaign_agg)} campaigns")
        
        return campaign_agg
    
    def train(self, df_train):
        """Train KMeans clustering"""
        print("\n[Training] KMeans Clustering...")
        
        # Prepare campaign features
        campaign_df = self.prepare_campaign_features(df_train)
        
        # Feature columns
        feature_cols = [
            'rev_sum', 'rev_max', 'rev_d0_d1_ratio',
            'engagement_score', 'user_quality_index',
            'actual_cpi', 'cpi_quality_score',
            'campaign_total_installs'
        ]
        
        X = campaign_df[feature_cols].fillna(0)
        
        # Standardize
        X_scaled = self.scaler.fit_transform(X)
        
        # KMeans
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.config['modeling']['random_seed'],
            n_init=10
        )
        
        campaign_df['cluster'] = self.kmeans.fit_predict(X_scaled)
        
        print(f"  ✓ Fitted KMeans with {self.n_clusters} clusters")
        
        # Calculate average LTV per cluster
        cluster_stats = campaign_df.groupby('cluster').agg({
            'ltv_d30': ['mean', 'median', 'count']
        })
        
        self.cluster_avg_ltv = campaign_df.groupby('cluster')['ltv_d30'].mean().to_dict()
        
        print(f"\n  Cluster Statistics:")
        for cluster_id, avg_ltv in sorted(self.cluster_avg_ltv.items()):
            count = (campaign_df['cluster'] == cluster_id).sum()
            print(f"    - Cluster {cluster_id}: {count} campaigns, Avg LTV=${avg_ltv:.4f}")
        
        return campaign_df
    
    def predict_campaign(self, campaign_features):
        """Predict LTV for one campaign"""
        
        # Feature columns
        feature_cols = [
            'rev_sum', 'rev_max', 'rev_d0_d1_ratio',
            'engagement_score', 'user_quality_index',
            'actual_cpi', 'cpi_quality_score',
            'campaign_total_installs'
        ]
        
        X = campaign_features[feature_cols].fillna(0).values.reshape(1, -1)
        
        # Standardize
        X_scaled = self.scaler.transform(X)
        
        # Find closest cluster
        cluster_id = self.kmeans.predict(X_scaled)[0]
        
        # Return average LTV of cluster
        ltv_pred = self.cluster_avg_ltv.get(cluster_id, np.mean(list(self.cluster_avg_ltv.values())))
        
        return ltv_pred, cluster_id
    
    def evaluate(self, df_val):
        """Evaluate on validation set"""
        print("\n[Evaluation] Look-alike Model...")
        
        # Prepare campaign features
        campaign_val = self.prepare_campaign_features(df_val)
        
        # Predict
        predictions = []
        clusters = []
        
        for idx, row in campaign_val.iterrows():
            ltv_pred, cluster_id = self.predict_campaign(row)
            predictions.append(ltv_pred)
            clusters.append(cluster_id)
        
        campaign_val['ltv_pred'] = predictions
        campaign_val['cluster'] = clusters
        
        # Metrics
        mape = mean_absolute_percentage_error(campaign_val['ltv_d30'], campaign_val['ltv_pred'])
        r2 = r2_score(campaign_val['ltv_d30'], campaign_val['ltv_pred'])
        
        print(f"  ✓ MAPE: {mape:.4f}")
        print(f"  ✓ R²: {r2:.4f}")
        
        return {'mape': mape, 'r2': r2}
    
    def save_models(self, models_path='models/fallback'):
        """Save models"""
        models_path = Path(models_path)
        models_path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.scaler, models_path / 'lookalike_scaler.pkl')
        joblib.dump(self.kmeans, models_path / 'lookalike_kmeans.pkl')
        joblib.dump(self.cluster_avg_ltv, models_path / 'lookalike_cluster_avg_ltv.pkl')
        
        print(f"\n✓ Saved models to: {models_path}")

def main():
    """Main execution"""
    print("="*60)
    print("STEP 10: LOOK-ALIKE METHOD")
    print("="*60)
    
    # Load data
    df_train = pd.read_csv('data/features/train.csv')
    df_val = pd.read_csv('data/features/validation.csv')
    
    # Initialize
    lookalike = LookalikeModel(n_clusters=20)
    
    # Train
    campaign_df = lookalike.train(df_train)
    
    # Evaluate
    results = lookalike.evaluate(df_val)
    
    # Save
    lookalike.save_models()
    
    # Summary
    summary_df = pd.DataFrame([{
        'method': 'lookalike',
        'n_clusters': lookalike.n_clusters,
        'mape': results['mape'],
        'r2': results['r2']
    }])
    summary_df.to_csv('results/step10_lookalike_summary.csv', index=False)
    
    print("\n" + "="*60)
    print("✅ STEP 10 COMPLETED!")
    print("="*60)
    print("\nNext Step: step11_semantic_fallback.py")

if __name__ == "__main__":
    main()
```

---

## ✅ SUCCESS CRITERIA

- [x] KMeans trained with 20 clusters
- [x] Cluster avg LTV calculated
- [x] MAPE ≤ 15% on validation
- [x] Models saved

---

## 🎯 NEXT STEP

➡️ **[Step 11: Semantic Fallback](step11_semantic_fallback.md)**

---

**Estimated Time:** 3-4 hours  
**Difficulty:** ⭐⭐ Medium
