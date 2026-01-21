# Step 13: Production Pipeline Integration
## Deploy Prediction Pipeline & Streamlit App

**Thời gian:** 1 ngày  
**Độ khó:** ⭐⭐⭐ Khó  
**Prerequisites:** Step 12 completed  

---

## 🎯 MỤC TIÊU

1. **Production Prediction Engine:**
   - Single API: `predict_ltv(app_id, campaign, features)`
   - Automatic method selection based on Step 12
   - Calibration applied
   - Fallback chain: Hurdle → ML Multiplier → Look-alike → Semantic

2. **Streamlit App Integration:**
   - Update existing `tool_total/app.py`
   - Add new V2.1 models
   - Interactive prediction interface

---

## 📥 INPUT

- All trained models from Steps 7-12
- `models/ensemble/method_selection.csv`
- `models/calibration/calibration_params.pkl`
- Existing `tool_total/` app structure

---

## 📤 OUTPUT

- `tool_total/prediction_engine.py` (Updated prediction engine)
- `tool_total/app.py` (Updated Streamlit app)
- `README_DEPLOYMENT.md` (Deployment guide)

---

## 🔧 IMPLEMENTATION

### File: `tool_total/prediction_engine.py`

```python
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class LTVPredictionEngine:
    """Production LTV Prediction Engine (V2.1)"""
    
    def __init__(self, models_path='models'):
        self.models_path = Path(models_path)
        
        # Load all models
        self.models = self._load_all_models()
        
        # Load method selection
        self.method_selection = pd.read_csv(
            self.models_path / 'ensemble' / 'method_selection.csv'
        )
        
        # Load calibration params
        self.calibration_params = joblib.load(
            self.models_path / 'calibration' / 'calibration_params.pkl'
        )
        
        # Load tier classification
        self.campaign_tiers = pd.read_csv(
            'data/features/campaign_tiers.csv'
        )
        
        print("✓ Prediction Engine initialized")
        print(f"  - Models loaded: {len(self.models)} tiers")
        print(f"  - Method selections: {len(self.method_selection)} campaigns")
    
    def _load_all_models(self):
        """Load all trained models"""
        models = {}
        
        for tier in ['tier1', 'tier2', 'tier3']:
            tier_path = self.models_path / tier
            
            if not tier_path.exists():
                continue
            
            models[tier] = {}
            
            # Hurdle
            try:
                models[tier]['hurdle_stage1'] = joblib.load(tier_path / 'hurdle_stage1_classifier.pkl')
                models[tier]['hurdle_stage2'] = joblib.load(tier_path / 'hurdle_stage2_regressor.pkl')
            except:
                pass
            
            # Curve fitting (Tier 1 only)
            if tier == 'tier1':
                try:
                    models[tier]['curve_params'] = joblib.load(tier_path / 'curve_fitting_params.pkl')
                    models[tier]['curve_priors'] = joblib.load(tier_path / 'curve_fitting_priors.pkl')
                except:
                    pass
            
            # ML Multiplier
            try:
                models[tier]['ml_multiplier'] = joblib.load(tier_path / 'ml_multiplier.pkl')
            except:
                pass
        
        # Fallback models
        fallback_path = self.models_path / 'fallback'
        if fallback_path.exists():
            models['fallback'] = {}
            try:
                models['fallback']['lookalike_scaler'] = joblib.load(fallback_path / 'lookalike_scaler.pkl')
                models['fallback']['lookalike_kmeans'] = joblib.load(fallback_path / 'lookalike_kmeans.pkl')
                models['fallback']['lookalike_cluster_ltv'] = joblib.load(fallback_path / 'lookalike_cluster_avg_ltv.pkl')
            except:
                pass
        
        # Semantic
        semantic_path = self.models_path / 'semantic'
        if semantic_path.exists():
            models['semantic'] = {}
            try:
                models['semantic']['tfidf'] = joblib.load(semantic_path / 'tfidf_vectorizer.pkl')
                models['semantic']['campaign_vectors'] = joblib.load(semantic_path / 'campaign_vectors.pkl')
                models['semantic']['campaign_ltv_map'] = joblib.load(semantic_path / 'campaign_ltv_map.pkl')
                models['semantic']['campaign_names'] = joblib.load(semantic_path / 'campaign_names.pkl')
            except:
                pass
        
        return models
    
    def get_campaign_tier(self, app_id, campaign):
        """Get tier for a campaign"""
        mask = (self.campaign_tiers['app_id'] == app_id) & \
               (self.campaign_tiers['campaign'] == campaign)
        
        if mask.sum() > 0:
            return self.campaign_tiers[mask].iloc[0]['tier']
        else:
            return 'tier3'  # Default to tier3 for unknown campaigns
    
    def get_selected_method(self, app_id, campaign):
        """Get best method for a campaign"""
        mask = (self.method_selection['app_id'] == app_id) & \
               (self.method_selection['campaign'] == campaign)
        
        if mask.sum() > 0:
            return self.method_selection[mask].iloc[0]['best_method']
        else:
            return 'lookalike'  # Default fallback
    
    def predict_hurdle(self, tier, features):
        """Predict using Hurdle model"""
        if tier not in self.models or 'hurdle_stage1' not in self.models[tier]:
            return None
        
        try:
            stage1 = self.models[tier]['hurdle_stage1']
            stage2 = self.models[tier]['hurdle_stage2']
            
            p_payer = stage1.predict_proba(features)[:, 1]
            ltv_given_payer = stage2.predict(features)
            
            ltv_pred = p_payer * np.maximum(ltv_given_payer, 0)
            
            return ltv_pred[0]
        except:
            return None
    
    def predict_ml_multiplier(self, tier, features, rev_sum):
        """Predict using ML Multiplier"""
        if tier not in self.models or 'ml_multiplier' not in self.models[tier]:
            return None
        
        try:
            model = self.models[tier]['ml_multiplier']
            multiplier = model.predict(features, num_iteration=model.best_iteration)
            ltv_pred = rev_sum * max(multiplier[0], 0)
            
            return ltv_pred
        except:
            return None
    
    def predict_lookalike(self, features):
        """Predict using Look-alike"""
        if 'fallback' not in self.models:
            return None
        
        try:
            scaler = self.models['fallback']['lookalike_scaler']
            kmeans = self.models['fallback']['lookalike_kmeans']
            cluster_ltv = self.models['fallback']['lookalike_cluster_ltv']
            
            # Feature subset for lookalike
            lookalike_features = features[['rev_sum', 'actual_cpi', 'engagement_score']].values
            
            # Scale and predict cluster
            features_scaled = scaler.transform(lookalike_features)
            cluster_id = kmeans.predict(features_scaled)[0]
            
            # Get LTV
            ltv_pred = cluster_ltv.get(cluster_id, np.mean(list(cluster_ltv.values())))
            
            return ltv_pred
        except:
            return None
    
    def predict_semantic(self, campaign):
        """Predict using Semantic matching"""
        if 'semantic' not in self.models:
            return None
        
        try:
            tfidf = self.models['semantic']['tfidf']
            campaign_vectors = self.models['semantic']['campaign_vectors']
            campaign_ltv_map = self.models['semantic']['campaign_ltv_map']
            campaign_names = self.models['semantic']['campaign_names']
            
            # Vectorize query
            query_vec = tfidf.transform([campaign])
            
            # Find similar
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_vec, campaign_vectors).flatten()
            
            # Top 3
            top_k_indices = np.argsort(similarities)[-3:][::-1]
            
            # Weighted average
            weights = similarities[top_k_indices]
            if weights.sum() > 0:
                weights = weights / weights.sum()
                ltv_values = [campaign_ltv_map[campaign_names[i]] for i in top_k_indices]
                ltv_pred = np.sum(weights * ltv_values)
                
                return ltv_pred
            else:
                return np.mean(list(campaign_ltv_map.values()))
        except:
            return None
    
    def calibrate_prediction(self, prediction, tier):
        """Apply calibration"""
        if tier not in self.calibration_params:
            return prediction
        
        params = self.calibration_params[tier]
        anchor = params['anchor']
        alpha = params['alpha']
        
        # Anchor & Adjust
        calibrated = anchor * (1 + alpha * (prediction - anchor) / (anchor + 1e-6))
        
        return max(calibrated, 0)
    
    def predict(self, app_id, campaign, features_dict):
        """Main prediction function"""
        
        # Convert features to DataFrame
        features = pd.DataFrame([features_dict])
        
        # Get tier
        tier = self.get_campaign_tier(app_id, campaign)
        
        # Get selected method
        method = self.get_selected_method(app_id, campaign)
        
        print(f"Predicting: {app_id}::{campaign} | Tier: {tier} | Method: {method}")
        
        # Try selected method first
        ltv_pred = None
        
        if method == 'hurdle':
            ltv_pred = self.predict_hurdle(tier, features)
        elif method == 'ml_multiplier':
            ltv_pred = self.predict_ml_multiplier(tier, features, features_dict['rev_sum'])
        elif method == 'lookalike':
            ltv_pred = self.predict_lookalike(features)
        elif method == 'semantic':
            ltv_pred = self.predict_semantic(campaign)
        
        # Fallback chain if method fails
        if ltv_pred is None:
            print(f"  → Method '{method}' failed, trying fallback chain...")
            
            # Try Hurdle
            ltv_pred = self.predict_hurdle(tier, features)
            
            # Try ML Multiplier
            if ltv_pred is None:
                ltv_pred = self.predict_ml_multiplier(tier, features, features_dict['rev_sum'])
            
            # Try Look-alike
            if ltv_pred is None:
                ltv_pred = self.predict_lookalike(features)
            
            # Try Semantic
            if ltv_pred is None:
                ltv_pred = self.predict_semantic(campaign)
            
            # Final fallback: global median
            if ltv_pred is None:
                ltv_pred = 0.15  # Hardcoded fallback
        
        # Calibrate
        ltv_calibrated = self.calibrate_prediction(ltv_pred, tier)
        
        return {
            'ltv_pred': ltv_calibrated,
            'ltv_raw': ltv_pred,
            'tier': tier,
            'method': method
        }

# Example usage
if __name__ == "__main__":
    # Initialize engine
    engine = LTVPredictionEngine(models_path='../models')
    
    # Example prediction
    features = {
        'rev_sum': 0.05,
        'rev_max': 0.03,
        'engagement_score': 0.4,
        'actual_cpi': 0.8,
        'cpi_quality_score': 0.6
    }
    
    result = engine.predict(
        app_id='app_001',
        campaign='summer_2024_tier1',
        features_dict=features
    )
    
    print(f"\nPrediction Result:")
    print(f"  LTV (calibrated): ${result['ltv_pred']:.4f}")
    print(f"  LTV (raw): ${result['ltv_raw']:.4f}")
    print(f"  Tier: {result['tier']}")
    print(f"  Method: {result['method']}")
```

---

### Update `tool_total/app.py`

```python
import streamlit as st
import pandas as pd
from prediction_engine import LTVPredictionEngine

# Initialize engine (cache it)
@st.cache_resource
def load_engine():
    return LTVPredictionEngine(models_path='../models')

st.set_page_config(page_title="LTV Prediction V2.1", layout="wide")

st.title("🎯 LTV/ROAS Prediction System V2.1")
st.markdown("---")

# Sidebar
st.sidebar.header("Input Features")

app_id = st.sidebar.text_input("App ID", "app_001")
campaign = st.sidebar.text_input("Campaign Name", "summer_2024")

st.sidebar.markdown("### Revenue Features")
rev_sum = st.sidebar.number_input("Revenue Sum (D0+D1)", 0.0, 10.0, 0.05, 0.01)
rev_max = st.sidebar.number_input("Revenue Max", 0.0, 10.0, 0.03, 0.01)

st.sidebar.markdown("### Engagement")
engagement_score = st.sidebar.slider("Engagement Score", 0.0, 1.0, 0.4, 0.05)

st.sidebar.markdown("### CPI")
actual_cpi = st.sidebar.number_input("Actual CPI", 0.0, 5.0, 0.8, 0.1)
cpi_quality_score = st.sidebar.slider("CPI Quality Score", 0.0, 1.0, 0.6, 0.05)

# Predict button
if st.sidebar.button("🚀 Predict LTV"):
    
    # Load engine
    engine = load_engine()
    
    # Features
    features = {
        'rev_sum': rev_sum,
        'rev_max': rev_max,
        'engagement_score': engagement_score,
        'actual_cpi': actual_cpi,
        'cpi_quality_score': cpi_quality_score
    }
    
    # Predict
    with st.spinner("Predicting..."):
        result = engine.predict(app_id, campaign, features)
    
    # Display results
    st.success("✅ Prediction Complete!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("LTV D30 (Calibrated)", f"${result['ltv_pred']:.4f}")
    
    with col2:
        st.metric("Tier", result['tier'])
    
    with col3:
        st.metric("Method Used", result['method'])
    
    st.markdown("---")
    
    st.info(f"**Raw Prediction:** ${result['ltv_raw']:.4f}")
    
    # Show ROAS
    if actual_cpi > 0:
        roas = result['ltv_pred'] / actual_cpi
        st.metric("ROAS (LTV/CPI)", f"{roas:.2f}x")

st.markdown("---")
st.caption("LTV Prediction System V2.1 | Powered by XGBoost, LightGBM, TF-IDF")
```

---

## ✅ SUCCESS CRITERIA

- [x] Prediction engine deployed successfully
- [x] Streamlit app updated and functional
- [x] Fallback chain working
- [x] Calibration applied automatically

---

## 🎯 DEPLOYMENT

### Run Streamlit App

```bash
cd tool_total
streamlit run app.py
```

---

## 🎉 PROJECT COMPLETE!

All 13 steps implemented:
- ✅ Steps 1-6: Data preparation & features
- ✅ Steps 7-10: 4 modeling methods
- ✅ Steps 11-12: Fallback & optimization
- ✅ Step 13: Production deployment

**Next:** Monitor performance, retrain monthly, A/B test in production.

---

**Estimated Time:** 6-8 hours  
**Difficulty:** ⭐⭐⭐ Hard
