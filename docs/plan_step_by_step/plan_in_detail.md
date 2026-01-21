# Kế Hoạch Triển Khai Chi Tiết - LTV/ROAS Prediction V2.1
## Phân Tích Theo Từng Bước (Step by Step)

**Ngày:** 21/01/2026  
**Version:** 2.1 (Advanced)  
**Tổng thời gian:** 5-6 tuần  

---

## 📋 TỔNG QUAN KẾ HOẠCH

### Mục Tiêu Chính
Xây dựng hệ thống dự đoán LTV/ROAS D30-60 từ dữ liệu D0-D1 với:
- ✅ MAPE ≤ 5% cho ≥80% campaigns
- ✅ Coverage ≥98% (bao gồm new campaigns)
- ✅ 4 phương pháp modeling: Hurdle, Curve Fitting, ML Multiplier, Look-alike
- ✅ Calibration layer để giảm bias
- ✅ Semantic matching cho new campaigns

### Cấu Trúc Thư Mục Dự Án

```
new_technology/
├── data/
│   ├── raw/                    # Data gốc (đã có)
│   │   ├── *.csv              # Các file tháng 8-12
│   │   └── wool/              # Data wool riêng
│   ├── processed/             # Data đã xử lý (sẽ tạo)
│   ├── features/              # Features đã engineer (sẽ tạo)
│   └── interim/               # Data trung gian (sẽ tạo)
│
├── scripts/                   # Code thực thi
│   ├── step01_*.py           # Scripts theo từng step
│   ├── step02_*.py
│   └── ...
│
├── models/                    # Lưu trained models
│   ├── tier1/                # Models cho Tier 1 campaigns
│   ├── tier2/                # Models cho Tier 2 campaigns
│   ├── tier3/                # Models cho Tier 3 campaigns
│   ├── fallback/             # App-level & cluster models
│   └── semantic/             # TF-IDF index cho semantic matching
│
├── results/                   # Kết quả đánh giá
│   ├── validation/           # Kết quả validation (T11)
│   ├── test/                 # Kết quả test (T12)
│   └── comparisons/          # So sánh các methods
│
├── config/                    # Cấu hình
│   ├── config.yaml           # Config chung
│   ├── campaign_tiers.json   # Phân loại tiers (sẽ tạo)
│   ├── bias_db.json          # Historical bias (sẽ tạo)
│   └── model_registry.json   # Registry các models (sẽ tạo)
│
├── tool_total/                # Streamlit app (đã có)
│   ├── app.py
│   └── ...
│
└── docs/                      # Documentation
    └── plan_step_by_step/    # Folder này
        ├── plan_in_detail.md # File này
        └── step*.md          # Chi tiết từng step
```

---

## 🎯 CHIA NHỎ THÀNH 12 STEPS

### **GIAI ĐOẠN 1: SETUP & DATA PREPARATION (Week 1)**

#### [Step 1: Environment Setup & Data Loading](step01_environment_setup.md)
- **Thời gian:** 0.5 ngày
- **Mục tiêu:** Chuẩn bị môi trường, load và khám phá data
- **Output:** 
  - `config/config.yaml` (cấu hình project)
  - `data/interim/data_overview.csv` (thống kê tổng quan)

#### [Step 2: Data Cleaning & Validation](step02_data_cleaning.md)
- **Thời gian:** 1 ngày
- **Mục tiêu:** Làm sạch data, xử lý missing values, outliers
- **Output:** 
  - `data/processed/clean_data_T*.csv`
  - `results/data_quality_report.html`

#### [Step 3: Campaign Tier Classification](step03_tier_classification.md)
- **Thời gian:** 1 ngày
- **Mục tiêu:** Phân loại campaigns thành Tier 1/2/3
- **Output:** 
  - `config/campaign_tiers.json`
  - `results/tier_distribution.png`

---

### **GIAI ĐOẠN 2: FEATURE ENGINEERING (Week 1-2)**

#### [Step 4: Basic Feature Engineering](step04_basic_features.md)
- **Thời gian:** 1.5 ngày
- **Mục tiêu:** Tạo revenue, velocity, engagement features
- **Output:** 
  - `data/features/basic_features_T*.parquet`
  - `results/feature_stats.csv`

#### [Step 5: Advanced Feature Engineering](step05_advanced_features.md)
- **Thời gian:** 1 ngày
- **Mục tiêu:** Historical profiles, CPI quality signals, comparative features
- **Output:** 
  - `data/features/full_features_T*.parquet`
  - `config/feature_definitions.json`

#### [Step 6: Historical Bias Calculation](step06_bias_calculation.md)
- **Thời gian:** 0.5 ngày
- **Mục tiêu:** Tính toán historical bias cho calibration
- **Output:** 
  - `config/bias_db.json`
  - `results/bias_analysis.html`

---

### **GIAI ĐOẠN 3: MODEL TRAINING - MULTI-METHOD (Week 2-3)**

#### [Step 7: Two-Stage Hurdle Model Training](step07_hurdle_model.md)
- **Thời gian:** 2 ngày
- **Mục tiêu:** Train Stage 1 (Classification) + Stage 2 (Regression)
- **Output:** 
  - `models/tier*/hurdle_models/`
  - `results/validation/hurdle_performance.csv`

#### [Step 8: Curve Fitting with Bayesian Priors](step08_curve_fitting.md)
- **Thời gian:** 1.5 ngày
- **Mục tiêu:** Fit Exponential, Power, Log curves với Bayesian priors
- **Output:** 
  - `models/tier*/curve_models/`
  - `results/validation/curve_performance.csv`

#### [Step 9: ML Multiplier Models Training](step09_ml_multiplier.md)
- **Thời gian:** 2 ngày
- **Mục tiêu:** Train XGBoost + LightGBM cho growth multiplier
- **Output:** 
  - `models/tier*/ml_models/`
  - `results/validation/ml_performance.csv`

#### [Step 10: Look-alike System Building](step10_lookalike.md)
- **Thời gian:** 1.5 ngày
- **Mục tiêu:** Build similarity index, nearest neighbor matching
- **Output:** 
  - `models/tier*/lookalike_indices/`
  - `results/validation/lookalike_performance.csv`

---

### **GIAI ĐOẠN 4: FALLBACK & OPTIMIZATION (Week 3-4)**

#### [Step 11: Semantic Similarity Mapping](step11_semantic_matching.md)
- **Thời gian:** 1 ngày
- **Mục tiêu:** Build TF-IDF index cho new campaigns
- **Output:** 
  - `models/semantic/tfidf_vectorizer.pkl`
  - `models/semantic/campaign_embeddings.npy`
  - `results/semantic_match_quality.csv`

#### [Step 12: Model Selection & Calibration](step12_selection_calibration.md)
- **Thời gian:** 2 ngày
- **Mục tiêu:** Chọn best method per campaign, apply calibration
- **Output:** 
  - `config/model_registry.json`
  - `models/calibration_params.json`
  - `results/test/final_performance_T12.csv`

---

### **GIAI ĐOẠN 5: PRODUCTION & DEPLOYMENT (Week 4-5)**

#### [Step 13: Production Pipeline Integration](step13_production_pipeline.md)
- **Thời gian:** 2 ngày (không có trong 12 steps ban đầu - bổ sung)
- **Mục tiêu:** Tích hợp vào Streamlit app, API endpoints
- **Output:** 
  - `tool_total/prediction_engine.py` (updated)
  - `tool_total/model_loader.py`
  - `results/production_validation.csv`

---

## 📊 TIMELINE CHI TIẾT

### Week 1: Setup & Data + Feature Engineering
```
Day 1-2:   Steps 1-3 (Setup, Cleaning, Tier Classification)
Day 3-4:   Steps 4-5 (Basic & Advanced Features)
Day 5:     Step 6 (Bias Calculation)
```

### Week 2: Model Training - Part 1
```
Day 1-2:   Step 7 (Hurdle Model)
Day 3-4:   Step 8 (Curve Fitting)
Day 5:     Step 9 (ML Multiplier - Start)
```

### Week 3: Model Training - Part 2 + Fallback
```
Day 1-2:   Step 9 (ML Multiplier - Finish)
Day 3-4:   Step 10 (Look-alike)
Day 5:     Step 11 (Semantic Matching)
```

### Week 4: Optimization & Testing
```
Day 1-3:   Step 12 (Selection & Calibration)
Day 4-5:   Step 13 (Production Integration)
```

### Week 5: Buffer & Fine-tuning
```
Day 1-3:   Testing, debugging, optimization
Day 4-5:   Documentation, handover
```

---

## 🎯 SUCCESS CRITERIA PER STEP

| Step | Deliverable | Success Metric |
|------|-------------|----------------|
| 1 | Data loaded | All 5 months data accessible |
| 2 | Clean data | <1% missing values, no duplicates |
| 3 | Tier classification | 3 tiers defined, ~30/40/30% distribution |
| 4 | Basic features | 15+ features created |
| 5 | Advanced features | 30+ total features |
| 6 | Bias DB | Historical bias for 1000+ campaigns |
| 7 | Hurdle models | Stage 1 AUC ≥0.75, Stage 2 R² ≥0.6 |
| 8 | Curve fitting | R² ≥0.65 for Tier 1 campaigns |
| 9 | ML models | MAPE <8% on validation |
| 10 | Look-alike | Top-50 similarity matching working |
| 11 | Semantic index | Match rate ≥85% for new campaigns |
| 12 | Final model | MAPE ≤5% for ≥80% campaigns |
| 13 | Production | Streamlit app working with new models |

---

## 🔧 DEPENDENCIES GIỮA CÁC STEPS

```
Step 1 (Setup)
    ↓
Step 2 (Cleaning) ─────────────┐
    ↓                          │
Step 3 (Tier Classification)   │
    ↓                          │
Step 4 (Basic Features) ←──────┘
    ↓
Step 5 (Advanced Features)
    ├──→ Step 6 (Bias Calculation)
    │
    ├──→ Step 7 (Hurdle Model) ──┐
    │                             │
    ├──→ Step 8 (Curve Fitting) ──┤
    │                             ├─→ Step 12 (Selection & Calibration)
    ├──→ Step 9 (ML Multiplier) ──┤                ↓
    │                             │           Step 13 (Production)
    └──→ Step 10 (Look-alike) ────┘
                                  │
Step 11 (Semantic Matching) ──────┘
```

---

## 📁 FILES CẦN TẠO

### Scripts (trong `scripts/`)
1. `step01_setup_and_load.py`
2. `step02_data_cleaning.py`
3. `step03_classify_tiers.py`
4. `step04_engineer_basic_features.py`
5. `step05_engineer_advanced_features.py`
6. `step06_calculate_bias.py`
7. `step07_train_hurdle_model.py`
8. `step08_fit_curves_bayesian.py`
9. `step09_train_ml_multiplier.py`
10. `step10_build_lookalike.py`
11. `step11_build_semantic_index.py`
12. `step12_select_and_calibrate.py`
13. `step13_integrate_production.py`

### Config Files (trong `config/`)
- `config.yaml` - Cấu hình chung
- `campaign_tiers.json` - Tier classification results
- `bias_db.json` - Historical bias database
- `model_registry.json` - Tracking best models per campaign
- `feature_definitions.json` - Feature metadata

### Utility Scripts (trong `scripts/utils/`)
- `data_utils.py` - Load/save data functions
- `feature_utils.py` - Feature engineering helpers
- `model_utils.py` - Model training/evaluation helpers
- `plot_utils.py` - Visualization functions
- `metric_utils.py` - MAPE, R², AUC calculations

---

## 🚀 CÁCH SỬ DỤNG PLAN NÀY

### 1. Đọc Plan Tổng Quan (file này)
- Hiểu được roadmap tổng thể
- Xác định dependencies giữa các steps
- Ước tính thời gian cần thiết

### 2. Đọc Chi Tiết Từng Step
- Mở file `stepXX_*.md` tương ứng
- Đọc mục tiêu, input/output
- Xem code examples và pseudo-code
- Hiểu success criteria

### 3. Triển Khai Từng Step
- Tạo script theo template trong file step
- Chạy và kiểm tra output
- Validate theo success criteria
- Commit code và move to next step

### 4. Tracking Progress
- [ ] Step 1: Environment Setup ✅ (đánh dấu khi xong)
- [ ] Step 2: Data Cleaning
- [ ] Step 3: Tier Classification
- ... (tiếp tục)

---

## ⚠️ LƯU Ý QUAN TRỌNG

### 1. Data Security
- Không commit raw data lên git (thêm vào `.gitignore`)
- Chỉ commit processed features nếu cần

### 2. Reproducibility
- Set random seed cho tất cả models
- Document versions của libraries
- Save configs cùng với models

### 3. Scalability
- Sử dụng `parquet` thay vì `csv` cho data lớn
- Parallel processing khi train nhiều campaigns
- Batch prediction trong production

### 4. Monitoring
- Log training metrics vào `results/`
- Track MAPE per campaign qua các tháng
- Alert nếu MAPE > threshold

---

## 📞 SUPPORT & TROUBLESHOOTING

### Common Issues

**Issue 1: Memory error khi load data**
→ Solution: Load từng tháng, merge sau

**Issue 2: Hurdle model Stage 1 AUC thấp (<0.7)**
→ Solution: Tăng features, adjust scale_pos_weight

**Issue 3: Semantic matching rate thấp (<80%)**
→ Solution: Tune TF-IDF parameters, thử sentence-transformers

**Issue 4: Calibration không improve MAPE**
→ Solution: Check bias calculation, increase validation data

---

## 🎓 LEARNING RESOURCES

Để hiểu rõ từng technique:
- **Two-Stage Hurdle:** Xem `V2.1_QUICK_REFERENCE.md`
- **Bayesian Priors:** Xem `V2.1_ENHANCEMENTS_SUMMARY.md`
- **Semantic Matching:** Xem `VERSION_EVOLUTION_SUMMARY.md`
- **Calibration:** Xem main strategy document Section 2.4

---

## ✅ FINAL CHECKLIST

Trước khi deploy production:
- [ ] All 13 steps completed
- [ ] MAPE ≤5% for ≥80% Tier 1 campaigns
- [ ] Coverage ≥98% (including new campaigns)
- [ ] Streamlit app updated and tested
- [ ] Model registry documented
- [ ] Bias update mechanism scheduled
- [ ] Production validation passed

---

**Document Version:** 1.0  
**Last Updated:** 21/01/2026  
**Next Review:** After Step 6 completion  

**Bắt đầu từ:** [Step 1: Environment Setup](step01_environment_setup.md) →
