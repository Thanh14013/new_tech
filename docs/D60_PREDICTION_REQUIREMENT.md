# 📊 D60 Prediction Requirement

**Date:** January 21, 2026  
**Version:** 2.1  
**Author:** GitHub Copilot

---

## ⭐ YÊU CẦU QUAN TRỌNG

### Nguyên Tắc Cơ Bản

**PREDICTION (Dự đoán):**
- ✅ **Luôn luôn predict đến D60** cho MỌI app và campaign
- ✅ Áp dụng cho tất cả: App thường, Wool app, campaigns mới, campaigns cũ
- ✅ Bất kể actual data có đến D60 hay không

**ACTUAL (Thực tế):**
- ✅ Có thể là D0, D1, D7, D30, hoặc bất kỳ ngày nào
- ✅ Tùy thuộc vào data thực sự có sẵn
- ✅ Ví dụ: Wool app Nov/Dec 2025 chỉ có actual đến D30, nhưng vẫn phải có predict D60

---

## 🎯 KHI SỬ DỤNG TOOL

### Tool Display Logic

```python
# PREDICTION (Always D60)
- Show predictions from D0 → D60 (full curve)
- All apps/campaigns must have D60 prediction
- Interpolate if needed (e.g., from D0, D7, D14, D30, D60 points)

# ACTUAL (Variable)
- Show actual data from D0 → D{max_actual_day}
- max_actual_day depends on:
  1. Data availability (which columns exist)
  2. Cohort age (calendar days since install)
  3. App-specific constraints (e.g., Wool D30 limit)
- Example scenarios:
  - Fresh cohort (installed yesterday): actual up to D1
  - Old cohort (installed 2 months ago): actual up to D60
  - Wool Nov/Dec cohorts: actual up to D30 only
```

### Visualization

```
Chart Display:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Predicted LTV (Blue): D0 ────────────────→ D60
  Actual LTV (Green):   D0 ──→ D{actual}
                                  ↑
                        (ends when data stops)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 🏗️ MODEL TRAINING

### All Methods Must Output D60

**Step 7 - Hurdle Model:**
```python
# Stage 1: Predict P(payer_d60)
y_is_payer = (df['ltv_d60'] > 0).astype(int)

# Stage 2: Predict E[LTV_D60 | payer]
y_ltv_payers = df.loc[is_payer, 'ltv_d60']

# Final: ltv_pred_d60 = prob_payer × amount
```

**Step 8 - Curve Fitting:**
```python
# Fit curve using historical data
# Predict at target_day = 60
ltv_d60 = power_law_curve(60, a, b, c)
```

**Step 9 - ML Multiplier:**
```python
# Train multiplier for D60
multiplier = ltv_d60 / (rev_d0 + rev_d1)
```

**Step 10 - Look-alike:**
```python
# Use ltv_d60 as target
campaign_avg['ltv_d60'].mean()
```

---

## 📦 DATA STRUCTURE

### Feature Files Must Include:

```python
Required Columns for Training:
- ltv_d60         # Target variable
- cumrev_d0...d60 # Actual revenue (if available)
- Other features  # Standard features

Required Columns for Prediction:
- pred_cumrev_d0...d60  # Predicted curve
- actual columns optional
```

### Config Structure:

```yaml
target:
  target_day: 60  # Always 60
  
windows:
  window_d7:
    feature_days: [0, 1, 2, 3, 4, 5, 6, 7]
    prediction_horizon: 60  # Not 30!
```

---

## 🔍 VALIDATION CHECKLIST

### Before Deployment:

- [ ] All models output `ltv_d60` (not `ltv_d30`)
- [ ] Prediction files contain `pred_cumrev_d60` column
- [ ] Tool shows D60 predictions for ALL campaigns
- [ ] Tool correctly truncates actual data (not predictions)
- [ ] Wool app has D60 predictions despite D30 actual limit
- [ ] New campaigns (via semantic matching) get D60 predictions
- [ ] Test cohorts (M12) have D60 predictions

### Example Test Cases:

```python
# Test 1: Wool app with recent cohorts
app_id = "wool"
install_date = "2025-12-15"  # Recent
# Expected:
# - Predicted LTV: D0 → D60 (full curve)
# - Actual LTV: D0 → D30 (limited by Wool constraint)

# Test 2: Regular app with old cohorts
app_id = "regular_app"
install_date = "2025-10-01"  # 2 months ago
# Expected:
# - Predicted LTV: D0 → D60 (full curve)
# - Actual LTV: D0 → D60 (cohort aged enough)

# Test 3: New campaign (no training data)
campaign = "new_campaign_2026"
# Expected:
# - Predicted LTV: D0 → D60 (via semantic matching)
# - Actual LTV: D0 → D{age} (depends on age)
```

---

## ⚠️ COMMON MISTAKES TO AVOID

### ❌ Wrong:
```python
# Training only for D30
y_train = df['ltv_d30']

# Tool predicting only to D30
if wool_app:
    target_day = 30  # WRONG!
```

### ✅ Correct:
```python
# Always train for D60
y_train = df['ltv_d60']

# Tool always predicts to D60
target_day = 60  # For all apps

# Only actual data is truncated
max_actual_day = min(cohort_age, 30) if wool_app else cohort_age
```

---

## 📝 WHY THIS MATTERS

### Business Justification:

1. **Consistency:** Tất cả campaigns comparable at same horizon (D60)
2. **Future Planning:** Luôn biết projected D60 LTV, dù actual chưa đến
3. **Tool Usability:** User không cần lo "predict đến ngày nào?"
4. **Model Fairness:** Tất cả models evaluated on same target (D60)

### Technical Benefits:

1. **Simpler Logic:** One target (D60) for all
2. **No Ambiguity:** Predict vs Actual roles rõ ràng
3. **Flexible Actual:** Actual data đến đâu cũng được
4. **Better UX:** Chart luôn hiển thị đầy đủ horizon

---

## 🚀 IMPLEMENTATION STATUS

| Component | Status | Notes |
|-----------|--------|-------|
| Strategy Document | ✅ Updated | Added D60 requirement section |
| Step 7 - Hurdle | ✅ Updated | Target changed to ltv_d60 |
| Step 8 - Curve Fitting | ✅ Updated | Predict at target_day=60 |
| Step 9 - ML Multiplier | ✅ Updated | Multiplier for ltv_d60 |
| Step 10 - Look-alike | ✅ Updated | Cluster avg for ltv_d60 |
| Tool - prediction_engine | ✅ Verified | Default target_day=60 |
| Tool - app.py | ✅ Verified | Uses target_day=None (60) |
| Validation Scripts | ⏳ Todo | Add checks for D60 presence |

---

## 📞 SUMMARY

> **"Predict luôn luôn đến D60. Actual đến bao nhiêu thì kệ nó."**

- ✅ Every app, every campaign → D60 prediction
- ✅ Actual data → flexible, depends on data availability
- ✅ Tool display → Blue curve (predict) always to D60, Green curve (actual) stops when data stops
- ✅ Training → ltv_d60 as target for all methods
- ✅ Validation → Ensure D60 columns exist in all prediction files

---

**Last Updated:** January 21, 2026  
**Questions?** Check [STRATEGY_APP_CAMPAIGN_LTV_PREDICTION.md](./STRATEGY_APP_CAMPAIGN_LTV_PREDICTION.md)
