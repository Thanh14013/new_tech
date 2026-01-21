# Step 1: Environment Setup & Data Loading
## Chuẩn Bị Môi Trường và Tải Dữ Liệu

**Thời gian:** 0.5 ngày  
**Độ khó:** ⭐ Dễ  
**Prerequisites:** None  

---

## 🎯 MỤC TIÊU

1. Cài đặt môi trường Python và thư viện cần thiết
2. Tạo cấu trúc thư mục dự án
3. Load và khám phá data từ tháng 8-12/2025
4. Tạo file config tổng quan

---

## 📥 INPUT

**Data Sources:**
- `data/raw/data_T8.csv` (Tháng 8)
- `data/raw/data_T9.csv` (Tháng 9)
- `data/raw/data_T10.csv` (Tháng 10)
- `data/raw/data_T11.csv` (Tháng 11)
- `data/raw/data_T12.csv` (Tháng 12)
- `data/raw/wool/data_wool_*.csv` (Data wool riêng)
- `data/raw/countries_tier.csv` (Phân loại quốc gia)
- `data/raw/apr.csv`, `jan.csv`, etc. (Data bổ sung nếu có)

---

## 📤 OUTPUT

### 1. Environment Files
- `requirements.txt` - Danh sách thư viện
- `.env` - Environment variables (nếu cần)

### 2. Config Files
- `config/config.yaml` - Cấu hình chung project

### 3. Data Overview
- `data/interim/data_overview.csv` - Thống kê tổng quan
- `results/step01_data_exploration.html` - Report HTML

### 4. Folder Structure
```
data/
  ├── processed/
  ├── features/
  └── interim/
models/
  ├── tier1/
  ├── tier2/
  ├── tier3/
  ├── fallback/
  └── semantic/
results/
  ├── validation/
  ├── test/
  └── comparisons/
```

---

## 🔧 IMPLEMENTATION

### 1. Install Dependencies

**File: `requirements.txt`**
```txt
# Core
pandas==2.1.4
numpy==1.26.3
scipy==1.11.4

# ML/Modeling
xgboost==2.0.3
lightgbm==4.1.0
scikit-learn==1.4.0
optuna==3.5.0

# Feature Engineering
featuretools==1.30.0

# Visualization
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.18.0

# NLP (for semantic matching)
scikit-learn==1.4.0
sentence-transformers==2.2.2  # Optional

# Utilities
pyyaml==6.0.1
tqdm==4.66.1
joblib==1.3.2

# Streamlit (đã có)
streamlit==1.30.0
```

**Cài đặt:**
```bash
pip install -r requirements.txt
```

---

### 2. Create Folder Structure

**File: `scripts/step01_setup_and_load.py`**

```python
import os
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from datetime import datetime

class ProjectSetup:
    """Setup project structure and load initial data"""
    
    def __init__(self, project_root="."):
        self.root = Path(project_root)
        
    def create_folder_structure(self):
        """Tạo cấu trúc thư mục"""
        folders = [
            "data/processed",
            "data/features",
            "data/interim",
            "models/tier1",
            "models/tier2",
            "models/tier3",
            "models/fallback",
            "models/semantic",
            "results/validation",
            "results/test",
            "results/comparisons",
            "config"
        ]
        
        for folder in folders:
            (self.root / folder).mkdir(parents=True, exist_ok=True)
            print(f"✓ Created: {folder}")
    
    def create_config(self):
        """Tạo file config.yaml"""
        config = {
            'project': {
                'name': 'LTV_ROAS_Prediction_V2.1',
                'version': '2.1.0',
                'created_date': datetime.now().strftime('%Y-%m-%d'),
                'target_mape': 0.05
            },
            
            'data': {
                'raw_path': 'data/raw',
                'processed_path': 'data/processed',
                'features_path': 'data/features',
                'train_months': ['T8', 'T9', 'T10', 'T11'],
                'test_months': ['T12'],
                'validation_month': 'T11'
            },
            
            'modeling': {
                'methods': ['hurdle', 'curve_fitting', 'ml_multiplier', 'lookalike'],
                'min_rows_tier1': 1000,
                'min_rows_tier2': 300,
                'min_rows_tier3': 100,
                'random_seed': 42
            },
            
            'tiers': {
                'tier1': {
                    'cv_threshold': 1.5,
                    'min_months': 3,
                    'target_mape': 0.04
                },
                'tier2': {
                    'cv_threshold': 2.5,
                    'min_months': 2,
                    'target_mape': 0.06
                },
                'tier3': {
                    'cv_threshold': float('inf'),
                    'min_months': 0,
                    'target_mape': 0.10
                }
            },
            
            'features': {
                'revenue_features': ['rev_sum', 'rev_max', 'rev_last', 'rev_d0_d1_ratio'],
                'engagement_features': ['retention_d1', 'avg_session_time_d1', 'avg_level_reached_d1'],
                'cpi_features': ['actual_cpi', 'cpi_vs_campaign_avg', 'cpi_quality_score']
            },
            
            'calibration': {
                'rolling_window_months': 2,
                'alpha': 0.3,
                'seasonal_adjustment': True
            },
            
            'semantic_matching': {
                'similarity_threshold': 0.6,
                'ngram_range': [2, 3],
                'max_features': 1000
            }
        }
        
        config_path = self.root / 'config' / 'config.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        
        print(f"✓ Created config: {config_path}")
        return config

class DataLoader:
    """Load and explore initial data"""
    
    def __init__(self, config):
        self.config = config
        self.data_path = Path(config['data']['raw_path'])
        
    def load_monthly_data(self, month):
        """Load data for a specific month"""
        file_path = self.data_path / f"data_{month}.csv"
        
        if not file_path.exists():
            print(f"⚠ Warning: {file_path} not found")
            return None
        
        print(f"Loading {month}...")
        df = pd.read_csv(file_path)
        df['month'] = month
        
        return df
    
    def load_all_data(self):
        """Load all months data"""
        all_months = self.config['data']['train_months'] + self.config['data']['test_months']
        
        dfs = []
        for month in all_months:
            df = self.load_monthly_data(month)
            if df is not None:
                dfs.append(df)
        
        if not dfs:
            raise ValueError("No data loaded!")
        
        combined = pd.concat(dfs, ignore_index=True)
        print(f"\n✓ Total rows loaded: {len(combined):,}")
        
        return combined
    
    def load_country_tiers(self):
        """Load country tier mapping"""
        file_path = self.data_path / "countries_tier.csv"
        
        if file_path.exists():
            df = pd.read_csv(file_path)
            print(f"✓ Loaded country tiers: {len(df)} countries")
            return df
        else:
            print("⚠ countries_tier.csv not found, will create default")
            return None
    
    def explore_data(self, df):
        """Khám phá dữ liệu ban đầu"""
        print("\n" + "="*60)
        print("DATA EXPLORATION")
        print("="*60)
        
        # Basic stats
        print(f"\n1. Shape: {df.shape}")
        print(f"   - Rows: {len(df):,}")
        print(f"   - Columns: {len(df.columns)}")
        
        # Columns
        print(f"\n2. Columns ({len(df.columns)}):")
        for col in df.columns:
            dtype = df[col].dtype
            null_pct = df[col].isna().sum() / len(df) * 100
            print(f"   - {col:30s} | {str(dtype):10s} | Null: {null_pct:.1f}%")
        
        # Key metrics
        if 'app_id' in df.columns and 'campaign' in df.columns:
            n_apps = df['app_id'].nunique()
            n_campaigns = df['campaign'].nunique()
            n_combos = df.groupby(['app_id', 'campaign']).ngroups
            
            print(f"\n3. Business Metrics:")
            print(f"   - Unique Apps: {n_apps:,}")
            print(f"   - Unique Campaigns: {n_campaigns:,}")
            print(f"   - App+Campaign Combos: {n_combos:,}")
        
        # Monthly distribution
        if 'month' in df.columns:
            month_dist = df['month'].value_counts().sort_index()
            print(f"\n4. Monthly Distribution:")
            for month, count in month_dist.items():
                pct = count / len(df) * 100
                print(f"   - {month}: {count:,} rows ({pct:.1f}%)")
        
        # Revenue stats (if exists)
        rev_cols = [c for c in df.columns if 'rev' in c.lower() or 'ltv' in c.lower()]
        if rev_cols:
            print(f"\n5. Revenue Columns:")
            for col in rev_cols[:5]:  # Show first 5
                mean_val = df[col].mean()
                median_val = df[col].median()
                max_val = df[col].max()
                print(f"   - {col}: Mean=${mean_val:.4f}, Median=${median_val:.4f}, Max=${max_val:.2f}")
        
        return {
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'n_apps': df['app_id'].nunique() if 'app_id' in df.columns else 0,
            'n_campaigns': df['campaign'].nunique() if 'campaign' in df.columns else 0,
            'n_combos': df.groupby(['app_id', 'campaign']).ngroups if {'app_id', 'campaign'}.issubset(df.columns) else 0
        }
    
    def save_overview(self, df, overview_stats):
        """Lưu overview summary"""
        # Save stats
        stats_df = pd.DataFrame([overview_stats])
        stats_path = Path('data/interim/data_overview.csv')
        stats_df.to_csv(stats_path, index=False)
        print(f"\n✓ Saved overview: {stats_path}")
        
        # Save sample data
        sample_path = Path('data/interim/sample_data.csv')
        df.head(1000).to_csv(sample_path, index=False)
        print(f"✓ Saved sample (1000 rows): {sample_path}")

def main():
    """Main execution"""
    print("="*60)
    print("STEP 1: ENVIRONMENT SETUP & DATA LOADING")
    print("="*60)
    
    # 1. Setup project structure
    print("\n[1/4] Creating folder structure...")
    setup = ProjectSetup()
    setup.create_folder_structure()
    
    # 2. Create config
    print("\n[2/4] Creating configuration...")
    config = setup.create_config()
    
    # 3. Load data
    print("\n[3/4] Loading data...")
    loader = DataLoader(config)
    
    # Load main data
    df = loader.load_all_data()
    
    # Load country tiers (optional)
    country_df = loader.load_country_tiers()
    
    # 4. Explore data
    print("\n[4/4] Exploring data...")
    overview_stats = loader.explore_data(df)
    
    # Save overview
    loader.save_overview(df, overview_stats)
    
    print("\n" + "="*60)
    print("✅ STEP 1 COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nNext Step: step02_data_cleaning.py")
    
    return df, config

if __name__ == "__main__":
    df, config = main()
```

---

## ▶️ EXECUTION

### Run Script

```bash
cd d:\WORKSPACE\new_technology
python scripts/step01_setup_and_load.py
```

### Expected Output

```
============================================================
STEP 1: ENVIRONMENT SETUP & DATA LOADING
============================================================

[1/4] Creating folder structure...
✓ Created: data/processed
✓ Created: data/features
✓ Created: data/interim
✓ Created: models/tier1
✓ Created: models/tier2
✓ Created: models/tier3
✓ Created: models/fallback
✓ Created: models/semantic
✓ Created: results/validation
✓ Created: results/test
✓ Created: results/comparisons
✓ Created: config

[2/4] Creating configuration...
✓ Created config: config/config.yaml

[3/4] Loading data...
Loading T8...
Loading T9...
Loading T10...
Loading T11...
Loading T12...

✓ Total rows loaded: 2,928,239

✓ Loaded country tiers: 195 countries

[4/4] Exploring data...

============================================================
DATA EXPLORATION
============================================================

1. Shape: (2928239, 25)
   - Rows: 2,928,239
   - Columns: 25

2. Columns (25):
   - install_date                | object     | Null: 0.0%
   - app_id                      | object     | Null: 0.0%
   - campaign                    | object     | Null: 0.0%
   - country                     | object     | Null: 0.0%
   - installs                    | int64      | Null: 0.0%
   - cost                        | float64    | Null: 0.0%
   - rev_d0                      | float64    | Null: 0.0%
   - rev_d1                      | float64    | Null: 0.0%
   ...

3. Business Metrics:
   - Unique Apps: 48
   - Unique Campaigns: 4,766
   - App+Campaign Combos: 4,800

4. Monthly Distribution:
   - T8: 585,648 rows (20.0%)
   - T9: 585,648 rows (20.0%)
   - T10: 585,005 rows (20.0%)
   - T11: 600,000 rows (20.5%)
   - T12: 571,938 rows (19.5%)

5. Revenue Columns:
   - rev_d0: Mean=$0.0123, Median=$0.0000, Max=$5.21
   - rev_d1: Mean=$0.0428, Median=$0.0000, Max=$12.50
   - ltv_d30: Mean=$0.1245, Median=$0.0000, Max=$45.32
   ...

✓ Saved overview: data/interim/data_overview.csv
✓ Saved sample (1000 rows): data/interim/sample_data.csv

============================================================
✅ STEP 1 COMPLETED SUCCESSFULLY!
============================================================

Next Step: step02_data_cleaning.py
```

---

## ✅ SUCCESS CRITERIA

- [x] All folders created successfully
- [x] `config/config.yaml` exists and valid
- [x] Data loaded: ~2.9M rows
- [x] Data overview saved to `data/interim/`
- [x] No errors during execution

---

## 🔍 VALIDATION

### Check Folder Structure
```bash
ls -R data models results config
```

### Check Config
```bash
cat config/config.yaml
```

### Check Data Overview
```python
import pandas as pd
overview = pd.read_csv('data/interim/data_overview.csv')
print(overview)
```

---

## ⚠️ TROUBLESHOOTING

### Issue 1: Missing CSV files
**Error:** `FileNotFoundError: data_T8.csv not found`  
**Solution:** 
- Kiểm tra path: `data/raw/data_T8.csv`
- Download missing files nếu cần

### Issue 2: Memory error
**Error:** `MemoryError when loading all data`  
**Solution:**
- Load từng tháng riêng lẻ
- Hoặc dùng `chunksize` parameter:
```python
df = pd.read_csv('file.csv', chunksize=100000)
```

### Issue 3: Encoding error
**Error:** `UnicodeDecodeError`  
**Solution:**
```python
df = pd.read_csv('file.csv', encoding='utf-8-sig')
```

---

## 📊 OUTPUT EXAMPLES

### config/config.yaml
```yaml
project:
  name: LTV_ROAS_Prediction_V2.1
  version: 2.1.0
  created_date: '2026-01-21'
  target_mape: 0.05

data:
  raw_path: data/raw
  processed_path: data/processed
  features_path: data/features
  train_months: [T8, T9, T10, T11]
  test_months: [T12]
  validation_month: T11
...
```

### data/interim/data_overview.csv
```csv
n_rows,n_cols,n_apps,n_campaigns,n_combos
2928239,25,48,4766,4800
```

---

## 🎯 NEXT STEP

➡️ **[Step 2: Data Cleaning & Validation](step02_data_cleaning.md)**

Clean data, xử lý missing values, outliers, và validate data quality.

---

**Estimated Time:** 2-4 hours  
**Difficulty:** ⭐ Easy  
**Status:** Ready to implement
