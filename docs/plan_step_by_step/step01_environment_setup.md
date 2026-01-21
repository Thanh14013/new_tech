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
- `data/raw/jan.csv` (Tháng 1/2025)
- `data/raw/feb.csv` (Tháng 2/2025)
- `data/raw/mar.csv` (Tháng 3/2025)
- `data/raw/apr.csv` (Tháng 4/2025)
- `data/raw/may.csv` (Tháng 5/2025)
- `data/raw/jun.csv` (Tháng 6/2025)
- `data/raw/jul.csv` (Tháng 7/2025)
- `data/raw/data_T8.csv` (Tháng 8/2025)
- `data/raw/data_T9.csv` (Tháng 9/2025)
- `data/raw/data_T10.csv` (Tháng 10/2025)
- `data/raw/data_T11.csv` (Tháng 11/2025)
- `data/raw/data_T12.csv` (Tháng 12/2025)
- **`data/raw/wool/data_wool_D30_T11_T12.csv`** (Wool app - D30 data, high-revenue)
- **`data/raw/wool/data_wool_D60_T7_T10.csv`** (Wool app - D60 data, high-revenue)
- `data/raw/countries_tier.csv` (Phân loại quốc gia)

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
            },all_months': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12'],
                'train_months': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10'],
                'validation_month': 'M11',
                'test_months': ['M12']
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
        # Month mapping: M1->jan, M2->feb, ..., M8->T8, ...
        self.month_files = {
            'M1': 'jan.csv',
            'M2': 'feb.csv',
            'M3': 'mar.csv',
            'M4': 'apr.csv',
            'M5': 'may.csv',
            'M6': 'jun.csv',
            'M7': 'jul.csv',
            'M8': 'data_T8.csv',
            'M9': 'data_T9.csv',
            'M10': 'data_T10.csv',
            'M11': 'data_T11.csv',
            'M12': 'data_T12.csv'
        }
        12 months data (Jan-Dec 2025)"""
        all_months = self.config['data']['all_months']  # M1-M12
        
        dfs = []
        for month in all_months:
            df = self.load_monthly_data(month)
            if df is not None:
                dfs.append(df)
        
        if not dfs:
            raise ValueError("No data loaded!")
        
        combined = pd.concat(dfs, ignore_index=True)
        print(f"\n✓ Total rows loaded: {len(combined):,}")
        print(f"✓ Months loaded: {len(dfs)}/12
            print(f"⚠ Warning: {file_path} not found")
            return None
        
        print(f"Loading {month} ({filename})...")
        df = pd.read_csv(file_path)
        df['month'] = month  # Standardize to M1-M12le_path)
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
    
    def load_wool_data(self):
        """Load special wool app data (high-revenue app)"""
        wool_path = self.data_path / "wool"
        
        if not wool_path.exists():
            print("⚠ Wool folder not found, skipping wool data")
            return None
        
        print("\n📦 Loading WOOL data (high-revenue app)...")
        
        wool_dfs = []
        
        # Load D30 data (T11, T12)
        d30_file = wool_path / "data_wool_D30_T11_T12.csv"
        if d30_file.exists():
            df_d30 = pd.read_csv(d30_file)
            print(f"  ✓ Wool D30 (T11-T12): {len(df_d30):,} rows")
            
            # Add month info based on data
            # Assuming có column để identify month, hoặc split 50/50
            if 'month' not in df_d30.columns:
                # Split half-half for T11 and T12
                split_idx = len(df_d30) // 2
                df_d30.loc[:split_idx, 'month'] = 'M11'
                df_d30.loc[split_idx:, 'month'] = 'M12'
            
            df_d30['wool_data_type'] = 'D30'
            wool_dfs.append(df_d30)
        
        # Load D60 data (T7-T10)
        d60_file = wool_path / "data_wool_D60_T7_T10.csv"
        if d60_file.exists():
            df_d60 = pd.read_csv(d60_file)
            print(f"  ✓ Wool D60 (T7-T10): {len(df_d60):,} rows")
            
            # Map to months M7-M10
            if 'month' not in df_d60.columns:
                # Split into 4 parts for M7, M8, M9, M10
                n_rows = len(df_d60)
                chunk_size = n_rows // 4
                
                df_d60.loc[:chunk_size, 'month'] = 'M7'
                df_d60.loc[chunk_size:chunk_size*2, 'month'] = 'M8'
                df_d60.loc[chunk_size*2:chunk_size*3, 'month'] = 'M9'
                df_d60.loc[chunk_size*3:, 'month'] = 'M10'
            
            df_d60['wool_data_type'] = 'D60'
            wool_dfs.append(df_d60)
        
        if wool_dfs:
            wool_combined = pd.concat(wool_dfs, ignore_index=True)
            print(f"\n  ✓ Total Wool rows: {len(wool_combined):,}")
            return wool_combined
        else:
            print("  ⚠ No wool data files found")
            return None
    
    def explore_data(self, df):
        """Khám phá dữ liệu ban đầu"""
        print("\n" + "="*60)
        print("DATA EXPLORATION")
        print("="*60)
        
        # Basic stats (2025):")
            for month, count in month_dist.items():
                pct = count / len(df) * 100
                month_name = {
                    'M1': 'Jan', 'M2': 'Feb', 'M3': 'Mar', 'M4': 'Apr',
                    'M5': 'May', 'M6': 'Jun', 'M7': 'Jul', 'M8': 'Aug',
                    'M9': 'Sep', 'M10': 'Oct', 'M11': 'Nov', 'M12': 'Dec'
                }.get(month, month)
                print(f"   - {month} ({month_name})f.columns)}")
        
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
    
    # Load wool data (high-revenue app)
    wool_df = loader.load_wool_data()
    
    # Merge wool data with main data if available
    if wool_df is not None:
        print("\n🔗 Merging WOOL data with main dataset...")
        
        # Ensure wool has app_id
        if 'app_id' not in wool_df.columns:
            wool_df['app_id'] = 'wool'
        
        # Merge (append wool rows)
        df = pd.concat([df, wool_df], ignore_index=True)
        print(f"  ✓ Total rows after wool merge: {len(df):,}")
    
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

```bashM1 (jan.csv)...
Loading M2 (feb.csv)...
Loading M3 (mar.csv)...
Loading M4 (apr.csv)...
Loading M5 (may.csv)...
Loading M6 (jun.csv)...
Loading M7 (jul.csv)...
Loading M8 (data_T8.csv)...
Loading M9 (data_T9.csv)...
Loading M10 (data_T10.csv)...
Loading M11 (data_T11.csv)...
Loading M12 (data_T12.csv)...

✓ Total rows loaded: 5,856,478
✓ Months loaded: 12/12
```
===========5856478, 25)
   - Rows: 5,856,478SETUP & DATA LOADING
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
 (2025):
   - M1 (Jan): 488,040 rows (8.3%)
   - M2 (Feb): 488,040 rows (8.3%)
   - M3 (Mar): 488,040 rows (8.3%)
   - M4 (Apr): 488,040 rows (8.3%)
   - M5 (May): 488,040 rows (8.3%)
   - M6 (Jun): 488,040 rows (8.3%)
   - M7 (Jul): 488,040 rows (8.3%)
   - M8 (Aug): 488,040 rows (8.3%)
   - M9 (Sep): 488,040 rows (8.3%)
   - M10 (Oct): 488,040 rows (8.3%)
   - M11 (Nov): 488,040 rows (8.3%)
   - M12 (Dec): 488,038 rows (8.3
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
5856478ion:** 
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
