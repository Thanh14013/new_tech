# Step 2: Data Cleaning & Validation
## Làm Sạch và Kiểm Tra Chất Lượng Dữ Liệu

**Thời gian:** 1 ngày  
**Độ khó:** ⭐⭐ Trung bình  
**Prerequisites:** Step 1 completed  

---

## 🎯 MỤC TIÊU

1. Xử lý missing values và outliers
2. Chuẩn hóa kiểu dữ liệu
3. Validate business logic (rev_d0 ≤ rev_d1 ≤ ltv_d30)
4. Remove duplicates và data quality issues
5. Tạo data quality report

---

## 📥 INPUT

- `data/raw/jan.csv`, `data/raw/feb.csv`, ..., `data/raw/jul.csv` (Tháng 1-7)
- `data/raw/data_T8.csv`, `data/raw/data_T9.csv`, ..., `data/raw/data_T12.csv` (Tháng 8-12)
- `config/config.yaml` (Config)
- `data/interim/data_overview.csv` (Overview từ Step 1)

---

## 📤 OUTPUT

- `data/processed/clean_data_M1.csv` → `clean_data_M12.csv` (Clean data theo tháng)
- `data/processed/clean_data_all.csv` (Tất cả 12 tháng)
- `results/data_quality_report.html` (HTML report)
- `results/step02_cleaning_summary.csv` (Summary stats)

---

## 🔧 IMPLEMENTATION

### File: `scripts/step02_data_cleaning.py`

```python
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataCleaner:
    """Clean and validate data"""
    
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Month file mapping
        self.month_files = {
            'M1': 'jan.csv', 'M2': 'feb.csv', 'M3': 'mar.csv', 'M4': 'apr.csv',
            'M5': 'may.csv', 'M6': 'jun.csv', 'M7': 'jul.csv',
            'M8': 'data_T8.csv', 'M9': 'data_T9.csv', 'M10': 'data_T10.csv',
            'M11': 'data_T11.csv', 'M12': 'data_T12.csv'
        }
        
        self.issues = {
            'missing_values': [], (M1-M12)"""
        filename = self.month_files.get(month)
        if filename is None:
            raise ValueError(f"Unknown month: {month}")
        
        path = Path(self.config['data']['raw_path']) / filename
            'duplicates': [],
            'logic_violations': [],
            'invalid_types': []
        }
        
    def load_raw_data(self, month):
        """Load raw data for a month"""
        path = Path(self.config['data']['raw_path']) / f"data_{month}.csv"
        return pd.read_csv(path)
    
    def check_missing_values(self, df, month):
        """Kiểm tra missing values"""
        print(f"\n[{month}] Checking missing values...")
        
        missing_summary = df.isnull().sum()
        missing_pct = (missing_summary / len(df) * 100).round(2)
        
        missing_df = pd.DataFrame({
            'column': missing_summary.index,
            'missing_count': missing_summary.values,
            'missing_pct': missing_pct.values
        })
        
        missing_df = missing_df[missing_df['missing_count'] > 0].sort_values(
            'missing_pct', ascending=False
        )
        
        if len(missing_df) > 0:
            print(f"  ⚠ Found {len(missing_df)} columns with missing values:")
            for _, row in missing_df.iterrows():
                print(f"    - {row['column']}: {row['missing_count']:,} ({row['missing_pct']:.1f}%)")
            
            self.issues['missing_values'].append({
                'month': month,
                'details': missing_df.to_dict('records')
            })
        else:
            print("  ✓ No missing values")
        
        return missing_df
    
    def handle_missing_values(self, df):
        """Xử lý missing values"""
        df = df.copy()
        
        # Revenue columns: fill 0
        rev_cols = [c for c in df.columns if 'rev' in c.lower() or 'ltv' in c.lower()]
        for col in rev_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(0)
        
        # Engagement columns: fill median
        engagement_cols = ['retention_d1', 'avg_session_time_d1', 'avg_level_reached_d1']
        for col in engagement_cols:
            if col in df.columns and df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        # Categorical: fill 'unknown'
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna('unknown')
        
        # Numeric: fill median
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        return df
    
    def detect_outliers(self, df, column, method='iqr', threshold=3):
        """Phát hiện outliers bằng IQR hoặc Z-score"""
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            outliers = z_scores > threshold
        
        return outliers
    
    def remove_outliers(self, df, month):
        """Remove outliers từ revenue columns"""
        print(f"\n[{month}] Removing outliers...")
        
        df_clean = df.copy()
        outlier_summary = []
        
        # Revenue columns
        rev_cols = ['rev_d0', 'rev_d1', 'ltv_d7', 'ltv_d30', 'ltv_d60']
        
        for col in rev_cols:
            if col not in df.columns:
                continue
            
            # Tìm outliers (IQR với threshold=5 - liberal để giữ high spenders)
            outliers = self.detect_outliers(df_clean, col, method='iqr', threshold=5)
            n_outliers = outliers.sum()
            
            if n_outliers > 0:
                outlier_pct = n_outliers / len(df_clean) * 100
                max_outlier_val = df_clean.loc[outliers, col].max()
                
                print(f"  - {col}: {n_outliers:,} outliers ({outlier_pct:.2f}%), max={max_outlier_val:.2f}")
                
                # Winsorize thay vì remove (cap ở 99th percentile)
                p99 = df_clean[col].quantile(0.99)
                df_clean.loc[df_clean[col] > p99, col] = p99
                
                outlier_summary.append({
                    'column': col,
                    'n_outliers': n_outliers,
                    'outlier_pct': round(outlier_pct, 2),
                    'max_value': round(max_outlier_val, 2),
                    'capped_at': round(p99, 2)
                })
        
        if outlier_summary:
            self.issues['outliers'].append({
                'month': month,
                'details': outlier_summary
            })
        
        return df_clean
    
    def validate_business_logic(self, df, month):
        """Validate business logic constraints"""
        print(f"\n[{month}] Validating business logic...")
        
        violations = []
        
        # Rule 1: rev_d0 <= rev_d1
        if 'rev_d0' in df.columns and 'rev_d1' in df.columns:
            mask = df['rev_d1'] < df['rev_d0']
            n_violations = mask.sum()
            if n_violations > 0:
                pct = n_violations / len(df) * 100
                print(f"  ⚠ rev_d1 < rev_d0: {n_violations:,} rows ({pct:.2f}%)")
                violations.append({
                    'rule': 'rev_d1 >= rev_d0',
                    'violations': n_violations,
                    'pct': round(pct, 2)
                })
                # Fix: set rev_d0 = rev_d1
                df.loc[mask, 'rev_d0'] = df.loc[mask, 'rev_d1']
        
        # Rule 2: ltv_d30 >= rev_d1
        if 'ltv_d30' in df.columns and 'rev_d1' in df.columns:
            mask = df['ltv_d30'] < df['rev_d1']
            n_violations = mask.sum()
            if n_violations > 0:
                pct = n_violations / len(df) * 100
                print(f"  ⚠ ltv_d30 < rev_d1: {n_violations:,} rows ({pct:.2f}%)")
                violations.append({
                    'rule': 'ltv_d30 >= rev_d1',
                    'violations': n_violations,
                    'pct': round(pct, 2)
                })
                # Fix: set rev_d1 = ltv_d30
                df.loc[mask, 'rev_d1'] = df.loc[mask, 'ltv_d30']
        
        # Rule 3: installs > 0
        if 'installs' in df.columns:
            mask = df['installs'] <= 0
            n_violations = mask.sum()
            if n_violations > 0:
                pct = n_violations / len(df) * 100
                print(f"  ⚠ installs <= 0: {n_violations:,} rows ({pct:.2f}%)")
                violations.append({
                    'rule': 'installs > 0',
                    'violations': n_violations,
                    'pct': round(pct, 2)
                })
                # Remove these rows
                df = df[df['installs'] > 0]
        
        # Rule 4: cost >= 0
        if 'cost' in df.columns:
            mask = df['cost'] < 0
            n_violations = mask.sum()
            if n_violations > 0:
                pct = n_violations / len(df) * 100
                print(f"  ⚠ cost < 0: {n_violations:,} rows ({pct:.2f}%)")
                violations.append({
                    'rule': 'cost >= 0',
                    'violations': n_violations,
                    'pct': round(pct, 2)
                })
                df.loc[mask, 'cost'] = 0
        
        if violations:
            self.issues['logic_violations'].append({
                'month': month,
                'details': violations
            })
        else:
            print("  ✓ All business logic rules passed")
        
        return df
    
    def remove_duplicates(self, df, month):
        """Remove duplicate rows"""
        print(f"\n[{month}] Checking duplicates...")
        
        initial_rows = len(df)
        
        # Identify duplicates based on key columns
        key_cols = ['install_date', 'app_id', 'campaign', 'country']
        
        duplicates = df.duplicated(subset=key_cols, keep='first')
        n_duplicates = duplicates.sum()
        
        if n_duplicates > 0:
            pct = n_duplicates / initial_rows * 100
            print(f"  ⚠ Found {n_duplicates:,} duplicates ({pct:.2f}%)")
            
            df_clean = df[~duplicates].copy()
            
            self.issues['duplicates'].append({
                'month': month,
                'n_duplicates': n_duplicates,
                'pct': round(pct, 2)
            })
        else:
            print("  ✓ No duplicates found")
            df_clean = df.copy()
        
        return df_clean
    
    def standardize_dtypes(self, df):
        """Chuẩn hóa kiểu dữ liệu"""
        df = df.copy()
        
        # Date columns
        if 'install_date' in df.columns:
            df['install_date'] = pd.to_datetime(df['install_date'], errors='coerce')
        
        # Categorical
        categorical_cols = ['app_id', 'campaign', 'country', 'platform']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        # Numeric
        numeric_cols = ['installs', 'cost', 'rev_d0', 'rev_d1', 'ltv_d30', 'ltv_d60']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def clean_month(self, month):
        """Clean data for one month"""
        print(f"\n{'='*60}")
        print(f"CLEANING DATA: {month}")
        print(f"{'='*60}")
        
        # 1. Load
        df = self.load_raw_data(month)
        print(f"Loaded: {len(df):,} rows")
        
        # 2. Check missing
        self.check_missing_values(df, month)
        
        # 3. Handle missing
        df = self.handle_missing_values(df)
        
        # 4. Standardize dtypes
        df = self.standardize_dtypes(df)
        
        # 5. Remove outliers
        df = self.remove_outliers(df, month)
        
        # 6. Validate logic
        df = self.validate_business_logic(df, month)
        
        # 7. Remove duplicates
        df = self.remove_duplicates(df, month)
        
        print(f"\n✓ Clean data: {len(df):,} rows")
        
        return df
    
    def save_clean_data(self, df, month):
        """Save cleaned data"""
        output_path = Path(self.config['data']['processed_path']) / f"clean_data_{month}.csv"
        df.to_csv(output_path, index=False)
        print(f"✓ Saved: {output_path}")
    
    def generate_report(self):
        """Generate HTML quality report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Quality Report - Step 2</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th {{ background-color: #4CAF50; color: white; padding: 10px; text-align: left; }}
                td {{ border: 1px solid #ddd; padding: 8px; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .warning {{ color: #ff9800; }}
                .success {{ color: #4CAF50; }}
            </style>
        </head>
        <body>
            <h1>Data Quality Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>1. Missing Values</h2>
        """
        
        if self.issues['missing_values']:
            for issue in self.issues['missing_values']:
                html += f"<h3>Month: {issue['month']}</h3>"
                html += "<table><tr><th>Column</th><th>Missing Count</th><th>Missing %</th></tr>"
                for detail in issue['details']:
                    html += f"<tr><td>{detail['column']}</td><td>{detail['missing_count']}</td><td>{detail['missing_pct']}%</td></tr>"
                html += "</table>"
        else:
            html += "<p class='success'>✓ No missing values found</p>"
        
        html += "<h2>2. Outliers (Winsorized)</h2>"
        if self.issues['outliers']:
            for issue in self.issues['outliers']:
                html += f"<h3>Month: {issue['month']}</h3>"
                html += "<table><tr><th>Column</th><th>Outliers</th><th>%</th><th>Max Value</th><th>Capped At</th></tr>"
                for detail in issue['details']:
                    html += f"<tr><td>{detail['column']}</td><td>{detail['n_outliers']}</td><td>{detail['outlier_pct']}%</td><td>${detail['max_value']}</td><td>${detail['capped_at']}</td></tr>"
                html += "</table>"
        else:
            html += "<p class='success'>✓ No outliers detected</p>"
        
        html += "<h2>3. Business Logic Violations</h2>"
        if self.issues['logic_violations']:
            for issue in self.issues['logic_violations']:
                html += f"<h3>Month: {issue['month']}</h3>"
                html += "<table><tr><th>Rule</th><th>Violations</th><th>%</th></tr>"
                for detail in issue['details']:
                    html += f"<tr><td>{detail['rule']}</td><td>{detail['violations']}</td><td>{detail['pct']}%</td></tr>"
                html += "</table>"
        else:
            html += "<p class='success'>✓ No logic violations</p>"
        
        html += "<h2>4. Duplicates</h2>"
        if self.issues['duplicates']:
            for issue in self.issues['duplicates']:
                html += f"<p class='warning'>Month {issue['month']}: {issue['n_duplicates']} duplicates ({issue['pct']}%) removed</p>"
        else:
            html += "<p class='success'>✓ No duplicates found</p>"
        
        html += "</body></html>"
        
        # Save report
        report_path = Path('results/data_quality_report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"\n✓ Quality report saved: {report_path}")

def main():
    """Main execution"""
    print("="*60)
    print("STEP 2: DATA CLEANING & VALIDATION")
    print("="*60)
    
    # Initialize cleaner
    cleaner = DataCleaner()
    
    # Get months from config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    all_months = config['data']['train_months'] + config['data']['test_months']
    
    # Clean each month
    clean_dfs = []
    for month in all_months:
        df_clean = cleaner.clean_month(month)
        cleaner.save_clean_data(df_clean, month)
        clean_dfs.append(df_clean)
    
    # Combine all
    print(f"\n{'='*60}")
    print("COMBINING ALL MONTHS")
    print(f"{'='*60}")
    
    df_all = pd.concat(clean_dfs, ignore_index=True)
    print(f"Total rows: {len(df_all):,}")
    
    # Save combined
    output_path = Path(config['data']['processed_path']) / 'clean_data_all.csv'
    df_all.to_csv(output_path, index=False)
    print(f"✓ Saved combined data: {output_path}")
    
    # Generate report
    cleaner.generate_report()
    
    # Save summary
    summary = pd.DataFrame([{
        'total_rows': len(df_all),
        'n_apps': df_all['app_id'].nunique(),
        'n_campaigns': df_all['campaign'].nunique(),
        'n_combos': df_all.groupby(['app_id', 'campaign']).ngroups,
        'missing_issues': len(cleaner.issues['missing_values']),
        'outlier_issues': len(cleaner.issues['outliers']),
        'logic_violations': len(cleaner.issues['logic_violations']),
        'duplicates': len(cleaner.issues['duplicates'])
    }])
    
    summary.to_csv('results/step02_cleaning_summary.csv', index=False)
    
    print("\n" + "="*60)
    print("✅ STEP 2 COMPLETED!")
    print("="*60)
    print("\nNext Step: step03_tier_classification.py")

if __name__ == "__main__":
    main()
```

---

## ✅ SUCCESS CRITERIA

- [x] All months cleaned: `clean_data_T*.csv` created
- [x] Missing values < 1%
- [x] No business logic violations remaining
- [x] Duplicates removed
- [x] HTML quality report generated

---

## 🎯 NEXT STEP

➡️ **[Step 3: Tier Classification](step03_tier_classification.md)**

---

**Estimated Time:** 4-8 hours  
**Difficulty:** ⭐⭐ Medium
