"""
Step 2: Data Cleaning & Validation
===================================
Làm sạch và kiểm tra chất lượng dữ liệu

Author: LTV Prediction System V2.1
Date: 2026-01-21
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DataCleaner:
    """Clean and validate data for LTV prediction"""
    
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Month file mapping (same as Step 01)
        self.month_files = {
            'M1': 'jan.csv', 'M2': 'feb.csv', 'M3': 'mar.csv', 'M4': 'apr.csv',
            'M5': 'may.csv', 'M6': 'jun.csv', 'M7': 'jul.csv',
            'M8': 'data_T8.csv', 'M9': 'data_T9.csv', 'M10': 'data_T10.csv',
            'M11': 'data_T11.csv', 'M12': 'data_T12.csv'
        }
        
        self.month_names = {
            'M1': 'Jan', 'M2': 'Feb', 'M3': 'Mar', 'M4': 'Apr',
            'M5': 'May', 'M6': 'Jun', 'M7': 'Jul', 'M8': 'Aug',
            'M9': 'Sep', 'M10': 'Oct', 'M11': 'Nov', 'M12': 'Dec'
        }
        
        # Track issues
        self.issues = {
            'missing_values': [],
            'outliers': [],
            'logic_violations': [],
            'duplicates': []
        }
        
        # Wool data cache
        self.wool_data = None
    
    def load_wool_data_once(self):
        """Load wool data once and cache it"""
        if self.wool_data is not None:
            return self.wool_data
        
        wool_path = Path(self.config['data']['raw_path']) / "wool"
        
        if not wool_path.exists():
            return None
        
        wool_dfs = []
        
        # Load D30 data (M11, M12)
        d30_file = wool_path / "data_wool_D30_T11_T12.csv"
        if d30_file.exists():
            try:
                df_d30 = pd.read_csv(d30_file)
                split_idx = len(df_d30) // 2
                df_d30['month'] = 'M11'
                df_d30.loc[split_idx:, 'month'] = 'M12'
                df_d30['wool_data_type'] = 'D30'
                wool_dfs.append(df_d30)
            except Exception as e:
                print(f"  ⚠ Error loading wool D30: {e}")
        
        # Load D60 data (M7-M10)
        d60_file = wool_path / "data_wool_D60_T7_T10.csv"
        if d60_file.exists():
            try:
                df_d60 = pd.read_csv(d60_file)
                n_rows = len(df_d60)
                chunk_size = n_rows // 4
                
                df_d60['month'] = 'M7'
                df_d60.loc[chunk_size:chunk_size*2, 'month'] = 'M8'
                df_d60.loc[chunk_size*2:chunk_size*3, 'month'] = 'M9'
                df_d60.loc[chunk_size*3:, 'month'] = 'M10'
                df_d60['wool_data_type'] = 'D60'
                wool_dfs.append(df_d60)
            except Exception as e:
                print(f"  ⚠ Error loading wool D60: {e}")
        
        if wool_dfs:
            self.wool_data = pd.concat(wool_dfs, ignore_index=True)
            return self.wool_data
        
        return None
    
    def load_monthly_data(self, month):
        """Load raw data for a specific month (M1-M12)"""
        filename = self.month_files.get(month)
        if filename is None:
            raise ValueError(f"Unknown month: {month}")
        
        file_path = Path(self.config['data']['raw_path']) / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load main data
        df = pd.read_csv(file_path)
        df['month'] = month
        
        # Add wool data for relevant months (M7-M12)
        if month in ['M7', 'M8', 'M9', 'M10', 'M11', 'M12']:
            wool_data = self.load_wool_data_once()
            if wool_data is not None:
                wool_month = wool_data[wool_data['month'] == month]
                if len(wool_month) > 0:
                    # Ensure wool has app_id
                    if 'app_id' not in wool_month.columns:
                        wool_month = wool_month.copy()
                        wool_month['app_id'] = 'wool'
                    
                    df = pd.concat([df, wool_month], ignore_index=True)
        
        return df
    
    def check_missing_values(self, df, month):
        """Kiểm tra missing values"""
        print(f"\n  [Missing Values Check]")
        
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
            print(f"    ⚠ Found {len(missing_df)} columns with missing values")
            # Show top 5
            for idx, row in missing_df.head(5).iterrows():
                print(f"      - {row['column']}: {row['missing_count']:,} ({row['missing_pct']:.1f}%)")
            
            if len(missing_df) > 5:
                print(f"      ... and {len(missing_df) - 5} more columns")
            
            self.issues['missing_values'].append({
                'month': month,
                'n_columns': len(missing_df),
                'details': missing_df.to_dict('records')[:10]  # Save top 10
            })
        else:
            print("    ✓ No missing values")
        
        return missing_df
    
    def handle_missing_values(self, df):
        """Xử lý missing values"""
        df = df.copy()
        
        # Revenue columns: fill 0 (no revenue = 0)
        rev_cols = [c for c in df.columns if 'revenue' in c.lower()]
        for col in rev_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(0)
        
        # Engagement columns: fill 0 (no users = 0)
        engagement_cols = [c for c in df.columns if 'unique_users' in c.lower()]
        for col in engagement_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(0)
        
        # Cost: fill 0 or median based on campaign
        if 'cost' in df.columns and df['cost'].isnull().sum() > 0:
            df['cost'] = df['cost'].fillna(0)
        
        # Categorical: fill 'unknown'
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'month' and df[col].isnull().sum() > 0:
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
        if column not in df.columns:
            return pd.Series([False] * len(df))
        
        if df[column].isnull().all():
            return pd.Series([False] * len(df))
        
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        
        elif method == 'zscore':
            mean = df[column].mean()
            std = df[column].std()
            if std == 0:
                return pd.Series([False] * len(df))
            z_scores = np.abs((df[column] - mean) / std)
            outliers = z_scores > threshold
        
        return outliers
    
    def remove_outliers(self, df, month):
        """Remove/winsorize outliers from revenue columns"""
        print(f"\n  [Outlier Detection & Winsorization]")
        
        df_clean = df.copy()
        outlier_summary = []
        
        # Revenue columns to check (use first 30 days as representative)
        rev_cols = ['revenue_d0', 'revenue_d1', 'revenue_d7', 'revenue_d14', 'revenue_d30']
        
        for col in rev_cols:
            if col not in df.columns:
                continue
            
            # Skip if all null
            if df_clean[col].isnull().all():
                continue
            
            # Find outliers (IQR with threshold=5 - liberal to keep high spenders)
            outliers = self.detect_outliers(df_clean, col, method='iqr', threshold=5)
            n_outliers = outliers.sum()
            
            if n_outliers > 0:
                outlier_pct = n_outliers / len(df_clean) * 100
                max_outlier_val = df_clean.loc[outliers, col].max()
                
                # Winsorize at 99th percentile (cap extreme values)
                p99 = df_clean[col].quantile(0.99)
                df_clean.loc[df_clean[col] > p99, col] = p99
                
                outlier_summary.append({
                    'column': col,
                    'n_outliers': int(n_outliers),
                    'outlier_pct': round(float(outlier_pct), 2),
                    'max_value': round(float(max_outlier_val), 2),
                    'capped_at': round(float(p99), 2)
                })
        
        if outlier_summary:
            print(f"    ⚠ Winsorized {len(outlier_summary)} revenue columns")
            for item in outlier_summary[:3]:
                print(f"      - {item['column']}: {item['n_outliers']:,} outliers capped at ${item['capped_at']:.2f}")
            
            self.issues['outliers'].append({
                'month': month,
                'details': outlier_summary
            })
        else:
            print("    ✓ No significant outliers detected")
        
        return df_clean
    
    def validate_business_logic(self, df, month):
        """Validate business logic constraints
        
        NOTE: Revenue columns are DISCRETE (not cumulative)!
        - revenue_d0 = revenue only on day 0
        - revenue_d1 = revenue only on day 1
        - revenue_d7 = revenue only on day 7
        So revenue_d1 < revenue_d0 is normal and valid!
        """
        print(f"\n  [Business Logic Validation]")
        
        df = df.copy()
        violations = []
        
        # Rule 1: All revenue values >= 0
        rev_cols = [c for c in df.columns if 'revenue' in c.lower()]
        for col in rev_cols:
            if col in df.columns:
                mask = df[col] < 0
                n_violations = mask.sum()
                if n_violations > 0:
                    pct = n_violations / len(df) * 100
                    violations.append({
                        'rule': f'{col} >= 0',
                        'violations': int(n_violations),
                        'pct': round(float(pct), 2)
                    })
                    # Fix: set negative to 0
                    df.loc[mask, col] = 0
                    print(f"    ⚠ Fixed {n_violations:,} negative values in {col}")
        
        # Rule 2: installs > 0
        if 'installs' in df.columns:
            mask = df['installs'] <= 0
            n_violations = mask.sum()
            if n_violations > 0:
                pct = n_violations / len(df) * 100
                violations.append({
                    'rule': 'installs > 0',
                    'violations': int(n_violations),
                    'pct': round(float(pct), 2)
                })
                # Remove these rows
                df = df[df['installs'] > 0]
                print(f"    ⚠ Removed {n_violations:,} rows with installs <= 0")
        
        # Rule 3: cost >= 0
        if 'cost' in df.columns:
            mask = df['cost'] < 0
            n_violations = mask.sum()
            if n_violations > 0:
                pct = n_violations / len(df) * 100
                violations.append({
                    'rule': 'cost >= 0',
                    'violations': int(n_violations),
                    'pct': round(float(pct), 2)
                })
                df.loc[mask, 'cost'] = 0
                print(f"    ⚠ Fixed {n_violations:,} negative cost values")
        
        if violations:
            print(f"    ⚠ Found and fixed {len(violations)} types of violations")
            
            self.issues['logic_violations'].append({
                'month': month,
                'details': violations
            })
        else:
            print("    ✓ All business logic rules passed")
        
        return df
    
    def remove_duplicates(self, df, month):
        """Remove duplicate rows"""
        print(f"\n  [Duplicate Detection]")
        
        initial_rows = len(df)
        
        # Identify duplicates based on key columns
        key_cols = ['install_date', 'app_id', 'campaign', 'geo']
        # Only use columns that exist
        key_cols = [c for c in key_cols if c in df.columns]
        
        if not key_cols:
            print("    ⚠ Cannot check duplicates (missing key columns)")
            return df
        
        duplicates = df.duplicated(subset=key_cols, keep='first')
        n_duplicates = duplicates.sum()
        
        if n_duplicates > 0:
            pct = n_duplicates / initial_rows * 100
            print(f"    ⚠ Found and removed {n_duplicates:,} duplicates ({pct:.2f}%)")
            
            df_clean = df[~duplicates].copy()
            
            self.issues['duplicates'].append({
                'month': month,
                'n_duplicates': int(n_duplicates),
                'pct': round(float(pct), 2)
            })
        else:
            print("    ✓ No duplicates found")
            df_clean = df.copy()
        
        return df_clean
    
    def standardize_dtypes(self, df):
        """Chuẩn hóa kiểu dữ liệu"""
        df = df.copy()
        
        # Date columns
        if 'install_date' in df.columns:
            df['install_date'] = pd.to_datetime(df['install_date'], errors='coerce')
        
        # Categorical
        categorical_cols = ['app_id', 'campaign', 'geo', 'media_source', 'month']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        # Numeric - installs should be int, others float
        if 'installs' in df.columns:
            df['installs'] = pd.to_numeric(df['installs'], errors='coerce').fillna(0).astype(int)
        
        # Cost and revenue columns should be float
        numeric_cols = ['cost'] + [c for c in df.columns if 'revenue' in c.lower()]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
    
    def clean_month(self, month):
        """Clean data for one month"""
        print(f"\n{'='*70}")
        print(f"CLEANING: {month} ({self.month_names[month]} 2025)")
        print(f"{'='*70}")
        
        # 1. Load
        print(f"  [Loading Data]")
        df = self.load_monthly_data(month)
        print(f"    ✓ Loaded: {len(df):,} rows, {len(df.columns)} columns")
        
        # 2. Check missing
        self.check_missing_values(df, month)
        
        # 3. Handle missing
        print(f"\n  [Handling Missing Values]")
        df = self.handle_missing_values(df)
        print(f"    ✓ Missing values handled")
        
        # 4. Standardize dtypes
        print(f"\n  [Standardizing Data Types]")
        df = self.standardize_dtypes(df)
        print(f"    ✓ Data types standardized")
        
        # 5. Remove outliers
        df = self.remove_outliers(df, month)
        
        # 6. Validate logic
        df = self.validate_business_logic(df, month)
        
        # 7. Remove duplicates
        df = self.remove_duplicates(df, month)
        
        print(f"\n  ✅ Clean data: {len(df):,} rows")
        
        return df
    
    def save_clean_data(self, df, month):
        """Save cleaned data"""
        output_path = Path(self.config['data']['processed_path']) / f"clean_data_{month}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"  💾 Saved: {output_path}")
    
    def generate_report(self):
        """Generate HTML quality report"""
        print(f"\n{'='*70}")
        print("GENERATING QUALITY REPORT")
        print(f"{'='*70}")
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Data Quality Report - Step 2</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #bdc3c7; padding-bottom: 5px; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th {{ background-color: #3498db; color: white; padding: 12px; text-align: left; }}
        td {{ border: 1px solid #ddd; padding: 10px; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        tr:hover {{ background-color: #f0f0f0; }}
        .warning {{ color: #e67e22; font-weight: bold; }}
        .success {{ color: #27ae60; font-weight: bold; }}
        .error {{ color: #e74c3c; font-weight: bold; }}
        .stat-box {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        .timestamp {{ color: #95a5a6; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Data Quality Report - Step 2</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="stat-box">
            <strong>Summary:</strong> Data cleaning completed for 12 months (Jan-Dec 2025)
        </div>
"""
        
        # Missing Values
        html += "<h2>1. Missing Values</h2>"
        if self.issues['missing_values']:
            for issue in self.issues['missing_values']:
                html += f"<h3>Month: {issue['month']}</h3>"
                html += f"<p class='warning'>⚠ Found missing values in {issue['n_columns']} columns</p>"
                html += "<table><tr><th>Column</th><th>Missing Count</th><th>Missing %</th></tr>"
                for detail in issue['details'][:10]:  # Top 10
                    html += f"<tr><td>{detail['column']}</td><td>{detail['missing_count']:,}</td><td>{detail['missing_pct']}%</td></tr>"
                html += "</table>"
        else:
            html += "<p class='success'>✓ No missing values found</p>"
        
        # Outliers
        html += "<h2>2. Outliers (Winsorized)</h2>"
        if self.issues['outliers']:
            for issue in self.issues['outliers']:
                html += f"<h3>Month: {issue['month']}</h3>"
                html += "<table><tr><th>Column</th><th>Outliers</th><th>%</th><th>Max Value</th><th>Capped At (99th)</th></tr>"
                for detail in issue['details']:
                    html += f"<tr><td>{detail['column']}</td><td>{detail['n_outliers']:,}</td><td>{detail['outlier_pct']}%</td><td>${detail['max_value']:.2f}</td><td>${detail['capped_at']:.2f}</td></tr>"
                html += "</table>"
        else:
            html += "<p class='success'>✓ No significant outliers detected</p>"
        
        # Business Logic
        html += "<h2>3. Business Logic Violations</h2>"
        if self.issues['logic_violations']:
            for issue in self.issues['logic_violations']:
                html += f"<h3>Month: {issue['month']}</h3>"
                html += "<table><tr><th>Rule</th><th>Violations</th><th>%</th></tr>"
                for detail in issue['details']:
                    html += f"<tr><td>{detail['rule']}</td><td>{detail['violations']:,}</td><td>{detail['pct']}%</td></tr>"
                html += "</table>"
        else:
            html += "<p class='success'>✓ No logic violations</p>"
        
        # Duplicates
        html += "<h2>4. Duplicates</h2>"
        if self.issues['duplicates']:
            html += "<table><tr><th>Month</th><th>Duplicates Removed</th><th>%</th></tr>"
            for issue in self.issues['duplicates']:
                html += f"<tr><td>{issue['month']}</td><td>{issue['n_duplicates']:,}</td><td>{issue['pct']}%</td></tr>"
            html += "</table>"
        else:
            html += "<p class='success'>✓ No duplicates found</p>"
        
        html += """
    </div>
</body>
</html>
"""
        
        # Save report
        report_path = Path('results/data_quality_report.html')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"  ✓ Quality report saved: {report_path}")


def main():
    """Main execution"""
    print("\n")
    print("="*70)
    print(" STEP 2: DATA CLEANING & VALIDATION")
    print("="*70)
    print(f" Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Initialize cleaner
    cleaner = DataCleaner()
    
    # Get months from config
    all_months = cleaner.config['data']['all_months']
    
    # Clean each month
    print(f"\nProcessing {len(all_months)} months...")
    clean_dfs = []
    
    for i, month in enumerate(all_months, 1):
        print(f"\n[{i}/{len(all_months)}]", end=" ")
        df_clean = cleaner.clean_month(month)
        cleaner.save_clean_data(df_clean, month)
        clean_dfs.append(df_clean)
    
    # Combine all months
    print(f"\n{'='*70}")
    print("COMBINING ALL MONTHS")
    print(f"{'='*70}")
    
    df_all = pd.concat(clean_dfs, ignore_index=True)
    print(f"  ✓ Total rows: {len(df_all):,}")
    print(f"  ✓ Total columns: {len(df_all.columns)}")
    print(f"  ✓ Apps: {df_all['app_id'].nunique():,}")
    print(f"  ✓ Campaigns: {df_all['campaign'].nunique():,}")
    
    # Check wool data
    if 'app_id' in df_all.columns:
        wool_rows = (df_all['app_id'] == 'wool').sum()
        if wool_rows > 0:
            print(f"\n  📦 Wool app data: {wool_rows:,} rows ({wool_rows/len(df_all)*100:.1f}%)")
    
    # Save combined
    output_path = Path(cleaner.config['data']['processed_path']) / 'clean_data_all.csv'
    df_all.to_csv(output_path, index=False)
    print(f"\n  💾 Saved combined data: {output_path}")
    
    # Generate report
    cleaner.generate_report()
    
    # Save summary
    summary = pd.DataFrame([{
        'total_rows': len(df_all),
        'n_cols': len(df_all.columns),
        'n_apps': df_all['app_id'].nunique(),
        'n_campaigns': df_all['campaign'].nunique() if 'campaign' in df_all.columns else 0,
        'n_combos': df_all.groupby(['app_id', 'campaign']).ngroups if {'app_id', 'campaign'}.issubset(df_all.columns) else 0,
        'missing_issues': len(cleaner.issues['missing_values']),
        'outlier_issues': len(cleaner.issues['outliers']),
        'logic_violations': len(cleaner.issues['logic_violations']),
        'duplicates_removed': len(cleaner.issues['duplicates'])
    }])
    
    summary_path = Path('results/step02_cleaning_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"  💾 Saved summary: {summary_path}")
    
    print("\n" + "="*70)
    print("✅ STEP 2 COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n📁 Output files:")
    print(f"   - data/processed/clean_data_M1.csv ... clean_data_M12.csv")
    print(f"   - data/processed/clean_data_all.csv")
    print(f"   - results/data_quality_report.html")
    print(f"   - results/step02_cleaning_summary.csv")
    print(f"\n➡️  Next Step: step03_tier_classification.py")
    print("="*70)


if __name__ == "__main__":
    main()
