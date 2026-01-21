"""
Step 1: Environment Setup & Data Loading
==========================================
Chuẩn bị môi trường và tải dữ liệu 12 tháng (Jan-Dec 2025)

Author: LTV Prediction System V2.1
Date: 2026-01-21
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ProjectSetup:
    """Setup project structure and configuration"""
    
    def __init__(self, project_root="."):
        self.root = Path(project_root)
        
    def create_folder_structure(self):
        """Tạo cấu trúc thư mục dự án"""
        print("\n" + "="*70)
        print("CREATING FOLDER STRUCTURE")
        print("="*70)
        
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
            folder_path = self.root / folder
            folder_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created: {folder}")
        
        print(f"\n  ✓ All folders created successfully!")
        
    def load_config(self):
        """Load existing config.yaml"""
        config_path = self.root / 'config' / 'config.yaml'
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"\n  ✓ Loaded config: {config_path}")
        print(f"    - Project: {config['project']['name']}")
        print(f"    - Version: {config['project']['version']}")
        print(f"    - Target MAPE: {config['project']['target_mape']}")
        
        return config


class DataLoader:
    """Load and explore data from all months"""
    
    def __init__(self, config):
        self.config = config
        self.data_path = Path(config['data']['raw_path'])
        
        # Month mapping: M1->jan.csv, M2->feb.csv, ..., M8->data_T8.csv, ...
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
        
        # Month names for display
        self.month_names = {
            'M1': 'Jan', 'M2': 'Feb', 'M3': 'Mar', 'M4': 'Apr',
            'M5': 'May', 'M6': 'Jun', 'M7': 'Jul', 'M8': 'Aug',
            'M9': 'Sep', 'M10': 'Oct', 'M11': 'Nov', 'M12': 'Dec'
        }
        
    def load_monthly_data(self, month):
        """Load data for a specific month"""
        if month not in self.month_files:
            print(f"  ⚠ Warning: Invalid month {month}")
            return None
        
        filename = self.month_files[month]
        file_path = self.data_path / filename
        
        if not file_path.exists():
            print(f"  ⚠ Warning: {file_path} not found, skipping...")
            return None
        
        month_name = self.month_names.get(month, month)
        print(f"  Loading {month} ({month_name} 2025) from {filename}...", end=' ')
        
        try:
            df = pd.read_csv(file_path)
            df['month'] = month  # Standardize to M1-M12
            print(f"✓ {len(df):,} rows")
            return df
        except Exception as e:
            print(f"✗ Error: {e}")
            return None
    
    def load_all_data(self):
        """Load all 12 months data (Jan-Dec 2025)"""
        print("\n" + "="*70)
        print("LOADING DATA - ALL 12 MONTHS (JAN-DEC 2025)")
        print("="*70)
        
        all_months = self.config['data']['all_months']  # M1-M12
        
        dfs = []
        for month in all_months:
            df = self.load_monthly_data(month)
            if df is not None:
                dfs.append(df)
        
        if not dfs:
            raise ValueError("No data loaded! Please check data files.")
        
        combined = pd.concat(dfs, ignore_index=True)
        print(f"\n  ✓ Total rows loaded: {len(combined):,}")
        print(f"  ✓ Months loaded: {len(dfs)}/{len(all_months)}")
        
        return combined
    
    def load_country_tiers(self):
        """Load country tier mapping"""
        print("\n" + "="*70)
        print("LOADING COUNTRY TIERS")
        print("="*70)
        
        file_path = self.data_path / "countries_tier.csv"
        
        if file_path.exists():
            df = pd.read_csv(file_path)
            print(f"  ✓ Loaded country tiers: {len(df)} countries")
            return df
        else:
            print("  ⚠ countries_tier.csv not found, will create default tiers later")
            return None
    
    def load_wool_data(self):
        """Load special wool app data (high-revenue app)"""
        print("\n" + "="*70)
        print("LOADING WOOL DATA (HIGH-REVENUE APP)")
        print("="*70)
        
        wool_path = self.data_path / "wool"
        
        if not wool_path.exists():
            print("  ⚠ Wool folder not found, skipping wool data")
            return None
        
        wool_dfs = []
        
        # Load D30 data (T11, T12)
        d30_file = wool_path / "data_wool_D30_T11_T12.csv"
        if d30_file.exists():
            print(f"  Loading Wool D30 (Nov-Dec 2025)...", end=' ')
            try:
                df_d30 = pd.read_csv(d30_file)
                
                # Split 50/50 for M11 and M12
                split_idx = len(df_d30) // 2
                df_d30['month'] = 'M11'
                df_d30.loc[split_idx:, 'month'] = 'M12'
                df_d30['wool_data_type'] = 'D30'
                
                print(f"✓ {len(df_d30):,} rows")
                wool_dfs.append(df_d30)
            except Exception as e:
                print(f"✗ Error: {e}")
        
        # Load D60 data (T7-T10)
        d60_file = wool_path / "data_wool_D60_T7_T10.csv"
        if d60_file.exists():
            print(f"  Loading Wool D60 (Jul-Oct 2025)...", end=' ')
            try:
                df_d60 = pd.read_csv(d60_file)
                
                # Split into 4 parts for M7, M8, M9, M10
                n_rows = len(df_d60)
                chunk_size = n_rows // 4
                
                df_d60['month'] = 'M7'
                df_d60.loc[chunk_size:chunk_size*2, 'month'] = 'M8'
                df_d60.loc[chunk_size*2:chunk_size*3, 'month'] = 'M9'
                df_d60.loc[chunk_size*3:, 'month'] = 'M10'
                df_d60['wool_data_type'] = 'D60'
                
                print(f"✓ {len(df_d60):,} rows")
                wool_dfs.append(df_d60)
            except Exception as e:
                print(f"✗ Error: {e}")
        
        if wool_dfs:
            wool_combined = pd.concat(wool_dfs, ignore_index=True)
            print(f"\n  ✓ Total Wool rows: {len(wool_combined):,}")
            return wool_combined
        else:
            print("  ⚠ No wool data files found")
            return None
    
    def explore_data(self, df):
        """Khám phá dữ liệu ban đầu"""
        print("\n" + "="*70)
        print("DATA EXPLORATION")
        print("="*70)
        
        # Basic stats
        print(f"\n1. Dataset Shape:")
        print(f"   - Rows: {len(df):,}")
        print(f"   - Columns: {len(df.columns)}")
        
        # Columns info
        print(f"\n2. Columns ({len(df.columns)}):")
        for col in df.columns:
            dtype = df[col].dtype
            null_count = df[col].isna().sum()
            null_pct = null_count / len(df) * 100
            unique_count = df[col].nunique()
            print(f"   - {col:30s} | {str(dtype):10s} | Null: {null_pct:5.1f}% | Unique: {unique_count:,}")
        
        # Key business metrics
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
            print(f"\n4. Monthly Distribution (2025):")
            for month, count in month_dist.items():
                pct = count / len(df) * 100
                month_name = self.month_names.get(month, month)
                print(f"   - {month} ({month_name}): {count:,} rows ({pct:.1f}%)")
        
        # Revenue stats (if exists)
        rev_cols = [c for c in df.columns if 'rev' in c.lower() or 'ltv' in c.lower()]
        if rev_cols:
            print(f"\n5. Revenue Columns ({len(rev_cols)} columns):")
            for col in rev_cols[:10]:  # Show first 10
                mean_val = df[col].mean()
                median_val = df[col].median()
                max_val = df[col].max()
                print(f"   - {col:20s}: Mean=${mean_val:.4f}, Median=${median_val:.4f}, Max=${max_val:.2f}")
        
        # Check for wool data
        if 'wool_data_type' in df.columns:
            wool_counts = df['wool_data_type'].value_counts()
            print(f"\n6. Wool Data Breakdown:")
            for wool_type, count in wool_counts.items():
                pct = count / len(df) * 100
                print(f"   - {wool_type}: {count:,} rows ({pct:.1f}%)")
        
        # Compile overview stats
        overview_stats = {
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'n_apps': df['app_id'].nunique() if 'app_id' in df.columns else 0,
            'n_campaigns': df['campaign'].nunique() if 'campaign' in df.columns else 0,
            'n_combos': df.groupby(['app_id', 'campaign']).ngroups if {'app_id', 'campaign'}.issubset(df.columns) else 0,
            'months_loaded': df['month'].nunique() if 'month' in df.columns else 0,
            'has_wool_data': 'wool_data_type' in df.columns
        }
        
        return overview_stats
    
    def save_overview(self, df, overview_stats):
        """Lưu data overview và sample"""
        print("\n" + "="*70)
        print("SAVING DATA OVERVIEW")
        print("="*70)
        
        # Save overview stats
        stats_df = pd.DataFrame([overview_stats])
        stats_path = Path('data/interim/data_overview.csv')
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_df.to_csv(stats_path, index=False)
        print(f"  ✓ Saved overview stats: {stats_path}")
        
        # Save sample data (first 1000 rows)
        sample_path = Path('data/interim/sample_data.csv')
        df.head(1000).to_csv(sample_path, index=False)
        print(f"  ✓ Saved sample (1000 rows): {sample_path}")
        
        # Save column info
        col_info = pd.DataFrame({
            'column': df.columns,
            'dtype': [str(df[col].dtype) for col in df.columns],
            'null_count': [df[col].isna().sum() for col in df.columns],
            'null_pct': [df[col].isna().sum() / len(df) * 100 for col in df.columns],
            'unique_count': [df[col].nunique() for col in df.columns]
        })
        col_info_path = Path('data/interim/column_info.csv')
        col_info.to_csv(col_info_path, index=False)
        print(f"  ✓ Saved column info: {col_info_path}")


def main():
    """Main execution"""
    print("\n")
    print("="*70)
    print(" STEP 1: ENVIRONMENT SETUP & DATA LOADING")
    print("="*70)
    print(f" Project: LTV ROAS Prediction V2.1")
    print(f" Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # 1. Setup project structure
    print("\n[1/5] Creating folder structure...")
    setup = ProjectSetup()
    setup.create_folder_structure()
    
    # 2. Load config
    print("\n[2/5] Loading configuration...")
    config = setup.load_config()
    
    # 3. Load main data (12 months)
    print("\n[3/5] Loading main data...")
    loader = DataLoader(config)
    df = loader.load_all_data()
    
    # 4. Load auxiliary data
    print("\n[4/5] Loading auxiliary data...")
    
    # Load country tiers
    country_df = loader.load_country_tiers()
    
    # Load wool data (high-revenue app)
    wool_df = loader.load_wool_data()
    
    # Merge wool data with main data if available
    if wool_df is not None:
        print("\n  🔗 Merging WOOL data with main dataset...")
        
        # Ensure wool has app_id
        if 'app_id' not in wool_df.columns:
            wool_df['app_id'] = 'wool'
            print("     - Added app_id='wool' to wool data")
        
        # Append wool rows to main dataset
        df_before = len(df)
        df = pd.concat([df, wool_df], ignore_index=True)
        print(f"     ✓ Total rows after wool merge: {len(df):,} (+{len(df)-df_before:,})")
    
    # 5. Explore data and save overview
    print("\n[5/5] Exploring data and saving overview...")
    overview_stats = loader.explore_data(df)
    loader.save_overview(df, overview_stats)
    
    # Final summary
    print("\n" + "="*70)
    print("✅ STEP 1 COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"   - Total rows: {len(df):,}")
    print(f"   - Total columns: {len(df.columns)}")
    print(f"   - Months: {overview_stats['months_loaded']}/12")
    print(f"   - Apps: {overview_stats['n_apps']:,}")
    print(f"   - Campaigns: {overview_stats['n_campaigns']:,}")
    print(f"   - Wool data included: {'Yes' if overview_stats['has_wool_data'] else 'No'}")
    
    print(f"\n📁 Output files:")
    print(f"   - data/interim/data_overview.csv")
    print(f"   - data/interim/sample_data.csv")
    print(f"   - data/interim/column_info.csv")
    
    print(f"\n➡️  Next Step: step02_data_cleaning.py")
    print("="*70)
    
    return df, config


if __name__ == "__main__":
    df, config = main()
