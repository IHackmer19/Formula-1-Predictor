#!/usr/bin/env python3
"""
Migrate from sample data to real Kaggle F1 dataset
"""

import pandas as pd
import numpy as np
import os
import shutil

def migrate_to_real_data():
    """Migrate the prediction system to use real data"""
    print("🔄 MIGRATING TO REAL F1 DATASET")
    print("="*40)
    
    # Backup sample data
    if os.path.exists('f1_data'):
        if not os.path.exists('f1_data_sample_backup'):
            shutil.move('f1_data', 'f1_data_sample_backup')
            print("✅ Sample data backed up to 'f1_data_sample_backup'")
    
    # Create new f1_data directory with real data
    os.makedirs('f1_data', exist_ok=True)
    
    # Copy real dataset files to expected location
    real_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    core_files = ['circuits.csv', 'constructors.csv', 'drivers.csv', 'races.csv', 'results.csv']
    
    copied_files = []
    for file in core_files:
        if file in real_files:
            shutil.copy(file, f'f1_data/{file}')
            copied_files.append(file)
            print(f"✅ Copied {file}")
    
    # Handle qualifying data (might have different structure)
    if 'qualifying.csv' in real_files:
        shutil.copy('qualifying.csv', 'f1_data/qualifying.csv')
        copied_files.append('qualifying.csv')
        print("✅ Copied qualifying.csv")
    
    print(f"\n✅ Migrated {len(copied_files)} files to f1_data directory")
    
    # Analyze the real data structure
    print("\n📊 ANALYZING REAL DATA STRUCTURE")
    print("-"*40)
    
    for file in copied_files:
        try:
            df = pd.read_csv(f'f1_data/{file}')
            print(f"{file:<20}: {len(df):,} rows, {len(df.columns)} columns")
            
            # Show date range for time-series files
            if file in ['races.csv', 'results.csv'] and 'year' in df.columns:
                year_range = df['year'].agg(['min', 'max'])
                print(f"{'':20}  Years: {year_range['min']} - {year_range['max']}")
                
        except Exception as e:
            print(f"❌ Error analyzing {file}: {e}")
    
    return True

if __name__ == "__main__":
    migrate_to_real_data()
