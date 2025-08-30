#!/usr/bin/env python3
"""
Download Real Kaggle F1 Dataset

This script provides multiple methods to download and set up the authentic
Formula 1 dataset from Kaggle.
"""

import os
import json
import subprocess
import sys
from pathlib import Path
import requests
import zipfile

def setup_kaggle_api():
    """Interactive setup for Kaggle API"""
    print("🔑 KAGGLE API SETUP")
    print("="*50)
    
    print("To download the real F1 dataset, you need Kaggle API credentials.")
    print("\\n📋 Steps to get credentials:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Scroll to 'API' section")
    print("3. Click 'Create New API Token'")
    print("4. This downloads kaggle.json with your credentials")
    
    # Check if credentials already exist
    kaggle_dir = Path.home() / '.config' / 'kaggle'
    kaggle_file = kaggle_dir / 'kaggle.json'
    
    if kaggle_file.exists():
        print(f"\\n✅ Found existing credentials at {kaggle_file}")
        return True
    
    print(f"\\n🔧 SETUP OPTIONS:")
    print("\\nOption 1 - Environment Variables:")
    print("   export KAGGLE_USERNAME='your_username'")
    print("   export KAGGLE_KEY='your_api_key'")
    
    print("\\nOption 2 - Manual File Placement:")
    print(f"   mkdir -p {kaggle_dir}")
    print(f"   cp /path/to/kaggle.json {kaggle_file}")
    print(f"   chmod 600 {kaggle_file}")
    
    print("\\nOption 3 - Interactive Setup:")
    
    try:
        choice = input("\\nWould you like to set up credentials interactively? (y/n): ").lower().strip()
        if choice == 'y':
            username = input("Enter your Kaggle username: ").strip()
            api_key = input("Enter your Kaggle API key: ").strip()
            
            if username and api_key:
                # Create directory
                kaggle_dir.mkdir(parents=True, exist_ok=True)
                
                # Create credentials file
                credentials = {"username": username, "key": api_key}
                with open(kaggle_file, 'w') as f:
                    json.dump(credentials, f)
                
                # Set permissions
                os.chmod(kaggle_file, 0o600)
                
                print("✅ Credentials saved successfully!")
                return True
            else:
                print("❌ Invalid credentials provided")
                
    except KeyboardInterrupt:
        print("\\n⏭️ Skipping interactive setup")
    
    # Check for environment variables
    if os.getenv('KAGGLE_USERNAME') and os.getenv('KAGGLE_KEY'):
        print("✅ Found Kaggle credentials in environment variables!")
        return True
    
    return False

def download_with_kaggle_api():
    """Download dataset using Kaggle API"""
    print("\\n📥 DOWNLOADING WITH KAGGLE API")
    print("="*40)
    
    try:
        # Test kaggle authentication
        result = subprocess.run(['kaggle', 'datasets', 'list', '--max-size', '1'], 
                              capture_output=True, text=True, check=True)
        print("✅ Kaggle API authentication successful!")
        
        # Download the F1 dataset
        print("Downloading Formula 1 dataset...")
        result = subprocess.run([
            'kaggle', 'datasets', 'download', '-d', 
            'rohanrao/formula-1-world-championship-1950-2020',
            '--unzip'
        ], capture_output=True, text=True, check=True)
        
        print("✅ Dataset downloaded and extracted successfully!")
        print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Kaggle API error: {e}")
        print(f"Error details: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ Kaggle command not found")
        return False

def verify_downloaded_data():
    """Verify that the real dataset was downloaded correctly"""
    print("\\n🔍 VERIFYING DOWNLOADED DATASET")
    print("="*40)
    
    expected_files = [
        'circuits.csv', 'constructors.csv', 'drivers.csv', 'races.csv',
        'results.csv', 'qualifying.csv', 'driver_standings.csv',
        'constructor_standings.csv', 'seasons.csv', 'status.csv'
    ]
    
    found_files = []
    missing_files = []
    
    for file in expected_files:
        if os.path.exists(file):
            found_files.append(file)
            try:
                import pandas as pd
                df = pd.read_csv(file)
                print(f"✅ {file:<25} - {len(df):,} rows")
            except Exception as e:
                print(f"⚠️  {file:<25} - Error: {e}")
        else:
            missing_files.append(file)
    
    if missing_files:
        print(f"\\n⚠️ Missing files: {missing_files}")
        print("Some files might be optional or have different names in the dataset.")
    
    if len(found_files) >= 5:  # At least the core files
        print(f"\\n✅ Dataset verification successful! Found {len(found_files)} files.")
        return True
    else:
        print(f"\\n❌ Dataset verification failed. Only found {len(found_files)} files.")
        return False

def create_data_migration_script():
    """Create a script to migrate from sample to real data"""
    print("\\n📝 CREATING DATA MIGRATION SCRIPT")
    print("="*40)
    
    migration_script = '''#!/usr/bin/env python3
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
    
    print(f"\\n✅ Migrated {len(copied_files)} files to f1_data directory")
    
    # Analyze the real data structure
    print("\\n📊 ANALYZING REAL DATA STRUCTURE")
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
'''
    
    with open('migrate_to_real_data.py', 'w') as f:
        f.write(migration_script)
    
    print("✅ Migration script created: 'migrate_to_real_data.py'")

def main():
    """Main download function"""
    print("🏎️ KAGGLE F1 DATASET DOWNLOADER")
    print("="*60)
    
    # Method 1: Try Kaggle API
    if setup_kaggle_api():
        if download_with_kaggle_api():
            if verify_downloaded_data():
                create_data_migration_script()
                print("\\n🎉 SUCCESS! Real F1 dataset is ready!")
                print("\\nNext steps:")
                print("1. Run: python3 migrate_to_real_data.py")
                print("2. Run: python3 f1_simple_predictor.py")
                print("3. Compare predictions with real vs sample data!")
                return True
    
    # Method 2: Manual instructions
    print("\\n📋 MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*50)
    print("Since automatic download failed, please follow these steps:")
    print("\\n1. 🌐 Visit the dataset page:")
    print("   https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020")
    print("\\n2. 📥 Download the dataset:")
    print("   - Click the 'Download' button")
    print("   - This will download 'archive.zip' or similar")
    print("\\n3. 📂 Extract the files:")
    print("   - Extract all CSV files to this workspace directory")
    print("   - You should see files like: circuits.csv, drivers.csv, results.csv, etc.")
    print("\\n4. 🔄 Run the migration:")
    print("   python3 migrate_to_real_data.py")
    print("\\n5. 🚀 Retrain with real data:")
    print("   python3 f1_simple_predictor.py")
    
    # Create the migration script anyway
    create_data_migration_script()
    
    print("\\n💡 TIP: The migration script is ready for when you get the real data!")

if __name__ == "__main__":
    main()