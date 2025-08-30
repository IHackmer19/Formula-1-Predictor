#!/usr/bin/env python3
"""
Setup script to download the real Formula 1 dataset from Kaggle

This script provides multiple methods to authenticate with Kaggle and download
the real F1 dataset instead of using synthetic data.
"""

import os
import json
import subprocess
import zipfile
import pandas as pd
from pathlib import Path

def setup_kaggle_credentials():
    """Setup Kaggle API credentials"""
    print("🔑 KAGGLE API SETUP")
    print("="*50)
    
    # Check if kaggle.json already exists
    kaggle_dir = Path.home() / '.config' / 'kaggle'
    kaggle_file = kaggle_dir / 'kaggle.json'
    
    if kaggle_file.exists():
        print("✅ Kaggle credentials already found!")
        return True
    
    print("\n📋 To download the real F1 dataset, you need Kaggle API credentials:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Click 'Create New API Token'")
    print("3. Download the kaggle.json file")
    print("4. Choose one of the following setup methods:\n")
    
    print("METHOD 1 - Manual Setup:")
    print(f"   mkdir -p {kaggle_dir}")
    print(f"   cp /path/to/your/kaggle.json {kaggle_file}")
    print(f"   chmod 600 {kaggle_file}")
    
    print("\nMETHOD 2 - Environment Variables:")
    print("   export KAGGLE_USERNAME='your_username'")
    print("   export KAGGLE_KEY='your_api_key'")
    
    print("\nMETHOD 3 - Interactive Setup (if you have the credentials):")
    
    # Try interactive setup
    try:
        username = input("Enter your Kaggle username (or press Enter to skip): ").strip()
        if username:
            api_key = input("Enter your Kaggle API key: ").strip()
            if api_key:
                # Create kaggle directory
                kaggle_dir.mkdir(parents=True, exist_ok=True)
                
                # Create kaggle.json
                credentials = {"username": username, "key": api_key}
                with open(kaggle_file, 'w') as f:
                    json.dump(credentials, f)
                
                # Set proper permissions
                os.chmod(kaggle_file, 0o600)
                
                print("✅ Kaggle credentials saved successfully!")
                return True
    except KeyboardInterrupt:
        print("\n⏭️ Skipping interactive setup...")
    
    return False

def download_real_dataset():
    """Download the real F1 dataset from Kaggle"""
    print("\n📥 DOWNLOADING REAL F1 DATASET")
    print("="*50)
    
    try:
        # Try to download the dataset
        result = subprocess.run([
            'kaggle', 'datasets', 'download', '-d', 
            'rohanrao/formula-1-world-championship-1950-2020',
            '--unzip'
        ], capture_output=True, text=True, check=True)
        
        print("✅ Dataset downloaded successfully!")
        print(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download dataset: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ Kaggle command not found. Make sure kaggle is installed and in PATH.")
        return False

def verify_dataset():
    """Verify the downloaded dataset"""
    print("\n🔍 VERIFYING DATASET")
    print("="*30)
    
    # Expected files from the Kaggle dataset
    expected_files = [
        'circuits.csv', 'constructor_results.csv', 'constructor_standings.csv',
        'constructors.csv', 'driver_standings.csv', 'drivers.csv', 'lap_times.csv',
        'pit_stops.csv', 'qualifying.csv', 'races.csv', 'results.csv', 'seasons.csv',
        'sprint_results.csv', 'status.csv'
    ]
    
    found_files = []
    missing_files = []
    
    for file in expected_files:
        if os.path.exists(file):
            found_files.append(file)
            # Get file info
            try:
                df = pd.read_csv(file)
                print(f"✅ {file:<25} - {len(df):,} rows")
            except Exception as e:
                print(f"⚠️  {file:<25} - Error reading: {e}")
        else:
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        return False
    
    print(f"\n✅ All {len(found_files)} dataset files verified!")
    return True

def backup_sample_data():
    """Backup the sample data before replacing with real data"""
    print("\n💾 BACKING UP SAMPLE DATA")
    print("="*30)
    
    if os.path.exists('f1_data'):
        try:
            subprocess.run(['mv', 'f1_data', 'f1_sample_data_backup'], check=True)
            print("✅ Sample data backed up to 'f1_sample_data_backup'")
        except subprocess.CalledProcessError:
            print("⚠️ Could not backup sample data")

def organize_real_data():
    """Organize the real dataset into the expected structure"""
    print("\n📁 ORGANIZING REAL DATASET")
    print("="*30)
    
    # Create new f1_data directory for real data
    os.makedirs('f1_data_real', exist_ok=True)
    
    # Move CSV files to organized directory
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and f != 'f1_2025_predictions.csv']
    
    for file in csv_files:
        try:
            subprocess.run(['mv', file, f'f1_data_real/{file}'], check=True)
            print(f"✅ Moved {file}")
        except subprocess.CalledProcessError:
            print(f"⚠️ Could not move {file}")
    
    print(f"\n✅ Real dataset organized in 'f1_data_real' directory")

def analyze_real_dataset_structure():
    """Analyze the structure of the real F1 dataset"""
    print("\n🔍 ANALYZING REAL DATASET STRUCTURE")
    print("="*50)
    
    data_dir = 'f1_data_real'
    if not os.path.exists(data_dir):
        print("❌ Real dataset directory not found")
        return
    
    # Load and analyze key files
    key_files = ['results.csv', 'drivers.csv', 'constructors.csv', 'races.csv', 'circuits.csv']
    
    for file in key_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                print(f"\n📊 {file}:")
                print(f"   Shape: {df.shape}")
                print(f"   Columns: {list(df.columns)}")
                print(f"   Sample data:")
                print(f"   {df.head(2).to_string(index=False)}")
            except Exception as e:
                print(f"❌ Error analyzing {file}: {e}")
        else:
            print(f"❌ {file} not found")

def create_real_data_adapter():
    """Create an adapter script for the real dataset"""
    print("\n🔧 CREATING REAL DATA ADAPTER")
    print("="*40)
    
    adapter_script = '''#!/usr/bin/env python3
"""
Adapter for the real Kaggle F1 dataset

This script adapts the neural network to work with the real F1 dataset structure.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def load_real_f1_data():
    """Load the real F1 dataset"""
    print("Loading real F1 dataset...")
    
    # Load all datasets
    circuits = pd.read_csv('f1_data_real/circuits.csv')
    drivers = pd.read_csv('f1_data_real/drivers.csv') 
    constructors = pd.read_csv('f1_data_real/constructors.csv')
    races = pd.read_csv('f1_data_real/races.csv')
    results = pd.read_csv('f1_data_real/results.csv')
    
    # Load additional real dataset files
    try:
        qualifying = pd.read_csv('f1_data_real/qualifying.csv')
    except FileNotFoundError:
        print("⚠️ Qualifying data not found, creating basic structure")
        qualifying = pd.DataFrame()
    
    try:
        driver_standings = pd.read_csv('f1_data_real/driver_standings.csv')
    except FileNotFoundError:
        driver_standings = pd.DataFrame()
    
    try:
        constructor_standings = pd.read_csv('f1_data_real/constructor_standings.csv')
    except FileNotFoundError:
        constructor_standings = pd.DataFrame()
    
    print(f"Real dataset loaded:")
    print(f"- Circuits: {len(circuits)}")
    print(f"- Drivers: {len(drivers)}")
    print(f"- Constructors: {len(constructors)}")
    print(f"- Races: {len(races)}")
    print(f"- Results: {len(results)}")
    print(f"- Qualifying: {len(qualifying)}")
    
    return circuits, drivers, constructors, races, results, qualifying, driver_standings, constructor_standings

def analyze_data_coverage():
    """Analyze the temporal coverage of the real dataset"""
    circuits, drivers, constructors, races, results, qualifying, driver_standings, constructor_standings = load_real_f1_data()
    
    print("\\n📅 TEMPORAL DATA COVERAGE")
    print("="*40)
    
    # Analyze race coverage
    if 'year' in races.columns:
        year_coverage = races['year'].value_counts().sort_index()
        print(f"Year range: {year_coverage.index.min()} - {year_coverage.index.max()}")
        print(f"Total years: {len(year_coverage)}")
        print(f"Average races per year: {year_coverage.mean():.1f}")
        
        # Show recent years
        print("\\nRecent years coverage:")
        recent_years = year_coverage.tail(10)
        for year, count in recent_years.items():
            print(f"  {year}: {count} races")
    
    # Analyze driver coverage
    print(f"\\n👥 DRIVER COVERAGE")
    print(f"Total unique drivers: {len(drivers)}")
    
    # Analyze results coverage
    if len(results) > 0:
        print(f"\\n🏁 RESULTS COVERAGE")
        print(f"Total race results: {len(results):,}")
        print(f"Date range: {results.merge(races, on='raceId')['year'].min()} - {results.merge(races, on='raceId')['year'].max()}")

if __name__ == "__main__":
    analyze_data_coverage()
'''
    
    with open('real_data_adapter.py', 'w') as f:
        f.write(adapter_script)
    
    print("✅ Real data adapter created: 'real_data_adapter.py'")

def main():
    """Main setup function"""
    print("🏎️ KAGGLE F1 DATASET SETUP")
    print("="*60)
    
    # Step 1: Try to setup credentials
    if setup_kaggle_credentials():
        # Step 2: Backup sample data
        backup_sample_data()
        
        # Step 3: Download real dataset
        if download_real_dataset():
            # Step 4: Organize data
            organize_real_data()
            
            # Step 5: Verify dataset
            if verify_dataset():
                # Step 6: Analyze structure
                analyze_real_dataset_structure()
                
                # Step 7: Create adapter
                create_real_data_adapter()
                
                print("\n🎉 SUCCESS! Real F1 dataset is ready!")
                print("Next steps:")
                print("1. Run: python3 real_data_adapter.py")
                print("2. Modify f1_simple_predictor.py to use 'f1_data_real' directory")
                print("3. Retrain the model with real data")
            else:
                print("❌ Dataset verification failed")
        else:
            print("❌ Dataset download failed")
            print("\n🔧 ALTERNATIVE APPROACH:")
            print("You can manually download the dataset from:")
            print("https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020")
            print("Then extract the ZIP file in this directory.")
    else:
        print("\n🔧 MANUAL SETUP REQUIRED:")
        print("Please set up Kaggle API credentials manually:")
        print("1. Download kaggle.json from your Kaggle account")
        print("2. Place it in ~/.config/kaggle/kaggle.json") 
        print("3. Run: chmod 600 ~/.config/kaggle/kaggle.json")
        print("4. Re-run this script")

if __name__ == "__main__":
    main()