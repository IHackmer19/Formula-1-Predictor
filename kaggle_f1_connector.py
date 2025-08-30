#!/usr/bin/env python3
"""
Kaggle F1 Dataset Connector

This script provides multiple methods to connect to the real Kaggle F1 dataset
and automatically adapts the neural network to work with authentic data.
"""

import os
import json
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import zipfile
import urllib.request

def check_kaggle_auth():
    """Check if Kaggle API is properly authenticated"""
    try:
        result = subprocess.run(['kaggle', 'datasets', 'list', '--max-size', '1'], 
                              capture_output=True, text=True, timeout=30)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        return False

def setup_kaggle_credentials_interactive():
    """Interactive Kaggle credentials setup"""
    print("🔐 KAGGLE CREDENTIALS SETUP")
    print("="*50)
    
    kaggle_dir = Path.home() / '.config' / 'kaggle'
    kaggle_file = kaggle_dir / 'kaggle.json'
    
    if kaggle_file.exists():
        print("✅ Kaggle credentials already exist!")
        return True
    
    print("\\nTo get your Kaggle API credentials:")
    print("1. 🌐 Go to: https://www.kaggle.com/account")
    print("2. 📜 Scroll down to 'API' section")
    print("3. 🔑 Click 'Create New API Token'")
    print("4. 📥 Download the kaggle.json file")
    
    print("\\n" + "="*50)
    print("PASTE YOUR KAGGLE CREDENTIALS BELOW:")
    print("(You can find these in the downloaded kaggle.json file)")
    print("="*50)
    
    try:
        username = input("Kaggle Username: ").strip()
        if not username:
            print("❌ Username required")
            return False
            
        api_key = input("Kaggle API Key: ").strip()
        if not api_key:
            print("❌ API Key required")
            return False
        
        # Create kaggle directory
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        
        # Save credentials
        credentials = {"username": username, "key": api_key}
        with open(kaggle_file, 'w') as f:
            json.dump(credentials, f, indent=2)
        
        # Set proper permissions
        os.chmod(kaggle_file, 0o600)
        
        print("\\n✅ Kaggle credentials saved successfully!")
        print(f"📁 Saved to: {kaggle_file}")
        
        return True
        
    except KeyboardInterrupt:
        print("\\n⏭️ Setup cancelled by user")
        return False
    except Exception as e:
        print(f"❌ Error setting up credentials: {e}")
        return False

def download_f1_dataset():
    """Download the F1 dataset using Kaggle API"""
    print("\\n📥 DOWNLOADING F1 DATASET")
    print("="*40)
    
    try:
        # Download the dataset
        cmd = [
            'kaggle', 'datasets', 'download', '-d',
            'rohanrao/formula-1-world-championship-1950-2020',
            '--unzip', '--quiet'
        ]
        
        print("🔄 Downloading... (this may take a few minutes)")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Dataset downloaded successfully!")
            return True
        else:
            print(f"❌ Download failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Download timed out - dataset might be large")
        return False
    except Exception as e:
        print(f"❌ Download error: {e}")
        return False

def verify_real_dataset():
    """Verify the real dataset files"""
    print("\\n🔍 VERIFYING REAL DATASET")
    print("="*30)
    
    # Core files that should be present
    core_files = ['circuits.csv', 'constructors.csv', 'drivers.csv', 'races.csv', 'results.csv']
    optional_files = ['qualifying.csv', 'driver_standings.csv', 'constructor_standings.csv', 
                     'seasons.csv', 'status.csv', 'lap_times.csv', 'pit_stops.csv']
    
    found_core = 0
    found_optional = 0
    
    print("Core files:")
    for file in core_files:
        if os.path.exists(file):
            try:
                df = pd.read_csv(file)
                print(f"✅ {file:<20} - {len(df):,} rows")
                found_core += 1
            except Exception as e:
                print(f"❌ {file:<20} - Error: {e}")
        else:
            print(f"❌ {file:<20} - Missing")
    
    print("\\nOptional files:")
    for file in optional_files:
        if os.path.exists(file):
            try:
                df = pd.read_csv(file)
                print(f"✅ {file:<20} - {len(df):,} rows")
                found_optional += 1
            except Exception as e:
                print(f"⚠️  {file:<20} - Error: {e}")
    
    success = found_core >= 4  # Need at least 4 core files
    print(f"\\n{'✅ Verification successful!' if success else '❌ Verification failed!'}")
    print(f"Found {found_core}/{len(core_files)} core files, {found_optional}/{len(optional_files)} optional files")
    
    return success

def analyze_real_data_characteristics():
    """Analyze the characteristics of the real dataset"""
    print("\\n📊 ANALYZING REAL DATASET CHARACTERISTICS")
    print("="*50)
    
    try:
        # Load core datasets
        races = pd.read_csv('races.csv')
        results = pd.read_csv('results.csv')
        drivers = pd.read_csv('drivers.csv')
        constructors = pd.read_csv('constructors.csv')
        
        print("📅 TEMPORAL ANALYSIS:")
        year_range = races['year'].agg(['min', 'max'])
        print(f"   Data spans: {year_range['min']} - {year_range['max']} ({year_range['max'] - year_range['min'] + 1} years)")
        
        # Recent data
        recent_races = races[races['year'] >= 2010]
        recent_results = results.merge(recent_races[['raceId']], on='raceId')
        print(f"   Modern era (2010+): {len(recent_races)} races, {len(recent_results):,} results")
        
        print("\\n👥 DRIVER ANALYSIS:")
        print(f"   Total drivers in history: {len(drivers):,}")
        modern_drivers = recent_results['driverId'].nunique()
        print(f"   Modern era drivers: {modern_drivers}")
        
        print("\\n🏎️ CONSTRUCTOR ANALYSIS:")
        print(f"   Total constructors in history: {len(constructors):,}")
        modern_constructors = recent_results['constructorId'].nunique()
        print(f"   Modern era constructors: {modern_constructors}")
        
        print("\\n🏁 RACE RESULTS ANALYSIS:")
        print(f"   Total race results: {len(results):,}")
        print(f"   Modern era results: {len(recent_results):,}")
        
        # Show recent years
        print("\\n📈 RECENT YEARS COVERAGE:")
        recent_year_counts = recent_races['year'].value_counts().sort_index().tail(10)
        for year, count in recent_year_counts.items():
            print(f"   {year}: {count} races")
        
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing data: {e}")
        return False

def create_real_data_predictor():
    """Create an updated predictor for the real dataset"""
    print("\\n🔧 CREATING REAL DATA PREDICTOR")
    print("="*40)
    
    # Read the original predictor
    try:
        with open('f1_simple_predictor.py', 'r') as f:
            original_code = f.read()
        
        # Create enhanced version for real data
        enhanced_code = original_code.replace(
            'class F1PositionPredictor:',
            '''class RealF1PositionPredictor:
    """Enhanced predictor for real Kaggle F1 dataset"""'''
        )
        
        # Add real data specific enhancements
        enhanced_code = enhanced_code.replace(
            'def load_data(self):',
            '''def load_data(self):
        """Load real F1 datasets with enhanced error handling"""'''
        )
        
        # Update the data loading to handle real dataset structure
        enhanced_code = enhanced_code.replace(
            'self.results = pd.read_csv(f\\'{self.data_path}/results.csv\\')',
            '''# Load results with real dataset handling
        self.results = pd.read_csv(f\\'{self.data_path}/results.csv\\')
        
        # Handle position data - real dataset might have different column names
        if 'positionOrder' in self.results.columns and 'position' not in self.results.columns:
            self.results['position'] = self.results['positionOrder']
        
        # Filter out non-finishing positions (\\N in real data)
        if self.results['position'].dtype == 'object':
            self.results = self.results[self.results['position'] != '\\\\N']
            self.results['position'] = pd.to_numeric(self.results['position'])'''
        )
        
        # Save enhanced predictor
        with open('f1_real_predictor.py', 'w') as f:
            f.write(enhanced_code)
        
        print("✅ Enhanced predictor created: 'f1_real_predictor.py'")
        
        # Create a simple runner script
        runner_code = '''#!/usr/bin/env python3
"""
Run F1 predictions with real Kaggle dataset
"""

from f1_real_predictor import RealF1PositionPredictor

def main():
    print("🏎️ RUNNING F1 PREDICTIONS WITH REAL KAGGLE DATA")
    print("="*60)
    
    try:
        # Initialize with real data
        predictor = RealF1PositionPredictor(data_path='f1_data')
        
        # Run the full pipeline
        predictions = predictor.run_full_pipeline()
        
        print("\\n🎉 REAL DATA PREDICTIONS COMPLETE!")
        print("Check the generated files for results with authentic F1 data!")
        
        return predictions
        
    except Exception as e:
        print(f"❌ Error running real data predictions: {e}")
        print("\\nTroubleshooting:")
        print("1. Ensure all CSV files are in the f1_data directory")
        print("2. Check that the dataset was downloaded correctly")
        print("3. Verify file permissions and formats")

if __name__ == "__main__":
    main()
'''
        
        with open('run_with_real_data.py', 'w') as f:
            f.write(runner_code)
        
        print("✅ Runner script created: 'run_with_real_data.py'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating enhanced predictor: {e}")
        return False

def main():
    """Main connector function"""
    print("🔗 KAGGLE F1 DATASET CONNECTOR")
    print("="*60)
    print("Connecting to: https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020")
    print("="*60)
    
    # Check if dataset files already exist
    if os.path.exists('results.csv') and os.path.exists('drivers.csv'):
        print("✅ Real dataset files already present!")
        if verify_real_dataset():
            analyze_real_data_characteristics()
            create_real_data_predictor()
            print("\\n🎉 REAL DATASET CONNECTION ESTABLISHED!")
            print("\\nYou can now run: python3 run_with_real_data.py")
            return True
    
    # Try to set up Kaggle API and download
    print("\\n🔧 SETTING UP KAGGLE CONNECTION...")
    
    if setup_kaggle_credentials_interactive():
        print("\\n⏳ Attempting to download dataset...")
        if download_f1_dataset():
            if verify_real_dataset():
                analyze_real_data_characteristics()
                create_real_data_predictor()
                print("\\n🎉 SUCCESS! Real F1 dataset connected and ready!")
                return True
    
    # Provide manual instructions
    print("\\n📋 MANUAL SETUP REQUIRED")
    print("="*40)
    print("\\nTo connect to the real Kaggle F1 dataset:")
    print("\\n1. 🌐 Visit: https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020")
    print("2. 📥 Click 'Download' to get the ZIP file")
    print("3. 📂 Extract all CSV files to this directory")
    print("4. 🔄 Run: python3 migrate_to_real_data.py")
    print("5. 🚀 Run: python3 run_with_real_data.py")
    
    # Create the necessary scripts for when data is available
    create_real_data_predictor()
    
    print("\\n💡 All setup scripts are ready for when you get the real data!")
    
    return False

if __name__ == "__main__":
    main()