#!/usr/bin/env python3
"""
Process Real Kaggle F1 Dataset

This script processes the authentic Formula 1 dataset from Kaggle and adapts
our neural network to work with the real data structure.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

class RealF1DataProcessor:
    def __init__(self):
        self.datasets = {}
        self.processed_data = None
        
    def check_dataset_files(self):
        """Check if the real dataset files are available"""
        print("🔍 CHECKING FOR REAL F1 DATASET FILES")
        print("="*50)
        
        # Check for dataset files in current directory
        expected_files = [
            'circuits.csv', 'constructors.csv', 'drivers.csv', 'races.csv', 
            'results.csv', 'qualifying.csv', 'driver_standings.csv',
            'constructor_standings.csv', 'seasons.csv'
        ]
        
        found_files = []
        missing_files = []
        
        for file in expected_files:
            if os.path.exists(file):
                found_files.append(file)
                try:
                    df = pd.read_csv(file)
                    print(f"✅ {file:<25} - {len(df):,} rows, {len(df.columns)} columns")
                except Exception as e:
                    print(f"⚠️  {file:<25} - Error: {e}")
            else:
                missing_files.append(file)
        
        if missing_files:
            print(f"\n❌ Missing files: {missing_files}")
            print("\n📋 TO GET THE REAL DATASET:")
            print("1. Download from: https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020")
            print("2. Extract all CSV files to this directory")
            print("3. Re-run this script")
            return False
        
        print(f"\n✅ Found all {len(found_files)} required dataset files!")
        return True
    
    def load_real_datasets(self):
        """Load all real F1 datasets"""
        print("\n📥 LOADING REAL F1 DATASETS")
        print("="*40)
        
        try:
            self.datasets['circuits'] = pd.read_csv('circuits.csv')
            self.datasets['constructors'] = pd.read_csv('constructors.csv')
            self.datasets['drivers'] = pd.read_csv('drivers.csv')
            self.datasets['races'] = pd.read_csv('races.csv')
            self.datasets['results'] = pd.read_csv('results.csv')
            
            # Optional files
            optional_files = ['qualifying.csv', 'driver_standings.csv', 'constructor_standings.csv']
            for file in optional_files:
                try:
                    self.datasets[file.replace('.csv', '')] = pd.read_csv(file)
                    print(f"✅ Loaded {file}")
                except FileNotFoundError:
                    print(f"⚠️  {file} not found, skipping")
            
            print("\n📊 DATASET OVERVIEW:")
            for name, df in self.datasets.items():
                print(f"   {name:<20}: {len(df):,} rows")
                
            return True
            
        except Exception as e:
            print(f"❌ Error loading datasets: {e}")
            return False
    
    def analyze_real_data_structure(self):
        """Analyze the structure and characteristics of the real data"""
        print("\n🔬 ANALYZING REAL DATASET STRUCTURE")
        print("="*50)
        
        # Analyze temporal coverage
        races_df = self.datasets['races']
        results_df = self.datasets['results']
        
        print(f"\n📅 TEMPORAL COVERAGE:")
        if 'year' in races_df.columns:
            year_range = races_df['year'].agg(['min', 'max'])
            total_years = year_range['max'] - year_range['min'] + 1
            print(f"   Years: {year_range['min']} - {year_range['max']} ({total_years} years)")
            
            # Recent data analysis
            recent_data = races_df[races_df['year'] >= 2010]
            print(f"   Recent data (2010+): {len(recent_data)} races")
            
            # Show year distribution
            year_counts = races_df['year'].value_counts().sort_index().tail(10)
            print(f"\\n   Recent years race count:")
            for year, count in year_counts.items():
                print(f"     {year}: {count} races")
        
        # Analyze driver coverage
        drivers_df = self.datasets['drivers']
        print(f"\\n👥 DRIVER ANALYSIS:")
        print(f"   Total drivers: {len(drivers_df)}")
        
        # Current era drivers (post-2010)
        if len(results_df) > 0:
            recent_results = results_df.merge(races_df, on='raceId')
            recent_results = recent_results[recent_results['year'] >= 2010]
            modern_drivers = recent_results['driverId'].nunique()
            print(f"   Modern era drivers (2010+): {modern_drivers}")
        
        # Analyze constructor coverage
        constructors_df = self.datasets['constructors']
        print(f"\\n🏎️  CONSTRUCTOR ANALYSIS:")
        print(f"   Total constructors: {len(constructors_df)}")
        
        if len(results_df) > 0:
            recent_constructors = recent_results['constructorId'].nunique()
            print(f"   Modern era constructors (2010+): {recent_constructors}")
        
        # Analyze circuit coverage
        circuits_df = self.datasets['circuits']
        print(f"\\n🏁 CIRCUIT ANALYSIS:")
        print(f"   Total circuits: {len(circuits_df)}")
        
        return True
    
    def create_modern_dataset(self, start_year=2010):
        """Create a modern F1 dataset for better predictions"""
        print(f"\\n🚀 CREATING MODERN DATASET (from {start_year})")
        print("="*50)
        
        races_df = self.datasets['races']
        results_df = self.datasets['results']
        
        # Filter for modern era
        modern_races = races_df[races_df['year'] >= start_year]
        modern_results = results_df.merge(modern_races[['raceId']], on='raceId')
        
        print(f"Modern dataset ({start_year}+):")
        print(f"   Races: {len(modern_races)}")
        print(f"   Results: {len(modern_results)}")
        print(f"   Years: {modern_races['year'].min()} - {modern_races['year'].max()}")
        
        # Create modern data directory
        os.makedirs('f1_data_modern', exist_ok=True)
        
        # Save modern datasets
        modern_races.to_csv('f1_data_modern/races.csv', index=False)
        modern_results.to_csv('f1_data_modern/results.csv', index=False)
        
        # Copy other relevant files
        for name in ['circuits', 'drivers', 'constructors']:
            if name in self.datasets:
                self.datasets[name].to_csv(f'f1_data_modern/{name}.csv', index=False)
        
        # Handle qualifying data if available
        if 'qualifying' in self.datasets:
            modern_qualifying = self.datasets['qualifying'].merge(modern_races[['raceId']], on='raceId')
            modern_qualifying.to_csv('f1_data_modern/qualifying.csv', index=False)
            print(f"   Qualifying: {len(modern_qualifying)}")
        
        print(f"\\n✅ Modern dataset saved to 'f1_data_modern' directory")
        return True
    
    def create_updated_predictor(self):
        """Create an updated predictor that works with real data"""
        print("\\n🔧 CREATING UPDATED PREDICTOR")
        print("="*40)
        
        # Read the existing predictor
        with open('f1_simple_predictor.py', 'r') as f:
            predictor_code = f.read()
        
        # Update to use real data directory
        updated_code = predictor_code.replace("data_path='f1_data'", "data_path='f1_data_modern'")
        
        # Save updated predictor
        with open('f1_real_data_predictor.py', 'w') as f:
            f.write(updated_code)
        
        print("✅ Updated predictor saved as 'f1_real_data_predictor.py'")
        
        # Create run script for real data
        run_script = '''#!/usr/bin/env python3
"""
Run F1 predictions with real Kaggle dataset
"""

import subprocess
import sys

def main():
    print("🏎️ RUNNING F1 PREDICTIONS WITH REAL DATA")
    print("="*50)
    
    try:
        result = subprocess.run([sys.executable, 'f1_real_data_predictor.py'], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running predictor: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")

if __name__ == "__main__":
    main()
'''
        
        with open('run_real_predictions.py', 'w') as f:
            f.write(run_script)
        
        print("✅ Run script created: 'run_real_predictions.py'")

def main():
    """Main processing function"""
    processor = RealF1DataProcessor()
    
    # Check if real dataset is available
    if processor.check_dataset_files():
        # Load and analyze real data
        if processor.load_real_datasets():
            processor.analyze_real_data_structure()
            processor.create_modern_dataset()
            processor.create_updated_predictor()
            
            print("\n🎉 REAL DATASET PROCESSING COMPLETE!")
            print("="*50)
            print("Next steps:")
            print("1. Run: python3 run_real_predictions.py")
            print("2. Compare results with sample data predictions")
            print("3. The model will now use authentic F1 data!")
        else:
            print("❌ Failed to load real datasets")
    else:
        print("\n📥 MANUAL DOWNLOAD REQUIRED")
        print("="*40)
        print("To use the real F1 dataset:")
        print("1. Visit: https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020")
        print("2. Download the dataset ZIP file")
        print("3. Extract all CSV files to this directory")
        print("4. Re-run: python3 process_real_f1_data.py")
        
        print("\n🔧 OR use Kaggle API:")
        print("1. Set up kaggle.json credentials")
        print("2. Run: kaggle datasets download -d rohanrao/formula-1-world-championship-1950-2020 --unzip")
        print("3. Re-run this script")

if __name__ == "__main__":
    main()