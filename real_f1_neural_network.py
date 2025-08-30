#!/usr/bin/env python3
"""
Real Kaggle F1 Dataset Neural Network Predictor

This script is specifically designed to work with the authentic Kaggle F1 dataset
from rohanrao/formula-1-world-championship-1950-2020
"""

import pandas as pd
import numpy as np
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

class RealKaggleF1Predictor:
    def __init__(self, data_path='.'):
        """Initialize with real Kaggle dataset"""
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model = None
        self.datasets = {}
        
    def check_real_dataset(self):
        """Check if real Kaggle dataset is available"""
        print("🔍 CHECKING FOR REAL KAGGLE F1 DATASET")
        print("="*50)
        
        required_files = ['circuits.csv', 'constructors.csv', 'drivers.csv', 'races.csv', 'results.csv']
        optional_files = ['qualifying.csv', 'driver_standings.csv', 'constructor_standings.csv', 
                         'lap_times.csv', 'pit_stops.csv', 'seasons.csv', 'status.csv']
        
        missing_files = []
        available_files = []
        
        for file in required_files:
            file_path = os.path.join(self.data_path, file) if self.data_path != '.' else file
            if os.path.exists(file_path):
                available_files.append(file)
                try:
                    df = pd.read_csv(file_path)
                    print(f"✅ {file:<25} - {len(df):,} rows")
                except Exception as e:
                    print(f"⚠️  {file:<25} - Error: {e}")
            else:
                missing_files.append(file)
                print(f"❌ {file:<25} - Missing")
        
        if missing_files:
            print(f"\\n❌ Missing required files: {missing_files}")
            print("\\n📥 TO GET THE REAL DATASET:")
            print("1. Download from: https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020")
            print("2. Extract all CSV files to this directory")
            print("3. Re-run this script")
            return False
        
        print(f"\\n✅ All required files found! Checking optional files...")
        
        for file in optional_files:
            file_path = os.path.join(self.data_path, file) if self.data_path != '.' else file
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    print(f"✅ {file:<25} - {len(df):,} rows")
                    available_files.append(file)
                except Exception as e:
                    print(f"⚠️  {file:<25} - Error: {e}")
        
        print(f"\\n🎉 Dataset verification complete! Found {len(available_files)} files.")
        return True
    
    def load_real_data(self):
        """Load the real Kaggle F1 dataset"""
        print("\\n📥 LOADING REAL KAGGLE F1 DATASET")
        print("="*50)
        
        # Load core datasets
        file_path = lambda f: os.path.join(self.data_path, f) if self.data_path != '.' else f
        
        self.datasets['circuits'] = pd.read_csv(file_path('circuits.csv'))
        self.datasets['constructors'] = pd.read_csv(file_path('constructors.csv'))
        self.datasets['drivers'] = pd.read_csv(file_path('drivers.csv'))
        self.datasets['races'] = pd.read_csv(file_path('races.csv'))
        self.datasets['results'] = pd.read_csv(file_path('results.csv'))
        
        # Load optional datasets
        optional_datasets = ['qualifying', 'driver_standings', 'constructor_standings', 'seasons', 'status']
        for dataset in optional_datasets:
            try:
                self.datasets[dataset] = pd.read_csv(file_path(f'{dataset}.csv'))
                print(f"✅ Loaded {dataset}.csv")
            except FileNotFoundError:
                print(f"⚠️  {dataset}.csv not found, skipping")
        
        print("\\n📊 REAL DATASET OVERVIEW:")
        for name, df in self.datasets.items():
            if 'year' in df.columns:
                year_range = df['year'].agg(['min', 'max'])
                print(f"   {name:<20}: {len(df):,} rows ({year_range['min']}-{year_range['max']})")
            else:
                print(f"   {name:<20}: {len(df):,} rows")
        
        return True
    
    def analyze_real_data_quality(self):
        """Analyze the quality and structure of real data"""
        print("\\n🔬 REAL DATA QUALITY ANALYSIS")
        print("="*50)
        
        results = self.datasets['results']
        races = self.datasets['races']
        
        # Analyze position data
        print("🏁 POSITION DATA ANALYSIS:")
        
        # Check for missing positions (\\N in real data)
        if results['position'].dtype == 'object':
            null_positions = results[results['position'] == '\\\\N']
            print(f"   Results with \\\\N positions: {len(null_positions):,}")
            
            # Convert to numeric, handling \\N values
            results_clean = results[results['position'] != '\\\\N'].copy()
            results_clean['position'] = pd.to_numeric(results_clean['position'])
            self.datasets['results'] = results_clean
            
            print(f"   Clean results for analysis: {len(results_clean):,}")
        else:
            print("   Position data already numeric")
        
        # Temporal analysis
        results_with_dates = self.datasets['results'].merge(races, on='raceId')
        
        print("\\n📅 TEMPORAL DISTRIBUTION:")
        year_counts = results_with_dates['year'].value_counts().sort_index()
        print(f"   Years: {year_counts.index.min()} - {year_counts.index.max()}")
        print(f"   Total race results: {len(results_with_dates):,}")
        
        # Modern era focus (2000+)
        modern_data = results_with_dates[results_with_dates['year'] >= 2000]
        print(f"   Modern era (2000+): {len(modern_data):,} results")
        
        # Show data density by decade
        print("\\n📈 DATA DENSITY BY DECADE:")
        for decade_start in range(1950, 2030, 10):
            decade_data = results_with_dates[
                (results_with_dates['year'] >= decade_start) & 
                (results_with_dates['year'] < decade_start + 10)
            ]
            if len(decade_data) > 0:
                print(f"   {decade_start}s: {len(decade_data):,} results")
        
        return True
    
    def create_enhanced_features(self):
        """Create enhanced features using real F1 data"""
        print("\\n🚀 ENHANCED FEATURE ENGINEERING WITH REAL DATA")
        print("="*60)
        
        # Use modern era data (2000+) for better relevance to 2025 predictions
        races = self.datasets['races']
        results = self.datasets['results']
        
        # Filter for modern era
        modern_races = races[races['year'] >= 2000]
        modern_results = results.merge(modern_races[['raceId']], on='raceId')
        
        print(f"Using modern era data: {len(modern_results):,} results from {modern_races['year'].min()}-{modern_races['year'].max()}")
        
        # Merge with other datasets
        df = modern_results.merge(modern_races, on='raceId')
        df = df.merge(self.datasets['drivers'], on='driverId')
        df = df.merge(self.datasets['constructors'], on='constructorId')
        df = df.merge(self.datasets['circuits'], on='circuitId')
        
        # Add qualifying data if available
        if 'qualifying' in self.datasets:
            qualifying = self.datasets['qualifying']
            qualifying_modern = qualifying.merge(modern_races[['raceId']], on='raceId')
            df = df.merge(qualifying_modern[['raceId', 'driverId', 'position']], 
                         on=['raceId', 'driverId'], suffixes=('', '_qual'), how='left')
            print(f"✅ Added qualifying data: {len(qualifying_modern):,} entries")
        
        # Convert date and sort
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['date', 'driverId']).reset_index(drop=True)
        
        print(f"Merged modern dataset: {df.shape}")
        
        # Enhanced feature engineering for real data
        features = []
        
        print("🔄 Processing enhanced features...")
        total_rows = len(df)
        
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"   Progress: {idx:,}/{total_rows:,} ({idx/total_rows*100:.1f}%)")
            
            feature_row = self._create_enhanced_feature_row(row, df, idx)
            features.append(feature_row)
        
        features_df = pd.DataFrame(features)
        features_df = features_df.dropna(subset=['target_position'])
        
        print(f"\\n✅ Enhanced features created: {features_df.shape}")
        print(f"Features: {list(features_df.columns)}")
        
        self.processed_data = features_df
        return features_df
    
    def _create_enhanced_feature_row(self, row, df, idx):
        """Create enhanced feature row with real F1 data insights"""
        driver_id = row['driverId']
        constructor_id = row['constructorId']
        circuit_id = row['circuitId']
        current_date = row['date']
        current_year = row['year']
        
        # Basic features
        feature_row = {
            'driverId': driver_id,
            'constructorId': constructor_id,
            'circuitId': circuit_id,
            'year': current_year,
            'round': row['round'],
            'grid_position': row['grid'],
            'target_position': row['position']
        }
        
        # Add qualifying position if available
        if 'position_qual' in row and pd.notna(row['position_qual']):
            feature_row['qualifying_position'] = row['position_qual']
        else:
            feature_row['qualifying_position'] = row['grid']  # Use grid as fallback
        
        # Circuit characteristics
        feature_row['circuit_lat'] = row['lat']
        feature_row['circuit_lng'] = row['lng']
        feature_row['circuit_alt'] = row['alt'] if 'alt' in row else 0
        
        # Historical data for this driver
        historical_mask = (df['driverId'] == driver_id) & (df['date'] < current_date)
        historical_data = df[historical_mask]
        
        if len(historical_data) > 0:
            # Recent performance (last 5 races)
            recent_5 = historical_data.tail(5)
            feature_row['avg_position_5'] = recent_5['position'].mean()
            feature_row['avg_points_5'] = recent_5['points'].mean()
            feature_row['avg_grid_5'] = recent_5['grid'].mean()
            
            # Medium term (last 10 races)
            recent_10 = historical_data.tail(10)
            feature_row['avg_position_10'] = recent_10['position'].mean()
            feature_row['avg_points_10'] = recent_10['points'].mean()
            
            # Season performance
            season_data = historical_data[historical_data['year'] == current_year]
            feature_row['season_avg_position'] = season_data['position'].mean() if len(season_data) > 0 else 10.5
            feature_row['season_points'] = season_data['points'].sum() if len(season_data) > 0 else 0
            
            # Career statistics
            feature_row['career_avg_position'] = historical_data['position'].mean()
            feature_row['career_total_points'] = historical_data['points'].sum()
            feature_row['career_races'] = len(historical_data)
            feature_row['career_wins'] = len(historical_data[historical_data['position'] == 1])
            feature_row['career_podiums'] = len(historical_data[historical_data['position'] <= 3])
            feature_row['career_top10s'] = len(historical_data[historical_data['position'] <= 10])
            
            # Recent form (last 3 races)
            recent_3 = historical_data.tail(3)
            feature_row['recent_form_3'] = recent_3['position'].mean()
            feature_row['recent_points_3'] = recent_3['points'].mean()
            
        else:
            # Default values for rookie drivers
            defaults = {
                'avg_position_5': 15.0, 'avg_points_5': 0, 'avg_grid_5': 15.0,
                'avg_position_10': 15.0, 'avg_points_10': 0, 'season_avg_position': 15.0,
                'season_points': 0, 'career_avg_position': 15.0, 'career_total_points': 0,
                'career_races': 0, 'career_wins': 0, 'career_podiums': 0, 'career_top10s': 0,
                'recent_form_3': 15.0, 'recent_points_3': 0
            }
            feature_row.update(defaults)
        
        # Constructor performance
        constructor_mask = (df['constructorId'] == constructor_id) & (df['date'] < current_date)
        constructor_data = df[constructor_mask]
        
        if len(constructor_data) > 0:
            constructor_recent = constructor_data.tail(20)
            feature_row['constructor_avg_position'] = constructor_recent['position'].mean()
            feature_row['constructor_avg_points'] = constructor_recent['points'].mean()
            feature_row['constructor_recent_wins'] = len(constructor_recent[constructor_recent['position'] == 1])
        else:
            feature_row['constructor_avg_position'] = 10.5
            feature_row['constructor_avg_points'] = 0
            feature_row['constructor_recent_wins'] = 0
        
        # Circuit-specific performance
        circuit_driver_mask = (df['driverId'] == driver_id) & (df['circuitId'] == circuit_id) & (df['date'] < current_date)
        circuit_driver_data = df[circuit_driver_mask]
        
        if len(circuit_driver_data) > 0:
            feature_row['circuit_driver_avg_position'] = circuit_driver_data['position'].mean()
            feature_row['circuit_driver_best_position'] = circuit_driver_data['position'].min()
            feature_row['circuit_driver_races'] = len(circuit_driver_data)
        else:
            feature_row['circuit_driver_avg_position'] = 10.5
            feature_row['circuit_driver_best_position'] = 20
            feature_row['circuit_driver_races'] = 0
        
        # Era-specific features (account for F1 evolution)
        if current_year >= 2014:  # Turbo hybrid era
            feature_row['era_turbo_hybrid'] = 1
        else:
            feature_row['era_turbo_hybrid'] = 0
            
        if current_year >= 2009:  # Modern F1
            feature_row['era_modern'] = 1
        else:
            feature_row['era_modern'] = 0
        
        return feature_row
    
    def build_enhanced_model(self, input_dim):
        """Build enhanced neural network for real F1 data"""
        print("\\n🧠 BUILDING ENHANCED NEURAL NETWORK")
        print("="*50)
        
        model = keras.Sequential([
            # Input layer with more capacity for real data complexity
            layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            
            # Deep hidden layers to capture F1 complexity
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.4),
            layers.BatchNormalization(),
            
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            
            # Output layer for position prediction
            layers.Dense(1, activation='linear')
        ])
        
        # Enhanced compilation for better F1 prediction
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),  # Lower learning rate
            loss='huber',  # More robust to outliers
            metrics=['mae', 'mse']
        )
        
        print("Enhanced model architecture:")
        model.summary()
        
        self.model = model
        return model
    
    def train_with_real_data(self):
        """Train the model with real F1 data"""
        print("\\n🎓 TRAINING WITH REAL F1 DATA")
        print("="*50)
        
        if self.processed_data is None:
            print("❌ No processed data available. Run create_enhanced_features() first.")
            return None
        
        # Prepare features
        feature_columns = [col for col in self.processed_data.columns 
                          if col not in ['target_position', 'driverId']]
        
        X = self.processed_data[feature_columns].copy()
        y = self.processed_data['target_position'].copy()
        
        print(f"Training with {len(X)} samples and {len(feature_columns)} features")
        
        # Handle categorical variables
        categorical_columns = ['constructorId', 'circuitId']
        for col in categorical_columns:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                self.label_encoders[col] = le
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data with stratification by year to ensure temporal diversity
        years = self.processed_data['year']
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=years
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Build model
        self.build_enhanced_model(X_train.shape[1])
        
        # Enhanced training with real data
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7),
            keras.callbacks.ModelCheckpoint('best_f1_model.h5', save_best_only=True, monitor='val_loss')
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=150,  # More epochs for real data
            batch_size=64,  # Larger batch size
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        self.evaluate_real_model(X_test, y_test)
        
        return history, X_test, y_test, feature_columns
    
    def evaluate_real_model(self, X_test, y_test):
        """Evaluate model performance with real data"""
        print("\\n📊 REAL DATA MODEL EVALUATION")
        print("="*50)
        
        # Predictions
        y_pred = self.model.predict(X_test).flatten()
        y_pred_rounded = np.clip(np.round(y_pred), 1, 20)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred_rounded)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_rounded))
        
        print(f"🎯 PERFORMANCE METRICS:")
        print(f"   Mean Absolute Error: {mae:.2f} positions")
        print(f"   Root Mean Square Error: {rmse:.2f} positions")
        
        # Accuracy metrics
        exact_accuracy = np.mean(y_test == y_pred_rounded)
        print(f"   Exact position accuracy: {exact_accuracy:.1%}")
        
        for n in [1, 2, 3, 5]:
            within_n = np.mean(np.abs(y_test - y_pred_rounded) <= n)
            print(f"   Within {n} position(s): {within_n:.1%}")
        
        return mae, rmse, exact_accuracy
    
    def predict_2025_with_real_data(self, feature_columns):
        """Generate 2025 predictions using real F1 data insights"""
        print("\\n🔮 2025 PREDICTIONS WITH REAL DATA")
        print("="*50)
        
        # Get current F1 drivers (2020 data as latest available)
        latest_year_data = self.processed_data[self.processed_data['year'] == self.processed_data['year'].max()]
        current_drivers = latest_year_data.groupby('driverId').last().reset_index()
        
        # Ensure we have 20 drivers for 2025
        if len(current_drivers) < 20:
            print(f"⚠️  Only {len(current_drivers)} drivers in latest data. Using available drivers.")
        
        # Update for 2025 prediction
        current_drivers['year'] = 2025
        current_drivers['round'] = 1
        current_drivers['circuitId'] = 1  # Bahrain (typical season opener)
        
        # Prepare features
        X_2025 = current_drivers[feature_columns].copy()
        
        # Handle categorical encoding
        for col in ['constructorId', 'circuitId']:
            if col in X_2025.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                X_2025[col] = X_2025[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )
        
        # Scale features
        X_2025_scaled = self.scaler.transform(X_2025)
        
        # Make predictions
        predictions = self.model.predict(X_2025_scaled).flatten()
        
        # Assign unique positions
        driver_pred_pairs = list(zip(current_drivers['driverId'], predictions))
        driver_pred_pairs.sort(key=lambda x: x[1])  # Sort by predicted value
        
        # Create results
        results_2025 = []
        for position, (driver_id, pred_val) in enumerate(driver_pred_pairs, 1):
            results_2025.append({
                'driverId': driver_id,
                'predicted_position': position,
                'prediction_value': pred_val
            })
        
        results_df = pd.DataFrame(results_2025)
        
        # Add driver and constructor info
        results_df = results_df.merge(self.datasets['drivers'][['driverId', 'forename', 'surname']], on='driverId')
        results_df['driver_name'] = results_df['forename'] + ' ' + results_df['surname']
        
        results_df = results_df.merge(current_drivers[['driverId', 'constructorId']], on='driverId')
        results_df = results_df.merge(self.datasets['constructors'][['constructorId', 'name']], on='constructorId')
        results_df.rename(columns={'name': 'team'}, inplace=True)
        
        # Sort by position
        results_df = results_df.sort_values('predicted_position')
        
        print("\\n🏁 2025 F1 PREDICTIONS (REAL DATA)")
        print("="*70)
        for _, row in results_df.iterrows():
            print(f"P{row['predicted_position']:2d}: {row['driver_name']:<25} ({row['team']:<20})")
        
        # Save results
        results_df.to_csv('f1_2025_real_predictions.csv', index=False)
        print(f"\\n💾 Real data predictions saved to 'f1_2025_real_predictions.csv'")
        
        return results_df
    
    def run_real_data_pipeline(self):
        """Run the complete pipeline with real Kaggle data"""
        print("🏎️ REAL KAGGLE F1 PREDICTION PIPELINE")
        print("="*60)
        
        # Check and load real data
        if not self.check_real_dataset():
            return None
        
        if not self.load_real_data():
            return None
        
        # Analyze data quality
        self.analyze_real_data_quality()
        
        # Create enhanced features
        self.create_enhanced_features()
        
        # Train model
        history, X_test, y_test, feature_columns = self.train_with_real_data()
        
        # Generate 2025 predictions
        predictions_2025 = self.predict_2025_with_real_data(feature_columns)
        
        print("\\n🏆 REAL DATA PIPELINE COMPLETE!")
        print("The model is now trained on authentic Formula 1 data!")
        
        return predictions_2025

def main():
    """Main function to run real data predictions"""
    predictor = RealKaggleF1Predictor()
    predictions = predictor.run_real_data_pipeline()
    
    if predictions is not None:
        print("\\n✅ Success! Real F1 data predictions generated.")
    else:
        print("\\n❌ Could not complete real data predictions.")
        print("Please ensure the Kaggle F1 dataset files are available.")

if __name__ == "__main__":
    main()