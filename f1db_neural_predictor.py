#!/usr/bin/env python3
"""
F1DB Live Data Neural Network Predictor

This system uses the live F1DB database (https://github.com/f1db/f1db) 
which gets updated after every race weekend to provide the most accurate
and up-to-date Formula 1 position predictions.
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
import requests
import zipfile
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class F1DBNeuralPredictor:
    def __init__(self):
        """Initialize F1DB Neural Network Predictor"""
        self.f1db_version = None
        self.datasets = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model = None
        self.features_df = None
        
    def check_f1db_update(self):
        """Check for latest F1DB release and download if needed"""
        print("🔄 CHECKING FOR LATEST F1DB DATA")
        print("="*50)
        
        try:
            # Get latest release info
            response = requests.get('https://api.github.com/repos/f1db/f1db/releases/latest')
            release_data = response.json()
            
            latest_version = release_data['tag_name']
            release_date = release_data['published_at']
            
            print(f"📊 Latest F1DB Version: {latest_version}")
            print(f"📅 Release Date: {release_date}")
            print(f"📝 Description: {release_data['name']}")
            
            # Check if we already have this version
            if os.path.exists('f1db_version.txt'):
                with open('f1db_version.txt', 'r') as f:
                    current_version = f.read().strip()
                
                if current_version == latest_version:
                    print(f"✅ Already have latest version: {latest_version}")
                    self.f1db_version = latest_version
                    return True
            
            # Download latest CSV data
            csv_asset = None
            for asset in release_data['assets']:
                if asset['name'] == 'f1db-csv.zip':
                    csv_asset = asset
                    break
            
            if csv_asset:
                download_url = csv_asset['browser_download_url']
                print(f"📥 Downloading latest F1DB CSV data ({csv_asset['size'] / 1024 / 1024:.1f} MB)...")
                
                # Download and extract
                response = requests.get(download_url)
                with open('f1db-csv-latest.zip', 'wb') as f:
                    f.write(response.content)
                
                # Extract CSV files
                with zipfile.ZipFile('f1db-csv-latest.zip', 'r') as zip_ref:
                    zip_ref.extractall('.')
                
                # Clean up
                os.remove('f1db-csv-latest.zip')
                
                # Save version info
                with open('f1db_version.txt', 'w') as f:
                    f.write(latest_version)
                
                self.f1db_version = latest_version
                print(f"✅ Downloaded and extracted F1DB {latest_version}")
                return True
            else:
                print("❌ CSV data not found in latest release")
                return False
                
        except Exception as e:
            print(f"⚠️ Error checking F1DB updates: {e}")
            print("Proceeding with existing data if available...")
            return self._check_existing_data()
    
    def _check_existing_data(self):
        """Check if F1DB data files exist locally"""
        required_files = [
            'f1db-races.csv',
            'f1db-races-race-results.csv', 
            'f1db-drivers.csv',
            'f1db-constructors.csv',
            'f1db-circuits.csv'
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            print(f"❌ Missing F1DB files: {missing_files}")
            return False
        
        print("✅ F1DB data files found locally")
        return True
    
    def load_f1db_data(self):
        """Load F1DB datasets"""
        print("\n📥 LOADING F1DB LIVE DATA")
        print("="*50)
        
        try:
            # Load core datasets
            self.datasets['races'] = pd.read_csv('f1db-races.csv')
            self.datasets['race_results'] = pd.read_csv('f1db-races-race-results.csv')
            self.datasets['drivers'] = pd.read_csv('f1db-drivers.csv')
            self.datasets['constructors'] = pd.read_csv('f1db-constructors.csv')
            self.datasets['circuits'] = pd.read_csv('f1db-circuits.csv')
            
            # Load additional datasets
            optional_datasets = {
                'qualifying_results': 'f1db-races-qualifying-results.csv',
                'driver_standings': 'f1db-races-driver-standings.csv',
                'constructor_standings': 'f1db-races-constructor-standings.csv',
                'pit_stops': 'f1db-races-pit-stops.csv',
                'fastest_laps': 'f1db-races-fastest-laps.csv'
            }
            
            for name, filename in optional_datasets.items():
                try:
                    self.datasets[name] = pd.read_csv(filename)
                    print(f"✅ Loaded {filename}")
                except FileNotFoundError:
                    print(f"⚠️ {filename} not found, skipping")
            
            print(f"\n📊 F1DB DATASET OVERVIEW:")
            for name, df in self.datasets.items():
                if 'year' in df.columns:
                    year_range = df['year'].agg(['min', 'max'])
                    print(f"   {name:<20}: {len(df):,} rows ({year_range['min']}-{year_range['max']})")
                else:
                    print(f"   {name:<20}: {len(df):,} rows")
            
            # Show latest data info
            latest_race = self.datasets['races'].iloc[-1]
            print(f"\n🏁 Latest Race: {latest_race['officialName']} ({latest_race['date']})")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading F1DB data: {e}")
            return False
    
    def analyze_f1db_data(self):
        """Analyze F1DB data quality and coverage"""
        print("\n🔬 ANALYZING F1DB DATA QUALITY")
        print("="*50)
        
        races = self.datasets['races']
        results = self.datasets['race_results']
        drivers = self.datasets['drivers']
        
        # Temporal analysis
        print("📅 TEMPORAL COVERAGE:")
        year_range = races['year'].agg(['min', 'max'])
        total_years = year_range['max'] - year_range['min'] + 1
        print(f"   Years: {year_range['min']} - {year_range['max']} ({total_years} years)")
        
        # Current season analysis
        current_year = year_range['max']
        current_season_races = races[races['year'] == current_year]
        print(f"   Current season ({current_year}): {len(current_season_races)} races completed")
        
        # Modern era analysis (2000+)
        modern_races = races[races['year'] >= 2000]
        modern_results = results[results['year'] >= 2000]
        print(f"   Modern era (2000+): {len(modern_races)} races, {len(modern_results):,} results")
        
        # Driver analysis
        print(f"\n👥 DRIVER ANALYSIS:")
        print(f"   Total drivers in F1 history: {len(drivers):,}")
        
        # Active drivers (recent races)
        recent_results = results[results['year'] >= current_year - 1]
        active_drivers = recent_results['driverId'].nunique()
        print(f"   Recent active drivers: {active_drivers}")
        
        # Constructor analysis
        constructors = self.datasets['constructors']
        print(f"\n🏎️ CONSTRUCTOR ANALYSIS:")
        print(f"   Total constructors in F1 history: {len(constructors):,}")
        
        recent_constructors = recent_results['constructorId'].nunique()
        print(f"   Recent active constructors: {recent_constructors}")
        
        # Data quality metrics
        print(f"\n📊 DATA QUALITY:")
        print(f"   Total race results: {len(results):,}")
        print(f"   Results with valid positions: {len(results[results['positionNumber'].notna()]):,}")
        print(f"   Results with lap times: {len(results[results['timeMillis'].notna()]):,}")
        
        return True
    
    def create_f1db_features(self):
        """Create features using F1DB data structure"""
        print("\n🚀 CREATING FEATURES FROM F1DB DATA")
        print("="*60)
        
        # Merge core datasets
        results = self.datasets['race_results'].copy()
        races = self.datasets['races'].copy()
        drivers = self.datasets['drivers'].copy()
        constructors = self.datasets['constructors'].copy()
        circuits = self.datasets['circuits'].copy()
        
        # Filter for valid race results (exclude DNF, DNS, etc.)
        results = results[results['positionNumber'].notna()].copy()
        results = results[results['positionNumber'] > 0].copy()
        
        print(f"Valid race results: {len(results):,}")
        
        # Focus on modern era (2000+) for better 2025 relevance
        modern_races = races[races['year'] >= 2000]
        results = results.merge(modern_races[['id']], left_on='raceId', right_on='id')
        
        print(f"Modern era results (2000+): {len(results):,}")
        
        # Merge all data
        df = results.merge(races, left_on='raceId', right_on='id', suffixes=('', '_race'))
        df = df.merge(drivers, left_on='driverId', right_on='id', suffixes=('', '_driver'))
        df = df.merge(constructors, left_on='constructorId', right_on='id', suffixes=('', '_constructor'))
        df = df.merge(circuits, left_on='circuitId', right_on='id', suffixes=('', '_circuit'))
        
        # Add qualifying data if available
        if 'qualifying_results' in self.datasets:
            qualifying = self.datasets['qualifying_results']
            qualifying_modern = qualifying[qualifying['year'] >= 2000]
            df = df.merge(
                qualifying_modern[['raceId', 'driverId', 'positionNumber']], 
                on=['raceId', 'driverId'], 
                suffixes=('', '_qual'), 
                how='left'
            )
            print("✅ Added qualifying data")
        
        # Convert date and sort
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['date', 'driverId']).reset_index(drop=True)
        
        print(f"Merged F1DB dataset: {df.shape}")
        
        # Create comprehensive features
        features = []
        total_rows = len(df)
        
        print("🔄 Engineering F1DB features...")
        
        for idx, row in df.iterrows():
            if idx % 2000 == 0:
                print(f"   Progress: {idx:,}/{total_rows:,} ({idx/total_rows*100:.1f}%)")
            
            feature_row = self._create_f1db_feature_row(row, df, idx)
            features.append(feature_row)
        
        features_df = pd.DataFrame(features)
        features_df = features_df.dropna(subset=['target_position'])
        
        print(f"\n✅ F1DB features created: {features_df.shape}")
        print(f"Feature columns: {list(features_df.columns)}")
        
        self.features_df = features_df
        return features_df
    
    def _create_f1db_feature_row(self, row, df, idx):
        """Create feature row using F1DB data structure"""
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
            'grid_position': row['gridPositionNumber'] if pd.notna(row['gridPositionNumber']) else 20,
            'target_position': row['positionNumber']
        }
        
        # Add qualifying position if available
        if 'positionNumber_qual' in row and pd.notna(row['positionNumber_qual']):
            feature_row['qualifying_position'] = row['positionNumber_qual']
        else:
            feature_row['qualifying_position'] = feature_row['grid_position']
        
        # Circuit characteristics (F1DB has detailed circuit info)
        feature_row['circuit_length'] = row['courseLength'] if pd.notna(row['courseLength']) else 5.0
        feature_row['circuit_turns'] = row['turns'] if pd.notna(row['turns']) else 15
        feature_row['circuit_laps'] = row['laps'] if pd.notna(row['laps']) else 50
        feature_row['circuit_type'] = 1 if row.get('circuitType') == 'STREET' else 0
        
        # Historical data for this driver
        historical_mask = (df['driverId'] == driver_id) & (df['date'] < current_date)
        historical_data = df[historical_mask]
        
        if len(historical_data) > 0:
            # Recent performance (last 5 races)
            recent_5 = historical_data.tail(5)
            feature_row['avg_position_5'] = recent_5['positionNumber'].mean()
            feature_row['avg_points_5'] = recent_5['points'].mean()
            feature_row['avg_grid_5'] = recent_5['gridPositionNumber'].mean()
            
            # Medium term (last 10 races)
            recent_10 = historical_data.tail(10)
            feature_row['avg_position_10'] = recent_10['positionNumber'].mean()
            feature_row['avg_points_10'] = recent_10['points'].mean()
            
            # Season performance
            season_data = historical_data[historical_data['year'] == current_year]
            feature_row['season_avg_position'] = season_data['positionNumber'].mean() if len(season_data) > 0 else 10.5
            feature_row['season_points'] = season_data['points'].sum() if len(season_data) > 0 else 0
            feature_row['season_races'] = len(season_data)
            
            # Career statistics
            feature_row['career_avg_position'] = historical_data['positionNumber'].mean()
            feature_row['career_total_points'] = historical_data['points'].sum()
            feature_row['career_races'] = len(historical_data)
            feature_row['career_wins'] = len(historical_data[historical_data['positionNumber'] == 1])
            feature_row['career_podiums'] = len(historical_data[historical_data['positionNumber'] <= 3])
            feature_row['career_top10s'] = len(historical_data[historical_data['positionNumber'] <= 10])
            
            # Recent form indicators
            recent_3 = historical_data.tail(3)
            feature_row['recent_form_3'] = recent_3['positionNumber'].mean()
            feature_row['recent_wins_5'] = len(recent_5[recent_5['positionNumber'] == 1])
            feature_row['recent_podiums_5'] = len(recent_5[recent_5['positionNumber'] <= 3])
            
            # Performance trends
            if len(historical_data) >= 10:
                first_half = historical_data.head(len(historical_data)//2)
                second_half = historical_data.tail(len(historical_data)//2)
                trend = first_half['positionNumber'].mean() - second_half['positionNumber'].mean()
                feature_row['performance_trend'] = trend  # Positive = improving
            else:
                feature_row['performance_trend'] = 0
                
        else:
            # Default values for new drivers
            defaults = {
                'avg_position_5': 15.0, 'avg_points_5': 0, 'avg_grid_5': 15.0,
                'avg_position_10': 15.0, 'avg_points_10': 0, 'season_avg_position': 15.0,
                'season_points': 0, 'season_races': 0, 'career_avg_position': 15.0,
                'career_total_points': 0, 'career_races': 0, 'career_wins': 0,
                'career_podiums': 0, 'career_top10s': 0, 'recent_form_3': 15.0,
                'recent_wins_5': 0, 'recent_podiums_5': 0, 'performance_trend': 0
            }
            feature_row.update(defaults)
        
        # Constructor performance
        constructor_mask = (df['constructorId'] == constructor_id) & (df['date'] < current_date)
        constructor_data = df[constructor_mask]
        
        if len(constructor_data) > 0:
            constructor_recent = constructor_data.tail(20)
            feature_row['constructor_avg_position'] = constructor_recent['positionNumber'].mean()
            feature_row['constructor_avg_points'] = constructor_recent['points'].mean()
            feature_row['constructor_recent_wins'] = len(constructor_recent[constructor_recent['positionNumber'] == 1])
            feature_row['constructor_recent_podiums'] = len(constructor_recent[constructor_recent['positionNumber'] <= 3])
        else:
            feature_row['constructor_avg_position'] = 10.5
            feature_row['constructor_avg_points'] = 0
            feature_row['constructor_recent_wins'] = 0
            feature_row['constructor_recent_podiums'] = 0
        
        # Circuit-specific performance
        circuit_driver_mask = (df['driverId'] == driver_id) & (df['circuitId'] == circuit_id) & (df['date'] < current_date)
        circuit_driver_data = df[circuit_driver_mask]
        
        if len(circuit_driver_data) > 0:
            feature_row['circuit_driver_avg_position'] = circuit_driver_data['positionNumber'].mean()
            feature_row['circuit_driver_best_position'] = circuit_driver_data['positionNumber'].min()
            feature_row['circuit_driver_races'] = len(circuit_driver_data)
        else:
            feature_row['circuit_driver_avg_position'] = 10.5
            feature_row['circuit_driver_best_position'] = 20
            feature_row['circuit_driver_races'] = 0
        
        # F1 era features (important for rule changes)
        if current_year >= 2022:  # Current regulations
            feature_row['era_current_regs'] = 1
        else:
            feature_row['era_current_regs'] = 0
            
        if current_year >= 2014:  # Turbo hybrid era
            feature_row['era_turbo_hybrid'] = 1
        else:
            feature_row['era_turbo_hybrid'] = 0
        
        return feature_row
    
    def build_f1db_model(self, input_dim):
        """Build enhanced neural network for F1DB data"""
        print("\n🧠 BUILDING F1DB NEURAL NETWORK")
        print("="*50)
        
        # Enhanced architecture for comprehensive F1DB data
        model = keras.Sequential([
            # Input layer
            layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            
            # Deep hidden layers for complex F1 patterns
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
            
            # Output layer
            layers.Dense(1, activation='linear')
        ])
        
        # Enhanced compilation for F1DB data
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0003),
            loss='huber',  # Robust to outliers
            metrics=['mae', 'mse']
        )
        
        print("F1DB Neural Network Architecture:")
        model.summary()
        
        self.model = model
        return model
    
    def train_f1db_model(self):
        """Train the model with F1DB data"""
        print("\n🎓 TRAINING WITH F1DB LIVE DATA")
        print("="*50)
        
        if self.features_df is None:
            print("❌ No features available. Run create_f1db_features() first.")
            return None
        
        # Prepare features
        feature_columns = [col for col in self.features_df.columns 
                          if col not in ['target_position', 'driverId']]
        
        X = self.features_df[feature_columns].copy()
        y = self.features_df['target_position'].copy()
        
        print(f"Training with {len(X):,} samples and {len(feature_columns)} features")
        print(f"Data spans: {self.features_df['year'].min()}-{self.features_df['year'].max()}")
        
        # Handle categorical variables
        categorical_columns = ['constructorId', 'circuitId']
        for col in categorical_columns:
            if col in X.columns:
                le = LabelEncoder()
                # Handle string IDs from F1DB
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data with temporal awareness
        # Use recent years for testing to simulate real prediction scenario
        test_years = [self.features_df['year'].max(), self.features_df['year'].max() - 1]
        test_mask = self.features_df['year'].isin(test_years)
        
        X_train = X_scaled[~test_mask]
        X_test = X_scaled[test_mask]
        y_train = y[~test_mask]
        y_test = y[test_mask]
        
        print(f"Training set: {X_train.shape} (historical data)")
        print(f"Test set: {X_test.shape} (recent years: {test_years})")
        
        # Build and train model
        self.build_f1db_model(X_train.shape[1])
        
        # Enhanced training callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=12, min_lr=1e-8),
            keras.callbacks.ModelCheckpoint('f1db_best_model.h5', save_best_only=True, monitor='val_loss')
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=200,  # More epochs for comprehensive F1DB data
            batch_size=128,  # Larger batch size for big dataset
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate performance
        self.evaluate_f1db_model(X_test, y_test)
        
        return history, X_test, y_test, feature_columns
    
    def evaluate_f1db_model(self, X_test, y_test):
        """Evaluate model performance with F1DB data"""
        print("\n📊 F1DB MODEL EVALUATION")
        print("="*50)
        
        # Predictions
        y_pred = self.model.predict(X_test).flatten()
        y_pred_rounded = np.clip(np.round(y_pred), 1, 20)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred_rounded)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_rounded))
        
        print(f"🎯 F1DB MODEL PERFORMANCE:")
        print(f"   Mean Absolute Error: {mae:.2f} positions")
        print(f"   Root Mean Square Error: {rmse:.2f} positions")
        
        # Accuracy metrics
        exact_accuracy = np.mean(y_test == y_pred_rounded)
        print(f"   Exact position accuracy: {exact_accuracy:.1%}")
        
        for n in [1, 2, 3, 5]:
            within_n = np.mean(np.abs(y_test - y_pred_rounded) <= n)
            print(f"   Within {n} position(s): {within_n:.1%}")
        
        # Create evaluation plots
        self._create_evaluation_plots(y_test, y_pred_rounded)
        
        return mae, rmse, exact_accuracy
    
    def _create_evaluation_plots(self, y_test, y_pred):
        """Create comprehensive evaluation plots"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Predictions vs Actual
        axes[0, 0].scatter(y_test, y_pred, alpha=0.6, color='#e10600')
        axes[0, 0].plot([1, 20], [1, 20], 'gold', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Actual Position')
        axes[0, 0].set_ylabel('Predicted Position')
        axes[0, 0].set_title('F1DB Model: Predicted vs Actual Positions')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Error distribution
        residuals = y_test - y_pred
        axes[0, 1].hist(residuals, bins=30, alpha=0.7, color='#ff6b00')
        axes[0, 1].set_xlabel('Prediction Error (Actual - Predicted)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Prediction Error Distribution')
        axes[0, 1].axvline(x=0, color='gold', linestyle='--', alpha=0.7)
        
        # Error by position
        error_df = pd.DataFrame({'actual': y_test, 'error': np.abs(residuals)})
        error_by_pos = error_df.groupby('actual')['error'].mean()
        axes[1, 0].bar(error_by_pos.index, error_by_pos.values, color='#ffd700')
        axes[1, 0].set_xlabel('Actual Position')
        axes[1, 0].set_ylabel('Mean Absolute Error')
        axes[1, 0].set_title('Prediction Error by Position')
        
        # Accuracy by position range
        position_ranges = [(1, 3), (4, 6), (7, 10), (11, 15), (16, 20)]
        range_accuracies = []
        range_labels = []
        
        for start, end in position_ranges:
            mask = (y_test >= start) & (y_test <= end)
            if mask.sum() > 0:
                accuracy = np.mean(np.abs(y_test[mask] - y_pred[mask]) <= 2)
                range_accuracies.append(accuracy)
                range_labels.append(f'P{start}-P{end}')
        
        axes[1, 1].bar(range_labels, range_accuracies, color='#00ff00')
        axes[1, 1].set_ylabel('Accuracy (Within 2 Positions)')
        axes[1, 1].set_title('Accuracy by Position Range')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('f1db_model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 F1DB evaluation plots saved as 'f1db_model_evaluation.png'")
    
    def predict_next_race_2025(self, feature_columns):
        """Predict the next race in 2025 using F1DB data"""
        print("\n🔮 PREDICTING NEXT 2025 RACE WITH F1DB DATA")
        print("="*60)
        
        # Get the most recent race to understand current grid
        latest_race = self.datasets['races'].iloc[-1]
        latest_results = self.datasets['race_results'][
            self.datasets['race_results']['raceId'] == latest_race['id']
        ]
        
        print(f"📊 Latest race: {latest_race['officialName']} ({latest_race['date']})")
        print(f"🏁 Results from latest race: {len(latest_results)} drivers")
        
        # Get current active drivers from recent races
        recent_years = [2024, 2025]
        recent_results = self.datasets['race_results'][
            self.datasets['race_results']['year'].isin(recent_years)
        ]
        
        # Get most recent data for each active driver
        current_drivers = recent_results.groupby('driverId').last().reset_index()
        current_drivers = current_drivers.head(20)  # Top 20 most recent drivers
        
        print(f"🏎️ Active drivers for 2025 prediction: {len(current_drivers)}")
        
        # Create prediction features for next race
        next_race_features = []
        
        for _, driver_row in current_drivers.iterrows():
            # Get latest feature data for this driver
            driver_features = self.features_df[
                self.features_df['driverId'] == driver_row['driverId']
            ].iloc[-1].copy() if len(self.features_df[self.features_df['driverId'] == driver_row['driverId']]) > 0 else None
            
            if driver_features is not None:
                # Update for next race prediction
                driver_features['year'] = 2025
                driver_features['round'] = latest_race['round'] + 1  # Next race
                
                # Predict next circuit (use most common first race circuit)
                next_circuit = 'bahrain'  # Typical season opener
                driver_features['circuitId'] = next_circuit
                
                next_race_features.append(driver_features)
        
        if not next_race_features:
            print("❌ No driver data available for prediction")
            return None
        
        # Convert to DataFrame and prepare for prediction
        prediction_df = pd.DataFrame(next_race_features)
        X_next = prediction_df[feature_columns].copy()
        
        # Handle categorical encoding
        for col in ['constructorId', 'circuitId']:
            if col in X_next.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                # Convert to string and handle unseen categories
                X_next[col] = X_next[col].astype(str)
                X_next[col] = X_next[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )
        
        # Scale features
        X_next_scaled = self.scaler.transform(X_next)
        
        # Make predictions
        predictions = self.model.predict(X_next_scaled).flatten()
        
        # Create results with unique positions
        driver_pred_pairs = list(zip(prediction_df['driverId'], predictions))
        driver_pred_pairs.sort(key=lambda x: x[1])  # Sort by predicted value
        
        # Create final results
        results_2025 = []
        for position, (driver_id, pred_val) in enumerate(driver_pred_pairs, 1):
            results_2025.append({
                'driverId': driver_id,
                'predicted_position': position,
                'prediction_value': pred_val
            })
        
        results_df = pd.DataFrame(results_2025)
        
        # Add driver information
        drivers = self.datasets['drivers']
        results_df = results_df.merge(
            drivers[['id', 'firstName', 'lastName', 'fullName']], 
            left_on='driverId', right_on='id', how='left'
        )
        
        # Add constructor information
        constructors = self.datasets['constructors']
        results_df = results_df.merge(
            prediction_df[['driverId', 'constructorId']], 
            on='driverId', how='left'
        )
        results_df = results_df.merge(
            constructors[['id', 'name']], 
            left_on='constructorId', right_on='id', 
            how='left', suffixes=('', '_constructor')
        )
        
        # Clean up and sort
        results_df['driver_name'] = results_df['fullName'].fillna(
            results_df['firstName'] + ' ' + results_df['lastName']
        )
        results_df['team'] = results_df['name'].fillna('Unknown Team')
        results_df = results_df.sort_values('predicted_position')
        
        # Display results
        print(f"\n🏁 NEXT 2025 F1 RACE PREDICTIONS (F1DB v{self.f1db_version})")
        print("="*70)
        
        for _, row in results_df.iterrows():
            pos_icon = "🥇" if row['predicted_position'] == 1 else "🥈" if row['predicted_position'] == 2 else "🥉" if row['predicted_position'] == 3 else "🏁" if row['predicted_position'] <= 10 else "  "
            print(f"{pos_icon} P{row['predicted_position']:2d}: {row['driver_name']:<25} ({row['team']:<20})")
        
        # Save predictions
        results_df.to_csv('f1db_2025_predictions.csv', index=False)
        print(f"\n💾 F1DB predictions saved to 'f1db_2025_predictions.csv'")
        
        # Create visualization
        self._create_prediction_visualization(results_df)
        
        return results_df
    
    def _create_prediction_visualization(self, results_df):
        """Create visualization for F1DB predictions"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Podium
        podium = results_df.head(3)
        colors = ['gold', 'silver', '#CD7F32']
        bars = axes[0, 0].bar(range(3), [3, 2, 1], color=colors)
        axes[0, 0].set_xticks(range(3))
        axes[0, 0].set_xticklabels([f"{row['driver_name']}\n{row['team']}" for _, row in podium.iterrows()], 
                                  rotation=0, ha='center')
        axes[0, 0].set_ylabel('Podium Position')
        axes[0, 0].set_title(f'🏆 F1DB 2025 Podium Prediction (v{self.f1db_version})')
        axes[0, 0].invert_yaxis()
        
        # Team performance
        team_avg = results_df.groupby('team')['predicted_position'].mean().sort_values()
        axes[0, 1].barh(range(len(team_avg)), team_avg.values, color='#e10600')
        axes[0, 1].set_yticks(range(len(team_avg)))
        axes[0, 1].set_yticklabels(team_avg.index, fontsize=9)
        axes[0, 1].set_xlabel('Average Predicted Position')
        axes[0, 1].set_title('🏎️ Constructor Performance Ranking')
        axes[0, 1].invert_yaxis()
        
        # Position distribution
        position_counts = results_df['predicted_position'].value_counts().sort_index()
        axes[1, 0].bar(position_counts.index, position_counts.values, color='#ffd700')
        axes[1, 0].set_xlabel('Predicted Position')
        axes[1, 0].set_ylabel('Number of Drivers')
        axes[1, 0].set_title('Position Distribution')
        
        # Championship points prediction
        points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
        results_df['predicted_points'] = results_df['predicted_position'].map(points_map).fillna(0)
        team_points = results_df.groupby('team')['predicted_points'].sum().sort_values(ascending=False)
        
        axes[1, 1].bar(range(len(team_points)), team_points.values, color='#00ff00')
        axes[1, 1].set_xticks(range(len(team_points)))
        axes[1, 1].set_xticklabels(team_points.index, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Predicted Points')
        axes[1, 1].set_title('Constructor Championship Points')
        
        plt.tight_layout()
        plt.savefig('f1db_2025_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 F1DB prediction visualization saved as 'f1db_2025_predictions.png'")
    
    def run_f1db_pipeline(self):
        """Run the complete F1DB prediction pipeline"""
        print("🏎️ F1DB LIVE DATA NEURAL NETWORK PREDICTOR")
        print("="*70)
        print("Using live F1 database: https://github.com/f1db/f1db")
        print("Data updated after every race weekend!")
        print("="*70)
        
        # Check for updates and load data
        if not self.check_f1db_update():
            print("❌ Failed to get F1DB data")
            return None
        
        if not self.load_f1db_data():
            print("❌ Failed to load F1DB data")
            return None
        
        # Analyze data
        self.analyze_f1db_data()
        
        # Create features and train
        self.create_f1db_features()
        history, X_test, y_test, feature_columns = self.train_f1db_model()
        
        # Generate 2025 predictions
        predictions_2025 = self.predict_next_race_2025(feature_columns)
        
        print("\n🏆 F1DB PIPELINE COMPLETE!")
        print(f"✅ Using authentic F1 data (v{self.f1db_version})")
        print("✅ Model trained on comprehensive historical data")
        print("✅ 2025 predictions generated with live data")
        
        return predictions_2025

def main():
    """Main function to run F1DB predictions"""
    predictor = F1DBNeuralPredictor()
    predictions = predictor.run_f1db_pipeline()
    
    if predictions is not None:
        print("\n🎉 SUCCESS! F1DB neural network predictions complete!")
        print("📊 Check 'f1db_2025_predictions.csv' for detailed results")
    else:
        print("\n❌ F1DB prediction pipeline failed")
        print("🔧 Please check F1DB data availability and try again")
    
    return predictor, predictions

if __name__ == "__main__":
    predictor, predictions = main()