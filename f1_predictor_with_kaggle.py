#!/usr/bin/env python3
"""
F1 Neural Network Predictor with Automatic Kaggle Dataset Detection

This script automatically detects whether the real Kaggle F1 dataset is available
and adapts the neural network accordingly for optimal predictions.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

class AdaptiveF1Predictor:
    def __init__(self):
        self.using_real_data = False
        self.data_source = "sample"
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model = None
        
    def detect_data_source(self):
        """Automatically detect whether real Kaggle data is available"""
        print("🔍 DETECTING DATA SOURCE")
        print("="*40)
        
        # Check for real Kaggle dataset files
        kaggle_files = ['circuits.csv', 'constructors.csv', 'drivers.csv', 'races.csv', 'results.csv']
        real_data_available = all(os.path.exists(f) for f in kaggle_files)
        
        if real_data_available:
            # Verify it's actually the Kaggle dataset by checking structure
            try:
                results = pd.read_csv('results.csv')
                races = pd.read_csv('races.csv')
                
                # Check for Kaggle dataset characteristics
                has_historical_data = races['year'].min() <= 1960  # Should go back to 1950s
                has_modern_data = races['year'].max() >= 2015
                has_large_dataset = len(results) > 10000  # Kaggle dataset is large
                
                if has_historical_data and has_modern_data and has_large_dataset:
                    self.using_real_data = True
                    self.data_source = "kaggle"
                    print("✅ REAL KAGGLE DATASET DETECTED!")
                    print(f"   📊 {len(results):,} race results from {races['year'].min()}-{races['year'].max()}")
                    return True
                    
            except Exception as e:
                print(f"⚠️ Error verifying real data: {e}")
        
        # Check for sample data
        if os.path.exists('f1_data'):
            print("✅ Using sample F1 dataset")
            self.data_source = "sample"
            return True
        
        print("❌ No dataset found!")
        return False
    
    def load_data(self):
        """Load data based on detected source"""
        if self.using_real_data:
            return self._load_real_kaggle_data()
        else:
            return self._load_sample_data()
    
    def _load_real_kaggle_data(self):
        """Load the real Kaggle F1 dataset"""
        print("\\n📥 LOADING REAL KAGGLE F1 DATASET")
        print("="*50)
        
        # Load all available datasets
        self.circuits = pd.read_csv('circuits.csv')
        self.constructors = pd.read_csv('constructors.csv')
        self.drivers = pd.read_csv('drivers.csv')
        self.races = pd.read_csv('races.csv')
        self.results = pd.read_csv('results.csv')
        
        # Handle real data quirks
        # Convert \\N to NaN and handle position data
        if self.results['position'].dtype == 'object':
            self.results = self.results[self.results['position'] != '\\\\N']
            self.results['position'] = pd.to_numeric(self.results['position'])
        
        # Load optional datasets
        try:
            self.qualifying = pd.read_csv('qualifying.csv')
            print("✅ Loaded qualifying data")
        except FileNotFoundError:
            # Create minimal qualifying structure
            self.qualifying = pd.DataFrame({
                'qualifyId': range(len(self.results)),
                'raceId': self.results['raceId'],
                'driverId': self.results['driverId'],
                'position': self.results['grid']  # Use grid position as fallback
            })
            print("⚠️ Created basic qualifying from grid positions")
        
        # Focus on modern era (2000+) for better 2025 predictions
        modern_races = self.races[self.races['year'] >= 2000]
        self.results = self.results.merge(modern_races[['raceId']], on='raceId')
        self.qualifying = self.qualifying.merge(modern_races[['raceId']], on='raceId')
        
        print(f"\\n📊 REAL DATASET LOADED:")
        print(f"   Circuits: {len(self.circuits):,}")
        print(f"   Drivers: {len(self.drivers):,}")
        print(f"   Constructors: {len(self.constructors):,}")
        print(f"   Modern races (2000+): {len(modern_races):,}")
        print(f"   Modern results: {len(self.results):,}")
        print(f"   Year range: {modern_races['year'].min()}-{modern_races['year'].max()}")
        
        return True
    
    def _load_sample_data(self):
        """Load the sample F1 dataset"""
        print("\\n📥 LOADING SAMPLE F1 DATASET")
        print("="*40)
        
        self.circuits = pd.read_csv('f1_data/circuits.csv')
        self.drivers = pd.read_csv('f1_data/drivers.csv')
        self.constructors = pd.read_csv('f1_data/constructors.csv')
        self.races = pd.read_csv('f1_data/races.csv')
        self.results = pd.read_csv('f1_data/results.csv')
        self.qualifying = pd.read_csv('f1_data/qualifying.csv')
        
        print(f"Sample dataset loaded: {len(self.results):,} results")
        return True
    
    def create_adaptive_features(self):
        """Create features that work with both real and sample data"""
        print(f"\\n🔧 CREATING FEATURES FOR {self.data_source.upper()} DATA")
        print("="*50)
        
        # Merge datasets
        df = self.results.merge(self.races, on='raceId')
        df = df.merge(self.drivers, on='driverId')
        df = df.merge(self.constructors, on='constructorId')
        df = df.merge(self.circuits, on='circuitId')
        df = df.merge(self.qualifying, on=['raceId', 'driverId'], suffixes=('', '_qual'), how='left')
        
        # Handle missing qualifying data
        if 'position_qual' not in df.columns:
            df['position_qual'] = df['grid']  # Use grid as fallback
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['date', 'driverId']).reset_index(drop=True)
        
        print(f"Merged dataset: {df.shape}")
        
        # Create features with enhanced logic for real data
        features = []
        total_rows = len(df)
        
        for idx, row in df.iterrows():
            if idx % (max(1, total_rows // 20)) == 0:  # Show progress
                print(f"   Progress: {idx:,}/{total_rows:,} ({idx/total_rows*100:.1f}%)")
            
            feature_row = self._create_adaptive_feature_row(row, df, idx)
            features.append(feature_row)
        
        features_df = pd.DataFrame(features)
        features_df = features_df.dropna(subset=['target_position'])
        
        print(f"\\n✅ Features created: {features_df.shape}")
        
        # Additional real data enhancements
        if self.using_real_data:
            features_df = self._add_real_data_enhancements(features_df, df)
        
        self.features_df = features_df
        return features_df
    
    def _create_adaptive_feature_row(self, row, df, idx):
        """Create feature row that adapts to data source"""
        driver_id = row['driverId']
        current_date = row['date']
        
        feature_row = {
            'driverId': driver_id,
            'constructorId': row['constructorId'],
            'circuitId': row['circuitId'],
            'year': row['year'],
            'round': row['round'],
            'grid_position': row['grid'],
            'qualifying_position': row['position_qual'],
            'target_position': row['position']
        }
        
        # Add circuit characteristics
        for col in ['lat', 'lng', 'alt']:
            if col in row:
                feature_row[f'circuit_{col}'] = row[col]
            else:
                feature_row[f'circuit_{col}'] = 0
        
        # Historical performance
        historical_data = df[(df['driverId'] == driver_id) & (df['date'] < current_date)]
        
        if len(historical_data) > 0:
            # Recent form
            for window in [3, 5, 10]:
                recent = historical_data.tail(window)
                feature_row[f'avg_position_{window}'] = recent['position'].mean()
                feature_row[f'avg_points_{window}'] = recent['points'].mean()
            
            # Career stats
            feature_row['career_avg_position'] = historical_data['position'].mean()
            feature_row['career_wins'] = len(historical_data[historical_data['position'] == 1])
            feature_row['career_podiums'] = len(historical_data[historical_data['position'] <= 3])
            feature_row['career_races'] = len(historical_data)
            
            # Season performance
            season_data = historical_data[historical_data['year'] == row['year']]
            feature_row['season_avg_position'] = season_data['position'].mean() if len(season_data) > 0 else 10.5
            feature_row['season_points'] = season_data['points'].sum() if len(season_data) > 0 else 0
        else:
            # Default values for new drivers
            defaults = {
                'avg_position_3': 15.0, 'avg_points_3': 0,
                'avg_position_5': 15.0, 'avg_points_5': 0,
                'avg_position_10': 15.0, 'avg_points_10': 0,
                'career_avg_position': 15.0, 'career_wins': 0,
                'career_podiums': 0, 'career_races': 0,
                'season_avg_position': 15.0, 'season_points': 0
            }
            feature_row.update(defaults)
        
        return feature_row
    
    def _add_real_data_enhancements(self, features_df, df):
        """Add enhancements specific to real Kaggle data"""
        print("🚀 Adding real data enhancements...")
        
        # Add era features for real data
        features_df['era_modern'] = (features_df['year'] >= 2009).astype(int)
        features_df['era_hybrid'] = (features_df['year'] >= 2014).astype(int)
        features_df['era_current'] = (features_df['year'] >= 2017).astype(int)
        
        print("✅ Added era-specific features")
        return features_df
    
    def build_adaptive_model(self, input_dim):
        """Build model adapted to the data source"""
        print(f"\\n🧠 BUILDING NEURAL NETWORK FOR {self.data_source.upper()} DATA")
        print("="*60)
        
        if self.using_real_data:
            # More complex model for real data
            model = keras.Sequential([
                layers.Dense(256, activation='relu', input_shape=(input_dim,)),
                layers.Dropout(0.3),
                layers.BatchNormalization(),
                
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.4),
                layers.BatchNormalization(),
                
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.3),
                layers.BatchNormalization(),
                
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.2),
                
                layers.Dense(64, activation='relu'),
                layers.Dense(1, activation='linear')
            ])
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                loss='huber',
                metrics=['mae']
            )
            
        else:
            # Simpler model for sample data
            model = keras.Sequential([
                layers.Dense(128, activation='relu', input_shape=(input_dim,)),
                layers.Dropout(0.3),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(64, activation='relu'),
                layers.Dense(1, activation='linear')
            ])
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
        
        print(f"Model architecture for {self.data_source} data:")
        model.summary()
        
        self.model = model
        return model
    
    def train_adaptive_model(self):
        """Train model with adaptive parameters"""
        print(f"\\n🎓 TRAINING WITH {self.data_source.upper()} DATA")
        print("="*50)
        
        # Prepare data
        feature_columns = [col for col in self.features_df.columns 
                          if col not in ['target_position', 'driverId']]
        
        X = self.features_df[feature_columns].copy()
        y = self.features_df['target_position'].copy()
        
        # Handle categorical variables
        categorical_columns = ['constructorId', 'circuitId']
        for col in categorical_columns:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                self.label_encoders[col] = le
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Build model
        self.build_adaptive_model(X_train.shape[1])
        
        # Adaptive training parameters
        if self.using_real_data:
            epochs = 100
            batch_size = 64
            patience = 20
        else:
            epochs = 50
            batch_size = 32
            patience = 15
        
        # Train model
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience//2, min_lr=1e-7)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return X_test, y_test, feature_columns, history
    
    def evaluate_and_predict(self, X_test, y_test, feature_columns):
        """Evaluate model and make 2025 predictions"""
        print(f"\\n📊 EVALUATION WITH {self.data_source.upper()} DATA")
        print("="*50)
        
        # Evaluate
        y_pred = self.model.predict(X_test).flatten()
        y_pred_rounded = np.clip(np.round(y_pred), 1, 20)
        
        mae = mean_absolute_error(y_test, y_pred_rounded)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_rounded))
        
        print(f"🎯 Model Performance ({self.data_source} data):")
        print(f"   MAE: {mae:.2f} positions")
        print(f"   RMSE: {rmse:.2f} positions")
        
        for n in [1, 2, 3, 5]:
            within_n = np.mean(np.abs(y_test - y_pred_rounded) <= n)
            print(f"   Within {n} positions: {within_n:.1%}")
        
        # Make 2025 predictions
        predictions_2025 = self._predict_2025(feature_columns)
        
        return predictions_2025
    
    def _predict_2025(self, feature_columns):
        """Generate 2025 predictions"""
        print(f"\\n🔮 2025 PREDICTIONS WITH {self.data_source.upper()} DATA")
        print("="*60)
        
        # Get latest data for each driver
        latest_data = self.features_df.groupby('driverId').last().reset_index()
        
        # Update for 2025
        latest_data['year'] = 2025
        latest_data['round'] = 1
        latest_data['circuitId'] = 1  # Bahrain
        
        # Prepare features
        X_2025 = latest_data[feature_columns].copy()
        
        # Handle categorical encoding
        for col in ['constructorId', 'circuitId']:
            if col in X_2025.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                X_2025[col] = X_2025[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )
        
        # Scale and predict
        X_2025_scaled = self.scaler.transform(X_2025)
        predictions = self.model.predict(X_2025_scaled).flatten()
        
        # Create results
        driver_pred_pairs = list(zip(latest_data['driverId'], predictions))
        driver_pred_pairs.sort(key=lambda x: x[1])
        
        results_2025 = []
        for position, (driver_id, pred_val) in enumerate(driver_pred_pairs, 1):
            results_2025.append({
                'driverId': driver_id,
                'predicted_position': position,
                'prediction_confidence': pred_val
            })
        
        results_df = pd.DataFrame(results_2025)
        
        # Add driver and team info
        results_df = results_df.merge(self.drivers[['driverId', 'forename', 'surname']], on='driverId')
        results_df['driver_name'] = results_df['forename'] + ' ' + results_df['surname']
        
        results_df = results_df.merge(latest_data[['driverId', 'constructorId']], on='driverId')
        results_df = results_df.merge(self.constructors[['constructorId', 'name']], on='constructorId')
        results_df.rename(columns={'name': 'team'}, inplace=True)
        
        results_df = results_df.sort_values('predicted_position')
        
        # Display results
        data_quality = "🔥 AUTHENTIC" if self.using_real_data else "📊 SAMPLE"
        print(f"\\n🏁 2025 F1 PREDICTIONS ({data_quality} DATA)")
        print("="*70)
        
        for _, row in results_df.iterrows():
            pos_icon = "🥇" if row['predicted_position'] == 1 else "🥈" if row['predicted_position'] == 2 else "🥉" if row['predicted_position'] == 3 else "🏁" if row['predicted_position'] <= 10 else "  "
            print(f"{pos_icon} P{row['predicted_position']:2d}: {row['driver_name']:<25} ({row['team']:<20})")
        
        # Save predictions with data source indicator
        filename = f'f1_2025_predictions_{self.data_source}.csv'
        results_df.to_csv(filename, index=False)
        print(f"\\n💾 Predictions saved to '{filename}'")
        
        return results_df
    
    def run_full_pipeline(self):
        """Run the complete adaptive prediction pipeline"""
        print("🏎️ ADAPTIVE F1 NEURAL NETWORK PREDICTOR")
        print("="*60)
        print("Automatically adapts to available data source (Real Kaggle vs Sample)")
        print("="*60)
        
        # Detect and load data
        if not self.detect_data_source():
            print("❌ No data source available")
            return None
        
        if not self.load_data():
            print("❌ Failed to load data")
            return None
        
        # Create features and train
        self.create_adaptive_features()
        X_test, y_test, feature_columns, history = self.train_adaptive_model()
        predictions_2025 = self.evaluate_and_predict(X_test, y_test, feature_columns)
        
        # Final summary
        print("\\n" + "🏆"*30)
        print(f"   PREDICTION COMPLETE WITH {self.data_source.upper()} DATA!")
        if self.using_real_data:
            print("   🔥 Using authentic 70+ years of F1 history!")
        else:
            print("   📊 Using sample data - get real Kaggle dataset for better accuracy!")
        print("🏆"*30)
        
        return predictions_2025

def main():
    """Main function"""
    predictor = AdaptiveF1Predictor()
    predictions = predictor.run_full_pipeline()
    return predictor, predictions

if __name__ == "__main__":
    predictor, predictions = main()