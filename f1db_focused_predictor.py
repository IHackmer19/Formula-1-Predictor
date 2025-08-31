#!/usr/bin/env python3
"""
Focused F1DB Neural Network Predictor for 2025 Season
Only uses current drivers and recent constructor data (past 5 years)
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class FocusedF1DBPredictor:
    def __init__(self):
        self.current_drivers = None
        self.current_constructors = None
        self.model = None
        self.scaler = StandardScaler()
        self.driver_encoder = LabelEncoder()
        self.constructor_encoder = LabelEncoder()
        self.circuit_encoder = LabelEncoder()
        
    def load_current_drivers(self):
        """Load current 2025 drivers configuration"""
        with open('current_2025_drivers.json', 'r') as f:
            config = json.load(f)
        self.current_drivers = config['2025_drivers']
        self.current_constructors = config['current_constructors']
        print(f"✅ Loaded {len(self.current_drivers)} current 2025 drivers")
        print(f"✅ Loaded {len(self.current_constructors)} current constructors")
        
    def load_focused_data(self):
        """Load only data for current drivers and recent constructors"""
        print("📊 LOADING FOCUSED F1DB DATA")
        print("="*50)
        
        # Load current drivers
        self.load_current_drivers()
        
        # Get current driver IDs
        current_driver_ids = [driver['id'] for driver in self.current_drivers]
        
        # Load race results for current drivers only
        print("📈 Loading race results for current drivers...")
        races_df = pd.read_csv('f1db-races.csv')
        results_df = pd.read_csv('f1db-races-race-results.csv')
        
        # Filter for current drivers only
        current_results = results_df[results_df['driverId'].isin(current_driver_ids)].copy()
        
        # Merge with races data
        current_results = current_results.merge(races_df[['id', 'year', 'round', 'circuitId']], 
                                              left_on='raceId', right_on='id', suffixes=('', '_race'))
        
        # Filter for recent years (2020-2024) for constructor data
        recent_results = current_results[current_results['year'] >= 2020].copy()
        
        print(f"📊 Found {len(recent_results)} race results for current drivers (2020-2024)")
        
        # Load constructor data for recent years
        print("🏎️ Loading recent constructor data...")
        constructors_df = pd.read_csv('f1db-constructors.csv')
        current_constructors_data = constructors_df[constructors_df['id'].isin(self.current_constructors)].copy()
        
        print(f"🏎️ Found data for {len(current_constructors_data)} current constructors")
        
        return recent_results, current_constructors_data
    
    def prepare_features(self, results_df):
        """Prepare features for the neural network"""
        print("🔧 PREPARING FEATURES")
        print("="*30)
        
        # Create features
        features = []
        valid_indices = []
        
        for idx, row in results_df.iterrows():
            # Skip rows with NaN position numbers
            if pd.isna(row['positionNumber']):
                continue
                
            # Driver features
            driver_id = row['driverId']
            constructor_id = row['constructorId']
            circuit_id = row['circuitId']
            year = row['year']
            round_num = row['round']
            
            # Get driver's historical performance
            driver_history = results_df[
                (results_df['driverId'] == driver_id) & 
                (results_df['year'] < year) &
                (results_df['positionNumber'].notna())
            ]
            
            # Get constructor's recent performance (last 5 years)
            constructor_history = results_df[
                (results_df['constructorId'] == constructor_id) & 
                (results_df['year'] >= year - 5) &
                (results_df['year'] < year) &
                (results_df['positionNumber'].notna())
            ]
            
            # Calculate features with NaN handling
            driver_avg_position = driver_history['positionNumber'].mean() if len(driver_history) > 0 else 10.0
            driver_std_position = driver_history['positionNumber'].std() if len(driver_history) > 0 else 5.0
            driver_races = len(driver_history)
            
            constructor_avg_position = constructor_history['positionNumber'].mean() if len(constructor_history) > 0 else 10.0
            constructor_std_position = constructor_history['positionNumber'].std() if len(constructor_history) > 0 else 5.0
            constructor_races = len(constructor_history)
            
            # Circuit experience
            circuit_history = results_df[
                (results_df['circuitId'] == circuit_id) & 
                (results_df['year'] < year) &
                (results_df['positionNumber'].notna())
            ]
            circuit_avg_position = circuit_history['positionNumber'].mean() if len(circuit_history) > 0 else 10.0
            
            # Season progress
            season_progress = round_num / 20.0  # Assuming 20 races per season
            
            # Create feature vector
            feature_vector = [
                float(driver_avg_position),
                float(driver_std_position),
                float(driver_races),
                float(constructor_avg_position),
                float(constructor_std_position),
                float(constructor_races),
                float(circuit_avg_position),
                float(season_progress),
                float(year - 2020)  # Years since 2020
            ]
            
            # Check for any NaN values in feature vector
            if not any(pd.isna(feature_vector)):
                features.append(feature_vector)
                valid_indices.append(idx)
        
        print(f"✅ Prepared {len(features)} valid feature vectors")
        return np.array(features), valid_indices
    
    def build_model(self, input_shape):
        """Build the neural network model"""
        print("🧠 BUILDING NEURAL NETWORK MODEL")
        print("="*40)
        
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=input_shape),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print(model.summary())
        return model
    
    def train_model(self, X, y):
        """Train the neural network model"""
        print("🎯 TRAINING FOCUSED F1DB MODEL")
        print("="*40)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build model
        self.model = self.build_model((X_train.shape[1],))
        
        # Train model
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=100,
            batch_size=32,
            verbose=1,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
            ]
        )
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        print(f"\n📊 MODEL PERFORMANCE:")
        print(f"   Mean Absolute Error: {mae:.2f} positions")
        print(f"   Root Mean Square Error: {np.sqrt(mse):.2f} positions")
        
        return history, X_test_scaled, y_test, y_pred
    
    def predict_2025_positions(self):
        """Predict positions for current 2025 drivers"""
        print("🔮 PREDICTING 2025 F1 POSITIONS")
        print("="*50)
        
        # Create prediction data for current drivers
        predictions = []
        
        for driver in self.current_drivers:
            driver_id = driver['id']
            constructor_id = driver['constructor']
            
            # Get driver's recent performance (2020-2024)
            driver_recent = self.results_df[
                (self.results_df['driverId'] == driver_id) & 
                (self.results_df['year'] >= 2020)
            ]
            
            # Get constructor's recent performance
            constructor_recent = self.results_df[
                (self.results_df['constructorId'] == constructor_id) & 
                (self.results_df['year'] >= 2020)
            ]
            
            # Calculate features for prediction
            driver_avg_position = driver_recent['positionNumber'].mean() if len(driver_recent) > 0 else 10
            driver_std_position = driver_recent['positionNumber'].std() if len(driver_recent) > 0 else 5
            driver_races = len(driver_recent)
            
            constructor_avg_position = constructor_recent['positionNumber'].mean() if len(constructor_recent) > 0 else 10
            constructor_std_position = constructor_recent['positionNumber'].std() if len(constructor_recent) > 0 else 5
            constructor_races = len(constructor_recent)
            
            # Create feature vector for 2025 prediction
            feature_vector = [
                driver_avg_position,
                driver_std_position,
                driver_races,
                constructor_avg_position,
                constructor_std_position,
                constructor_races,
                10.0,  # Average circuit position (placeholder)
                0.5,   # Mid-season
                5      # Years since 2020
            ]
            
            # Scale features
            feature_scaled = self.scaler.transform([feature_vector])
            
            # Predict position
            predicted_position = self.model.predict(feature_scaled)[0][0]
            
            predictions.append({
                'driver': driver['name'],
                'constructor': driver['constructor_name'],
                'predicted_position': max(1, round(predicted_position)),
                'confidence': 1.0 / (1.0 + abs(predicted_position - round(predicted_position)))
            })
        
        # Sort by predicted position
        predictions.sort(key=lambda x: x['predicted_position'])
        
        return predictions
    
    def save_predictions(self, predictions):
        """Save predictions to CSV and JSON"""
        print("💾 SAVING PREDICTIONS")
        print("="*30)
        
        # Save to CSV
        df = pd.DataFrame(predictions)
        df.to_csv('f1db_focused_2025_predictions.csv', index=False)
        
        # Save to JSON for web app
        web_predictions = []
        for i, pred in enumerate(predictions, 1):
            web_predictions.append({
                'position': i,
                'driver': pred['driver'],
                'team': pred['constructor']
            })
        
        with open('docs/data/f1db_focused_predictions.json', 'w') as f:
            json.dump(web_predictions, f, indent=2)
        
        print("✅ Predictions saved to:")
        print("   - f1db_focused_2025_predictions.csv")
        print("   - docs/data/f1db_focused_predictions.json")
    
    def run_focused_pipeline(self):
        """Run the complete focused prediction pipeline"""
        print("🚀 FOCUSED F1DB PREDICTION PIPELINE")
        print("="*60)
        print("🎯 Focus: Current 2025 drivers only")
        print("📅 Data: Past 5 years (2020-2024)")
        print("🏎️ Constructors: Current teams only")
        print("="*60)
        
        # Load focused data
        self.results_df, constructors_df = self.load_focused_data()
        
        # Prepare features
        X, valid_indices = self.prepare_features(self.results_df)
        y = self.results_df.loc[valid_indices, 'positionNumber'].values
        
        print(f"📊 Training data: {len(X)} samples")
        
        # Train model
        history, X_test, y_test, y_pred = self.train_model(X, y)
        
        # Predict 2025 positions
        predictions = self.predict_2025_positions()
        
        # Display predictions
        print("\n🏁 2025 F1 PREDICTIONS (Focused Model)")
        print("="*50)
        for i, pred in enumerate(predictions, 1):
            print(f"🏁 P {i:2d}: {pred['driver']:<20} ({pred['constructor']})")
        
        # Save predictions
        self.save_predictions(predictions)
        
        print("\n🎉 FOCUSED F1DB PIPELINE COMPLETE!")
        print("✅ Using only current 2025 drivers")
        print("✅ Using only recent constructor data (2020-2024)")
        print("✅ Predictions optimized for current season")

if __name__ == "__main__":
    predictor = FocusedF1DBPredictor()
    predictor.run_focused_pipeline()