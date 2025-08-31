#!/usr/bin/env python3
"""
Advanced Circuit-Specific F1 Predictor
Analyzes driver and constructor performance at specific circuits with comprehensive session data
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class CircuitSpecificPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.next_race = {
            'year': 2025,
            'round': 15,
            'circuit': 'zandvoort',
            'circuit_name': 'Zandvoort',
            'grand_prix': 'netherlands',
            'grand_prix_name': 'Dutch Grand Prix'
        }
        self.last_race = {
            'year': 2025,
            'round': 14,
            'circuit': 'hungaroring',
            'circuit_name': 'Hungaroring',
            'grand_prix': 'hungary',
            'grand_prix_name': 'Hungarian Grand Prix'
        }
        
    def load_data(self):
        """Load all necessary F1DB data"""
        print("📊 LOADING F1DB DATA FOR CIRCUIT-SPECIFIC ANALYSIS")
        print("="*60)
        
        # Load core data
        self.races_df = pd.read_csv('f1db-races.csv')
        self.results_df = pd.read_csv('f1db-races-race-results.csv')
        self.drivers_df = pd.read_csv('f1db-drivers.csv')
        self.constructors_df = pd.read_csv('f1db-constructors.csv')
        
        # Load session data
        try:
            self.qualifying_df = pd.read_csv('f1db-races-qualifying.csv')
            print("✅ Loaded qualifying data")
        except:
            self.qualifying_df = None
            print("⚠️  Qualifying data not available")
            
        try:
            self.sprint_df = pd.read_csv('f1db-races-sprint-results.csv')
            print("✅ Loaded sprint data")
        except:
            self.sprint_df = None
            print("⚠️  Sprint data not available")
        
        print(f"🎯 Next Race: {self.next_race['grand_prix_name']} at {self.next_race['circuit_name']}")
        print(f"📅 Last Race: {self.last_race['grand_prix_name']} at {self.last_race['circuit_name']}")
        
    def get_drivers_from_last_race(self):
        """Get drivers who raced in the last race (Round 14)"""
        print(f"\n🏁 GETTING DRIVERS FROM ROUND {self.last_race['round']}")
        print("="*50)
        
        # Get race ID for last race
        last_race_id = self.races_df[
            (self.races_df['year'] == self.last_race['year']) & 
            (self.races_df['round'] == self.last_race['round'])
        ]['id'].iloc[0]
        
        # Get results from last race
        last_race_results = self.results_df[self.results_df['raceId'] == last_race_id].copy()
        
        # Merge with driver info
        last_race_results = last_race_results.merge(
            self.drivers_df[['id', 'name', 'firstName', 'lastName']], 
            left_on='driverId', right_on='id', suffixes=('', '_driver')
        )
        
        # Get unique drivers who finished the race (not DNF/DNS)
        finished_drivers = last_race_results[
            (last_race_results['reasonRetired'].isna()) &
            (last_race_results['positionNumber'].notna())
        ]
        
        drivers_list = []
        for _, row in finished_drivers.iterrows():
            drivers_list.append({
                'driver_id': row['driverId'],
                'driver_name': row['name'],
                'constructor_id': row['constructorId'],
                'last_race_position': int(row['positionNumber']),
                'last_race_grid': int(row['gridPositionNumber']) if pd.notna(row['gridPositionNumber']) else 20
            })
        
        print(f"✅ Found {len(drivers_list)} drivers from Round {self.last_race['round']}")
        for driver in drivers_list:
            print(f"   {driver['driver_name']} ({driver['last_race_position']}th)")
            
        return drivers_list
    
    def get_circuit_history(self, circuit_id, years_back=5):
        """Get historical data for a specific circuit"""
        print(f"\n🏁 ANALYZING CIRCUIT HISTORY: {circuit_id.upper()}")
        print("="*50)
        
        # Get all races at this circuit in the last N years
        circuit_races = self.races_df[
            (self.races_df['circuitId'] == circuit_id) &
            (self.races_df['year'] >= 2025 - years_back)
        ].copy()
        
        print(f"📊 Found {len(circuit_races)} races at {circuit_id} in the last {years_back} years")
        
        # Get results for these races
        circuit_results = self.results_df[
            self.results_df['raceId'].isin(circuit_races['id'])
        ].copy()
        
        # Merge with driver and constructor info
        circuit_results = circuit_results.merge(
            self.drivers_df[['id', 'name']], 
            left_on='driverId', right_on='id', suffixes=('', '_driver')
        )
        
        circuit_results = circuit_results.merge(
            self.constructors_df[['id', 'name']], 
            left_on='constructorId', right_on='id', suffixes=('', '_constructor')
        )
        
        return circuit_results, circuit_races
    
    def analyze_driver_circuit_performance(self, driver_id, circuit_id, years_back=5):
        """Analyze a driver's performance at a specific circuit"""
        circuit_results, _ = self.get_circuit_history(circuit_id, years_back)
        
        # Filter for this driver
        driver_circuit_results = circuit_results[
            circuit_results['driverId'] == driver_id
        ].copy()
        
        if len(driver_circuit_results) == 0:
            return {
                'races': 0,
                'avg_position': 15.0,
                'best_position': 20,
                'worst_position': 20,
                'avg_grid': 15.0,
                'best_grid': 20,
                'consistency': 5.0,
                'experience_years': 0
            }
        
        # Calculate metrics
        positions = driver_circuit_results['positionNumber'].dropna()
        grids = driver_circuit_results['gridPositionNumber'].dropna()
        
        metrics = {
            'races': len(positions),
            'avg_position': float(positions.mean()) if len(positions) > 0 else 15.0,
            'best_position': int(positions.min()) if len(positions) > 0 else 20,
            'worst_position': int(positions.max()) if len(positions) > 0 else 20,
            'avg_grid': float(grids.mean()) if len(grids) > 0 else 15.0,
            'best_grid': int(grids.min()) if len(grids) > 0 else 20,
            'consistency': float(positions.std()) if len(positions) > 1 else 5.0,
            'experience_years': len(driver_circuit_results['year'].unique())
        }
        
        return metrics
    
    def analyze_constructor_circuit_performance(self, constructor_id, circuit_id, years_back=5):
        """Analyze a constructor's performance at a specific circuit"""
        circuit_results, _ = self.get_circuit_history(circuit_id, years_back)
        
        # Filter for this constructor
        constructor_circuit_results = circuit_results[
            circuit_results['constructorId'] == constructor_id
        ].copy()
        
        if len(constructor_circuit_results) == 0:
            return {
                'races': 0,
                'avg_position': 15.0,
                'best_position': 20,
                'worst_position': 20,
                'avg_grid': 15.0,
                'best_grid': 20,
                'consistency': 5.0,
                'experience_years': 0
            }
        
        # Calculate metrics
        positions = constructor_circuit_results['positionNumber'].dropna()
        grids = constructor_circuit_results['gridPositionNumber'].dropna()
        
        metrics = {
            'races': len(positions),
            'avg_position': float(positions.mean()) if len(positions) > 0 else 15.0,
            'best_position': int(positions.min()) if len(positions) > 0 else 20,
            'worst_position': int(positions.max()) if len(positions) > 0 else 20,
            'avg_grid': float(grids.mean()) if len(grids) > 0 else 15.0,
            'best_grid': int(grids.min()) if len(grids) > 0 else 20,
            'consistency': float(positions.std()) if len(positions) > 1 else 5.0,
            'experience_years': len(constructor_circuit_results['year'].unique())
        }
        
        return metrics
    
    def get_season_trends(self, driver_id, constructor_id, current_year=2025):
        """Get season performance trends for driver and constructor"""
        print(f"\n📈 ANALYZING SEASON TRENDS FOR {current_year}")
        print("="*50)
        
        # Get all races in current season
        current_season_races = self.races_df[
            self.races_df['year'] == current_year
        ].copy()
        
        # Get results for current season
        current_season_results = self.results_df[
            self.results_df['raceId'].isin(current_season_races['id'])
        ].copy()
        
        # Driver season trend
        driver_season_results = current_season_results[
            current_season_results['driverId'] == driver_id
        ].copy()
        
        # Constructor season trend
        constructor_season_results = current_season_results[
            current_season_results['constructorId'] == constructor_id
        ].copy()
        
        # Calculate trends
        driver_trend = self.calculate_trend(driver_season_results['positionNumber'])
        constructor_trend = self.calculate_trend(constructor_season_results['positionNumber'])
        
        # If season is not complete, also look at previous season
        if len(current_season_races) < 10:  # Less than 10 races completed
            print("📅 Season not complete, analyzing previous season trends...")
            prev_season_trend = self.get_previous_season_trend(driver_id, constructor_id, current_year - 1)
            driver_trend = (driver_trend + prev_season_trend['driver']) / 2
            constructor_trend = (constructor_trend + prev_season_trend['constructor']) / 2
        
        return {
            'driver_trend': driver_trend,
            'constructor_trend': constructor_trend,
            'races_completed': len(current_season_races)
        }
    
    def calculate_trend(self, positions):
        """Calculate trend from position data"""
        if len(positions) < 2:
            return 0.0
        
        # Convert to numeric, handling DNFs
        numeric_positions = []
        for pos in positions:
            if pd.isna(pos):
                numeric_positions.append(20)  # DNF penalty
            else:
                numeric_positions.append(float(pos))
        
        # Calculate trend (negative = improving, positive = declining)
        x = np.arange(len(numeric_positions))
        y = np.array(numeric_positions)
        
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return -slope  # Negative because lower positions are better
        else:
            return 0.0
    
    def get_previous_season_trend(self, driver_id, constructor_id, year):
        """Get trends from previous season"""
        prev_season_races = self.races_df[self.races_df['year'] == year].copy()
        
        if len(prev_season_races) == 0:
            return {'driver': 0.0, 'constructor': 0.0}
        
        # Get second half of previous season
        mid_point = len(prev_season_races) // 2
        second_half_races = prev_season_races.iloc[mid_point:]['id'].tolist()
        
        second_half_results = self.results_df[
            self.results_df['raceId'].isin(second_half_races)
        ].copy()
        
        driver_results = second_half_results[
            second_half_results['driverId'] == driver_id
        ]['positionNumber']
        
        constructor_results = second_half_results[
            second_half_results['constructorId'] == constructor_id
        ]['positionNumber']
        
        return {
            'driver': self.calculate_trend(driver_results),
            'constructor': self.calculate_trend(constructor_results)
        }
    
    def prepare_circuit_features(self, drivers_list):
        """Prepare features for circuit-specific prediction"""
        print(f"\n🔧 PREPARING CIRCUIT-SPECIFIC FEATURES")
        print("="*50)
        
        features = []
        feature_labels = []
        
        for driver in drivers_list:
            print(f"\n📊 Analyzing {driver['driver_name']}...")
            
            # Circuit-specific performance
            driver_circuit = self.analyze_driver_circuit_performance(
                driver['driver_id'], 
                self.next_race['circuit']
            )
            
            constructor_circuit = self.analyze_constructor_circuit_performance(
                driver['constructor_id'], 
                self.next_race['circuit']
            )
            
            # Season trends
            season_trends = self.get_season_trends(
                driver['driver_id'], 
                driver['constructor_id']
            )
            
            # Create feature vector
            feature_vector = [
                # Driver circuit performance
                driver_circuit['races'],
                driver_circuit['avg_position'],
                driver_circuit['best_position'],
                driver_circuit['consistency'],
                driver_circuit['experience_years'],
                
                # Constructor circuit performance
                constructor_circuit['races'],
                constructor_circuit['avg_position'],
                constructor_circuit['best_position'],
                constructor_circuit['consistency'],
                constructor_circuit['experience_years'],
                
                # Season trends
                season_trends['driver_trend'],
                season_trends['constructor_trend'],
                season_trends['races_completed'],
                
                # Last race performance
                driver['last_race_position'],
                driver['last_race_grid'],
                
                # Circuit characteristics (Zandvoort specific)
                4.259,  # Circuit length
                14,     # Number of turns
                72,     # Number of laps
                306.587 # Race distance
            ]
            
            features.append(feature_vector)
            feature_labels.append(driver['driver_name'])
            
            print(f"   Circuit races: {driver_circuit['races']}")
            print(f"   Avg position: {driver_circuit['avg_position']:.1f}")
            print(f"   Season trend: {season_trends['driver_trend']:.2f}")
        
        return np.array(features), feature_labels
    
    def build_circuit_model(self, input_shape):
        """Build neural network for circuit-specific prediction"""
        print(f"\n🧠 BUILDING CIRCUIT-SPECIFIC NEURAL NETWORK")
        print("="*50)
        
        model = keras.Sequential([
            layers.Dense(512, activation='relu', input_shape=input_shape),
            layers.Dropout(0.4),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
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
    
    def train_circuit_model(self, X, y):
        """Train the circuit-specific model"""
        print(f"\n🎯 TRAINING CIRCUIT-SPECIFIC MODEL")
        print("="*50)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build model
        self.model = self.build_circuit_model((X_train.shape[1],))
        
        # Train model
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=150,
            batch_size=16,
            verbose=1,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=8)
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
    
    def predict_circuit_positions(self, drivers_list, feature_labels):
        """Predict positions for the next race"""
        print(f"\n🔮 PREDICTING POSITIONS FOR {self.next_race['grand_prix_name']}")
        print("="*60)
        
        # Prepare features for prediction
        X_pred, _ = self.prepare_circuit_features(drivers_list)
        X_pred_scaled = self.scaler.transform(X_pred)
        
        # Make predictions
        predictions = self.model.predict(X_pred_scaled)
        
        # Create prediction results
        prediction_results = []
        for i, driver in enumerate(drivers_list):
            predicted_position = max(1, round(predictions[i][0]))
            
            prediction_results.append({
                'driver': driver['driver_name'],
                'constructor': driver['constructor_id'],
                'predicted_position': predicted_position,
                'confidence': 1.0 / (1.0 + abs(predictions[i][0] - predicted_position)),
                'last_race_position': driver['last_race_position']
            })
        
        # Sort by predicted position
        prediction_results.sort(key=lambda x: x['predicted_position'])
        
        # Display predictions
        print(f"\n🏁 PREDICTED POSITIONS FOR {self.next_race['grand_prix_name']}")
        print("="*60)
        for i, pred in enumerate(prediction_results, 1):
            change = pred['last_race_position'] - pred['predicted_position']
            change_symbol = "↗️" if change > 0 else "↘️" if change < 0 else "➡️"
            print(f"🏁 P {i:2d}: {pred['driver']:<20} ({pred['constructor']}) {change_symbol}")
        
        return prediction_results
    
    def save_circuit_predictions(self, predictions):
        """Save circuit-specific predictions"""
        print(f"\n💾 SAVING CIRCUIT-SPECIFIC PREDICTIONS")
        print("="*50)
        
        # Save to CSV
        df = pd.DataFrame(predictions)
        filename = f"circuit_predictions_{self.next_race['circuit']}_{self.next_race['year']}_{self.next_race['round']}.csv"
        df.to_csv(filename, index=False)
        
        # Save to JSON for web app
        web_predictions = []
        for i, pred in enumerate(predictions, 1):
            web_predictions.append({
                'position': i,
                'driver': pred['driver'],
                'team': pred['constructor'],
                'confidence': float(round(pred['confidence'], 3))
            })
        
        json_filename = f"docs/data/circuit_predictions_{self.next_race['circuit']}.json"
        with open(json_filename, 'w') as f:
            json.dump(web_predictions, f, indent=2)
        
        print(f"✅ Predictions saved to:")
        print(f"   - {filename}")
        print(f"   - {json_filename}")
    
    def run_circuit_pipeline(self):
        """Run the complete circuit-specific prediction pipeline"""
        print("🚀 CIRCUIT-SPECIFIC F1 PREDICTION PIPELINE")
        print("="*70)
        print(f"🎯 Target: {self.next_race['grand_prix_name']} at {self.next_race['circuit_name']}")
        print(f"📅 Date: Round {self.next_race['round']}, {self.next_race['year']}")
        print(f"🏁 Based on: Drivers from Round {self.last_race['round']}")
        print("="*70)
        
        # Load data
        self.load_data()
        
        # Get drivers from last race
        drivers_list = self.get_drivers_from_last_race()
        
        # Prepare features
        X, feature_labels = self.prepare_circuit_features(drivers_list)
        
        # Create target values (using last race positions as baseline)
        y = np.array([driver['last_race_position'] for driver in drivers_list])
        
        print(f"\n📊 Training data: {len(X)} drivers")
        
        # Train model
        history, X_test, y_test, y_pred = self.train_circuit_model(X, y)
        
        # Predict positions
        predictions = self.predict_circuit_positions(drivers_list, feature_labels)
        
        # Save predictions
        self.save_circuit_predictions(predictions)
        
        print(f"\n🎉 CIRCUIT-SPECIFIC PIPELINE COMPLETE!")
        print(f"✅ Predicted {len(predictions)} drivers for {self.next_race['grand_prix_name']}")
        print(f"✅ Circuit: {self.next_race['circuit_name']}")
        print(f"✅ Round: {self.next_race['round']}")

if __name__ == "__main__":
    predictor = CircuitSpecificPredictor()
    predictor.run_circuit_pipeline()