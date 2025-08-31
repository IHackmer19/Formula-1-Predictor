#!/usr/bin/env python3
"""
Enhanced Circuit-Specific F1 Predictor
Allows user to choose any race and fetches live race weekend data from Formula1.com
"""

import pandas as pd
import numpy as np
import json
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta
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

class EnhancedCircuitPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.session_data = {}
        self.current_race_weekend = False
        
    def load_data(self):
        """Load all necessary F1DB data"""
        print("📊 LOADING F1DB DATA FOR ENHANCED CIRCUIT ANALYSIS")
        print("="*60)
        
        # Load core data
        self.races_df = pd.read_csv('f1db-races.csv')
        self.results_df = pd.read_csv('f1db-races-race-results.csv')
        self.drivers_df = pd.read_csv('f1db-drivers.csv')
        self.constructors_df = pd.read_csv('f1db-constructors.csv')
        
        print("✅ Loaded core F1DB data")
        
    def display_available_races(self, year=2025):
        """Display available races for the user to choose from"""
        print(f"\n🏁 AVAILABLE RACES FOR {year}")
        print("="*50)
        
        year_races = self.races_df[self.races_df['year'] == year].copy()
        
        for _, race in year_races.iterrows():
            status = "🟢 COMPLETED" if race['round'] < 15 else "🟡 UPCOMING" if race['round'] == 15 else "🔴 FUTURE"
            print(f"Round {race['round']:2d}: {race['officialName']:<40} {status}")
            print(f"         Circuit: {race['circuitId']:<20} Date: {race['date']}")
            print()
            
        return year_races
    
    def get_race_info(self, round_number, year=2025):
        """Get information about a specific race"""
        race_info = self.races_df[
            (self.races_df['year'] == year) & 
            (self.races_df['round'] == round_number)
        ]
        
        if len(race_info) == 0:
            return None
            
        race = race_info.iloc[0]
        
        return {
            'year': race['year'],
            'round': race['round'],
            'circuit': race['circuitId'],
            'circuit_name': race['circuitId'].replace('-', ' ').title(),
            'grand_prix': race['grandPrixId'],
            'grand_prix_name': race['officialName'],
            'date': race['date'],
            'race_id': race['id'],
            'f1_race_number': race['id'] + 127  # Formula1.com race number
        }
    
    def check_if_race_weekend(self, race_info):
        """Check if we're currently in the race weekend timeframe"""
        race_date = datetime.strptime(race_info['date'], '%Y-%m-%d')
        today = datetime.now()
        
        # Check if we're within 3 days before or 1 day after the race
        weekend_start = race_date - timedelta(days=3)
        weekend_end = race_date + timedelta(days=1)
        
        return weekend_start <= today <= weekend_end
    
    def fetch_f1_session_data(self, race_info, session_type):
        """Fetch live session data from Formula1.com"""
        session_urls = {
            'fp1': f"https://www.formula1.com/en/results/{race_info['year']}/races/{race_info['f1_race_number']}/{race_info['grand_prix']}/practice/1",
            'fp2': f"https://www.formula1.com/en/results/{race_info['year']}/races/{race_info['f1_race_number']}/{race_info['grand_prix']}/practice/2", 
            'fp3': f"https://www.formula1.com/en/results/{race_info['year']}/races/{race_info['f1_race_number']}/{race_info['grand_prix']}/practice/3",
            'qualifying': f"https://www.formula1.com/en/results/{race_info['year']}/races/{race_info['f1_race_number']}/{race_info['grand_prix']}/qualifying"
        }
        
        url = session_urls.get(session_type)
        if not url:
            return None
            
        try:
            print(f"🌐 Fetching {session_type.upper()} data from: {url}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Parse the session results
            results = []
            result_table = soup.find('table', class_='resultsarchive-table')
            
            if result_table:
                rows = result_table.find_all('tr')[1:]  # Skip header
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 4:
                        position = cols[0].get_text(strip=True)
                        driver_name = cols[1].get_text(strip=True)
                        team = cols[2].get_text(strip=True)
                        time = cols[3].get_text(strip=True)
                        
                        if position.isdigit():
                            results.append({
                                'position': int(position),
                                'driver': driver_name,
                                'team': team,
                                'time': time
                            })
            
            print(f"✅ Fetched {len(results)} results for {session_type.upper()}")
            return results
            
        except Exception as e:
            print(f"⚠️  Could not fetch {session_type.upper()} data: {e}")
            return None
    
    def fetch_current_weekend_data(self, race_info):
        """Fetch all available session data for current race weekend"""
        print(f"\n🌐 FETCHING CURRENT RACE WEEKEND DATA")
        print("="*50)
        print(f"🏁 Race: {race_info['grand_prix_name']}")
        print(f"📅 Date: {race_info['date']}")
        print(f"🏟️  Circuit: {race_info['circuit_name']}")
        
        sessions = ['fp1', 'fp2', 'fp3', 'qualifying']
        weekend_data = {}
        
        for session in sessions:
            data = self.fetch_f1_session_data(race_info, session)
            if data:
                weekend_data[session] = data
                time.sleep(1)  # Be respectful to F1 servers
        
        self.session_data = weekend_data
        return weekend_data
    
    def get_drivers_from_previous_race(self, race_info):
        """Get drivers who raced in the previous race"""
        print(f"\n🏁 GETTING DRIVERS FROM PREVIOUS RACE")
        print("="*50)
        
        # Find the previous race
        previous_race = self.races_df[
            (self.races_df['year'] == race_info['year']) & 
            (self.races_df['round'] == race_info['round'] - 1)
        ]
        
        if len(previous_race) == 0:
            print("⚠️  No previous race found, using current season drivers")
            return self.get_current_season_drivers(race_info['year'])
        
        prev_race_id = previous_race.iloc[0]['id']
        
        # Get results from previous race
        prev_race_results = self.results_df[self.results_df['raceId'] == prev_race_id].copy()
        
        # Merge with driver info
        prev_race_results = prev_race_results.merge(
            self.drivers_df[['id', 'name', 'firstName', 'lastName']], 
            left_on='driverId', right_on='id', suffixes=('', '_driver')
        )
        
        # Get unique drivers who finished the race (not DNF/DNS)
        finished_drivers = prev_race_results[
            (prev_race_results['reasonRetired'].isna()) &
            (prev_race_results['positionNumber'].notna())
        ]
        
        drivers_list = []
        for _, row in finished_drivers.iterrows():
            drivers_list.append({
                'driver_id': row['driverId'],
                'driver_name': row['name'],
                'constructor_id': row['constructorId'],
                'prev_race_position': int(row['positionNumber']),
                'prev_race_grid': int(row['gridPositionNumber']) if pd.notna(row['gridPositionNumber']) else 20
            })
        
        print(f"✅ Found {len(drivers_list)} drivers from previous race")
        for driver in drivers_list:
            print(f"   {driver['driver_name']} ({driver['prev_race_position']}th)")
            
        return drivers_list
    
    def get_current_season_drivers(self, year):
        """Get all drivers from current season if no previous race"""
        print(f"📊 Getting all drivers from {year} season...")
        
        season_races = self.races_df[self.races_df['year'] == year]['id'].tolist()
        season_results = self.results_df[self.results_df['raceId'].isin(season_races)].copy()
        
        # Get unique drivers
        unique_drivers = season_results[['driverId', 'constructorId']].drop_duplicates()
        
        drivers_list = []
        for _, row in unique_drivers.iterrows():
            driver_info = self.drivers_df[self.drivers_df['id'] == row['driverId']].iloc[0]
            drivers_list.append({
                'driver_id': row['driverId'],
                'driver_name': driver_info['name'],
                'constructor_id': row['constructorId'],
                'prev_race_position': 10,  # Default position
                'prev_race_grid': 10       # Default grid
            })
        
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
    
    def get_current_weekend_performance(self, driver_name):
        """Get current weekend performance for a driver"""
        if not self.session_data:
            return {
                'fp1_position': 15,
                'fp2_position': 15,
                'fp3_position': 15,
                'qualifying_position': 15,
                'weekend_trend': 0.0
            }
        
        weekend_performance = {}
        positions = []
        
        for session, data in self.session_data.items():
            if data:
                # Find driver in session data
                driver_result = None
                for result in data:
                    if driver_name.lower() in result['driver'].lower():
                        driver_result = result
                        break
                
                position = driver_result['position'] if driver_result else 15
                weekend_performance[f'{session}_position'] = position
                positions.append(position)
            else:
                weekend_performance[f'{session}_position'] = 15
                positions.append(15)
        
        # Calculate weekend trend
        if len(positions) >= 2:
            weekend_trend = self.calculate_trend(positions)
        else:
            weekend_trend = 0.0
        
        weekend_performance['weekend_trend'] = weekend_trend
        
        return weekend_performance
    
    def prepare_enhanced_features(self, drivers_list, race_info):
        """Prepare enhanced features including current weekend data"""
        print(f"\n🔧 PREPARING ENHANCED CIRCUIT-SPECIFIC FEATURES")
        print("="*50)
        
        features = []
        feature_labels = []
        
        for driver in drivers_list:
            print(f"\n📊 Analyzing {driver['driver_name']}...")
            
            # Circuit-specific performance
            driver_circuit = self.analyze_driver_circuit_performance(
                driver['driver_id'], 
                race_info['circuit']
            )
            
            constructor_circuit = self.analyze_constructor_circuit_performance(
                driver['constructor_id'], 
                race_info['circuit']
            )
            
            # Season trends
            season_trends = self.get_season_trends(
                driver['driver_id'], 
                driver['constructor_id']
            )
            
            # Current weekend performance
            weekend_performance = self.get_current_weekend_performance(driver['driver_name'])
            
            # Create enhanced feature vector
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
                
                # Previous race performance
                driver['prev_race_position'],
                driver['prev_race_grid'],
                
                # Current weekend performance
                weekend_performance['fp1_position'],
                weekend_performance['fp2_position'],
                weekend_performance['fp3_position'],
                weekend_performance['qualifying_position'],
                weekend_performance['weekend_trend'],
                
                # Circuit characteristics (dynamic based on race)
                self.get_circuit_length(race_info['circuit']),
                self.get_circuit_turns(race_info['circuit']),
                self.get_circuit_laps(race_info),
                self.get_circuit_distance(race_info)
            ]
            
            features.append(feature_vector)
            feature_labels.append(driver['driver_name'])
            
            print(f"   Circuit races: {driver_circuit['races']}")
            print(f"   Avg position: {driver_circuit['avg_position']:.1f}")
            print(f"   Season trend: {season_trends['driver_trend']:.2f}")
            if self.current_race_weekend:
                print(f"   Qualifying: {weekend_performance['qualifying_position']}th")
        
        return np.array(features), feature_labels
    
    def get_circuit_length(self, circuit_id):
        """Get circuit length from race data"""
        circuit_race = self.races_df[self.races_df['circuitId'] == circuit_id].iloc[0]
        return float(circuit_race['courseLength']) if pd.notna(circuit_race['courseLength']) else 5.0
    
    def get_circuit_turns(self, circuit_id):
        """Get number of turns from race data"""
        circuit_race = self.races_df[self.races_df['circuitId'] == circuit_id].iloc[0]
        return int(circuit_race['turns']) if pd.notna(circuit_race['turns']) else 15
    
    def get_circuit_laps(self, race_info):
        """Get number of laps from race data"""
        race = self.races_df[
            (self.races_df['year'] == race_info['year']) & 
            (self.races_df['round'] == race_info['round'])
        ].iloc[0]
        return int(race['laps']) if pd.notna(race['laps']) else 50
    
    def get_circuit_distance(self, race_info):
        """Get race distance from race data"""
        race = self.races_df[
            (self.races_df['year'] == race_info['year']) & 
            (self.races_df['round'] == race_info['round'])
        ].iloc[0]
        return float(race['distance']) if pd.notna(race['distance']) else 300.0
    
    def build_enhanced_model(self, input_shape):
        """Build enhanced neural network for circuit-specific prediction"""
        print(f"\n🧠 BUILDING ENHANCED CIRCUIT-SPECIFIC NEURAL NETWORK")
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
    
    def train_enhanced_model(self, X, y):
        """Train the enhanced circuit-specific model"""
        print(f"\n🎯 TRAINING ENHANCED CIRCUIT-SPECIFIC MODEL")
        print("="*50)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build model
        self.model = self.build_enhanced_model((X_train.shape[1],))
        
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
    
    def predict_enhanced_positions(self, drivers_list, race_info, feature_labels):
        """Predict positions with enhanced features"""
        print(f"\n🔮 PREDICTING POSITIONS FOR {race_info['grand_prix_name']}")
        print("="*60)
        
        # Prepare features for prediction
        X_pred, _ = self.prepare_enhanced_features(drivers_list, race_info)
        X_pred_scaled = self.scaler.transform(X_pred)
        
        # Make predictions
        predictions = self.model.predict(X_pred_scaled)
        
        # Create prediction results
        prediction_results = []
        for i, driver in enumerate(drivers_list):
            predicted_position = max(1, round(predictions[i][0]))
            
            # Get current weekend performance
            weekend_performance = self.get_current_weekend_performance(driver['driver_name'])
            
            prediction_results.append({
                'driver': driver['driver_name'],
                'constructor': driver['constructor_id'],
                'predicted_position': predicted_position,
                'confidence': 1.0 / (1.0 + abs(predictions[i][0] - predicted_position)),
                'prev_race_position': driver['prev_race_position'],
                'qualifying_position': weekend_performance.get('qualifying_position', 15),
                'weekend_trend': weekend_performance.get('weekend_trend', 0.0)
            })
        
        # Sort by predicted position
        prediction_results.sort(key=lambda x: x['predicted_position'])
        
        # Display predictions
        print(f"\n🏁 PREDICTED POSITIONS FOR {race_info['grand_prix_name']}")
        print("="*60)
        for i, pred in enumerate(prediction_results, 1):
            change = pred['prev_race_position'] - pred['predicted_position']
            change_symbol = "↗️" if change > 0 else "↘️" if change < 0 else "➡️"
            
            quali_info = ""
            if self.current_race_weekend and pred['qualifying_position'] != 15:
                quali_info = f" (Q: {pred['qualifying_position']}th)"
            
            print(f"🏁 P {i:2d}: {pred['driver']:<20} ({pred['constructor']}) {change_symbol}{quali_info}")
        
        return prediction_results
    
    def run_enhanced_pipeline(self):
        """Run the complete enhanced circuit-specific prediction pipeline"""
        print("🚀 ENHANCED CIRCUIT-SPECIFIC F1 PREDICTION PIPELINE")
        print("="*70)
        
        # Load data
        self.load_data()
        
        # Display available races
        year_races = self.display_available_races(2025)
        
        # Get user input for race selection
        print("\n🎯 SELECT A RACE FOR PREDICTIONS")
        print("="*50)
        try:
            round_number = int(input("Enter round number (1-24): "))
            if round_number < 1 or round_number > 24:
                print("❌ Invalid round number. Using Round 15 (Dutch GP) as default.")
                round_number = 15
        except ValueError:
            print("❌ Invalid input. Using Round 15 (Dutch GP) as default.")
            round_number = 15
        
        # Get race information
        race_info = self.get_race_info(round_number, 2025)
        if not race_info:
            print("❌ Race not found!")
            return
        
        print(f"\n🎯 SELECTED RACE: {race_info['grand_prix_name']}")
        print(f"📅 Date: {race_info['date']}")
        print(f"🏟️  Circuit: {race_info['circuit_name']}")
        
        # Check if we're in race weekend
        self.current_race_weekend = self.check_if_race_weekend(race_info)
        if self.current_race_weekend:
            print("🟡 RACE WEEKEND DETECTED - Fetching live session data...")
            self.fetch_current_weekend_data(race_info)
        else:
            print("🟢 Not in race weekend - using historical data only")
        
        # Get drivers from previous race
        drivers_list = self.get_drivers_from_previous_race(race_info)
        
        # Prepare features
        X, feature_labels = self.prepare_enhanced_features(drivers_list, race_info)
        
        # Create target values (using previous race positions as baseline)
        y = np.array([driver['prev_race_position'] for driver in drivers_list])
        
        print(f"\n📊 Training data: {len(X)} drivers")
        print(f"🔧 Features: {X.shape[1]} dimensions")
        
        # Train model
        history, X_test, y_test, y_pred = self.train_enhanced_model(X, y)
        
        # Predict positions
        predictions = self.predict_enhanced_positions(drivers_list, race_info, feature_labels)
        
        print(f"\n🎉 ENHANCED CIRCUIT-SPECIFIC PIPELINE COMPLETE!")
        print(f"✅ Predicted {len(predictions)} drivers for {race_info['grand_prix_name']}")
        print(f"✅ Circuit: {race_info['circuit_name']}")
        print(f"✅ Round: {race_info['round']}")
        if self.current_race_weekend:
            print(f"✅ Live weekend data: {len(self.session_data)} sessions")
        
        return predictions, race_info

if __name__ == "__main__":
    predictor = EnhancedCircuitPredictor()
    predictions, race_info = predictor.run_enhanced_pipeline()