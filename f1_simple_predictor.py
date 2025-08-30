#!/usr/bin/env python3
"""
Simplified Formula 1 Driver Position Prediction Neural Network

This script creates a neural network to predict the finishing positions 
of 20 F1 drivers for the next race in 2025 based on historical data.
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

def load_f1_data():
    """Load all F1 datasets"""
    print("Loading F1 datasets...")
    
    circuits = pd.read_csv('f1_data/circuits.csv')
    drivers = pd.read_csv('f1_data/drivers.csv')
    constructors = pd.read_csv('f1_data/constructors.csv')
    races = pd.read_csv('f1_data/races.csv')
    results = pd.read_csv('f1_data/results.csv')
    qualifying = pd.read_csv('f1_data/qualifying.csv')
    
    print(f"Loaded data:")
    print(f"- {len(circuits)} circuits")
    print(f"- {len(drivers)} drivers") 
    print(f"- {len(constructors)} constructors")
    print(f"- {len(races)} races")
    print(f"- {len(results)} race results")
    print(f"- {len(qualifying)} qualifying results")
    
    return circuits, drivers, constructors, races, results, qualifying

def create_features(circuits, drivers, constructors, races, results, qualifying):
    """Create features for the neural network"""
    print("\n=== FEATURE ENGINEERING ===")
    
    # Merge datasets
    df = results.merge(races, on='raceId')
    df = df.merge(drivers, on='driverId')
    df = df.merge(constructors, on='constructorId')
    df = df.merge(circuits, on='circuitId')
    df = df.merge(qualifying, on=['raceId', 'driverId'], suffixes=('', '_qual'))
    
    # Convert date and sort
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'driverId']).reset_index(drop=True)
    
    print(f"Merged dataset shape: {df.shape}")
    
    # Create feature matrix
    features = []
    
    for idx, row in df.iterrows():
        if idx % 500 == 0:
            print(f"Processing row {idx}/{len(df)}")
            
        feature_row = {
            'driverId': row['driverId'],
            'constructorId': row['constructorId'],
            'circuitId': row['circuitId'],
            'year': row['year'],
            'round': row['round'],
            'grid_position': row['grid'],
            'qualifying_position': row['position_qual'],
            'circuit_lat': row['lat'],
            'circuit_lng': row['lng'],
            'circuit_alt': row['alt'],
            'target_position': row['position']
        }
        
        # Historical performance features
        driver_id = row['driverId']
        current_date = row['date']
        
        # Get historical data for this driver before current race
        historical_data = df[(df['driverId'] == driver_id) & (df['date'] < current_date)]
        
        if len(historical_data) > 0:
            # Last 5 races
            recent_5 = historical_data.tail(5)
            feature_row['avg_position_5'] = recent_5['position'].mean()
            feature_row['avg_points_5'] = recent_5['points'].mean()
            feature_row['avg_grid_5'] = recent_5['grid'].mean()
            
            # Last 10 races
            recent_10 = historical_data.tail(10)
            feature_row['avg_position_10'] = recent_10['position'].mean()
            feature_row['avg_points_10'] = recent_10['points'].mean()
            
            # Season performance
            current_year = row['year']
            season_data = historical_data[historical_data['year'] == current_year]
            if len(season_data) > 0:
                feature_row['season_avg_position'] = season_data['position'].mean()
                feature_row['season_points'] = season_data['points'].sum()
            else:
                feature_row['season_avg_position'] = 10.5
                feature_row['season_points'] = 0
                
            # Overall career stats
            feature_row['career_avg_position'] = historical_data['position'].mean()
            feature_row['career_total_points'] = historical_data['points'].sum()
            feature_row['career_races'] = len(historical_data)
            feature_row['career_wins'] = len(historical_data[historical_data['position'] == 1])
            feature_row['career_podiums'] = len(historical_data[historical_data['position'] <= 3])
        else:
            # First race for this driver - use defaults
            feature_row['avg_position_5'] = 10.5
            feature_row['avg_points_5'] = 0
            feature_row['avg_grid_5'] = 10.5
            feature_row['avg_position_10'] = 10.5
            feature_row['avg_points_10'] = 0
            feature_row['season_avg_position'] = 10.5
            feature_row['season_points'] = 0
            feature_row['career_avg_position'] = 10.5
            feature_row['career_total_points'] = 0
            feature_row['career_races'] = 0
            feature_row['career_wins'] = 0
            feature_row['career_podiums'] = 0
        
        # Constructor performance
        constructor_id = row['constructorId']
        constructor_historical = df[(df['constructorId'] == constructor_id) & (df['date'] < current_date)]
        
        if len(constructor_historical) > 0:
            constructor_recent = constructor_historical.tail(20)  # Last 20 results for constructor
            feature_row['constructor_avg_position'] = constructor_recent['position'].mean()
            feature_row['constructor_avg_points'] = constructor_recent['points'].mean()
        else:
            feature_row['constructor_avg_position'] = 10.5
            feature_row['constructor_avg_points'] = 0
        
        # Circuit-specific performance
        circuit_id = row['circuitId']
        circuit_driver_historical = df[
            (df['driverId'] == driver_id) & 
            (df['circuitId'] == circuit_id) & 
            (df['date'] < current_date)
        ]
        
        if len(circuit_driver_historical) > 0:
            feature_row['circuit_driver_avg_position'] = circuit_driver_historical['position'].mean()
        else:
            feature_row['circuit_driver_avg_position'] = 10.5
        
        features.append(feature_row)
    
    features_df = pd.DataFrame(features)
    
    # Remove rows with missing target
    features_df = features_df.dropna(subset=['target_position'])
    
    print(f"Final feature dataset shape: {features_df.shape}")
    return features_df

def build_and_train_model(features_df):
    """Build and train the neural network"""
    print("\n=== MODEL BUILDING AND TRAINING ===")
    
    # Prepare features and target
    feature_columns = [col for col in features_df.columns if col not in ['target_position', 'driverId']]
    X = features_df[feature_columns].copy()
    y = features_df['target_position'].copy()
    
    print(f"Feature columns: {feature_columns}")
    print(f"Features shape: {X.shape}")
    
    # Handle categorical variables
    label_encoders = {}
    categorical_columns = ['constructorId', 'circuitId']
    for col in categorical_columns:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Build model
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dropout(0.3),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(32, activation='relu'),
        
        # Output layer - regression for position prediction
        layers.Dense(1, activation='linear')
    ])
    
    # Compile model for regression
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    print("Model architecture:")
    model.summary()
    
    # Train model
    print("\nTraining model...")
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return model, scaler, label_encoders, X_test, y_test, history, feature_columns

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("\n=== MODEL EVALUATION ===")
    
    # Make predictions
    y_pred = model.predict(X_test).flatten()
    
    # Round to nearest integer and clip to valid range
    y_pred_rounded = np.clip(np.round(y_pred), 1, 20)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred_rounded)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_rounded))
    
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Root Mean Square Error: {rmse:.2f}")
    
    # Position accuracy
    exact_accuracy = np.mean(y_test == y_pred_rounded)
    print(f"Exact Position Accuracy: {exact_accuracy:.3f}")
    
    # Within N positions accuracy
    for n in [1, 2, 3, 5]:
        within_n = np.mean(np.abs(y_test - y_pred_rounded) <= n)
        print(f"Within {n} position(s) accuracy: {within_n:.3f}")
    
    # Create evaluation plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Predictions vs Actual
    axes[0].scatter(y_test, y_pred_rounded, alpha=0.6, color='blue')
    axes[0].plot([1, 20], [1, 20], 'r--', lw=2)
    axes[0].set_xlabel('Actual Position')
    axes[0].set_ylabel('Predicted Position')
    axes[0].set_title('Predicted vs Actual Positions')
    axes[0].grid(True, alpha=0.3)
    
    # Prediction error distribution
    residuals = y_test - y_pred_rounded
    axes[1].hist(residuals, bins=30, alpha=0.7, color='green')
    axes[1].set_xlabel('Prediction Error (Actual - Predicted)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Prediction Error Distribution')
    axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Error by actual position
    error_df = pd.DataFrame({'actual': y_test, 'error': np.abs(residuals)})
    error_by_pos = error_df.groupby('actual')['error'].mean()
    axes[2].bar(error_by_pos.index, error_by_pos.values, color='orange')
    axes[2].set_xlabel('Actual Position')
    axes[2].set_ylabel('Mean Absolute Error')
    axes[2].set_title('Prediction Error by Position')
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return mae, rmse, exact_accuracy

def predict_2025_race(model, scaler, label_encoders, features_df, drivers, constructors, feature_columns):
    """Predict positions for 2025 race"""
    print("\n=== 2025 RACE PREDICTION ===")
    
    # Get the most recent data for each driver
    latest_data = features_df.groupby('driverId').last().reset_index()
    
    # Update for 2025 prediction
    latest_data['year'] = 2025
    latest_data['round'] = 1
    latest_data['circuitId'] = 1  # Bahrain
    
    # Prepare features
    X_2025 = latest_data[feature_columns].copy()
    
    # Handle categorical encoding
    for col in ['constructorId', 'circuitId']:
        if col in X_2025.columns and col in label_encoders:
            le = label_encoders[col]
            X_2025[col] = X_2025[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )
    
    # Scale features
    X_2025_scaled = scaler.transform(X_2025)
    
    # Make predictions
    predictions = model.predict(X_2025_scaled).flatten()
    predicted_positions = np.clip(np.round(predictions), 1, 20)
    
    # Ensure unique positions (handle ties by adjusting slightly)
    unique_positions = []
    used_positions = set()
    
    # Sort drivers by their predicted position values
    driver_pred_pairs = list(zip(latest_data['driverId'], predictions))
    driver_pred_pairs.sort(key=lambda x: x[1])
    
    # Assign positions 1-20
    for i, (driver_id, pred_val) in enumerate(driver_pred_pairs):
        unique_positions.append((driver_id, i + 1))
    
    # Create results dataframe
    results_2025 = pd.DataFrame(unique_positions, columns=['driverId', 'predicted_position'])
    
    # Add driver information
    results_2025 = results_2025.merge(drivers[['driverId', 'forename', 'surname']], on='driverId')
    results_2025['driver_name'] = results_2025['forename'] + ' ' + results_2025['surname']
    
    # Add constructor information
    results_2025 = results_2025.merge(latest_data[['driverId', 'constructorId']], on='driverId')
    results_2025 = results_2025.merge(constructors[['constructorId', 'name']], on='constructorId')
    results_2025.rename(columns={'name': 'team'}, inplace=True)
    
    # Sort by predicted position
    results_2025 = results_2025.sort_values('predicted_position')
    
    print("\n🏁 PREDICTED 2025 RACE RESULTS 🏁")
    print("=" * 70)
    for idx, row in results_2025.iterrows():
        print(f"P{row['predicted_position']:2d}: {row['driver_name']:<20} ({row['team']:<15})")
    
    # Save predictions
    results_2025.to_csv('f1_2025_predictions.csv', index=False)
    print(f"\nPredictions saved to 'f1_2025_predictions.csv'")
    
    # Visualize predictions
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Position predictions
    y_pos = range(len(results_2025))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(results_2025)))
    
    axes[0].barh(y_pos, [21 - pos for pos in results_2025['predicted_position']], color=colors)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels([f"P{pos}: {name}" for pos, name in 
                            zip(results_2025['predicted_position'], results_2025['driver_name'])],
                           fontsize=8)
    axes[0].set_xlabel('Position Score (Higher = Better)')
    axes[0].set_title('2025 Race Position Predictions')
    axes[0].invert_yaxis()
    
    # Team performance
    team_positions = results_2025.groupby('team')['predicted_position'].mean().sort_values()
    y_pos_teams = range(len(team_positions))
    axes[1].barh(y_pos_teams, team_positions.values, color='lightcoral')
    axes[1].set_yticks(y_pos_teams)
    axes[1].set_yticklabels(team_positions.index, fontsize=8)
    axes[1].set_xlabel('Average Predicted Position')
    axes[1].set_title('Team Performance Predictions 2025')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('2025_race_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results_2025

def main():
    """Main function"""
    print("🏎️  FORMULA 1 POSITION PREDICTION NEURAL NETWORK 🏎️")
    print("=" * 70)
    
    # Load data
    circuits, drivers, constructors, races, results, qualifying = load_f1_data()
    
    # Create features
    features_df = create_features(circuits, drivers, constructors, races, results, qualifying)
    
    # Build and train model
    model, scaler, label_encoders, X_test, y_test, history, feature_columns = build_and_train_model(features_df)
    
    # Evaluate model
    mae, rmse, accuracy = evaluate_model(model, X_test, y_test)
    
    # Make 2025 predictions
    predictions_2025 = predict_2025_race(
        model, scaler, label_encoders, features_df, drivers, constructors, feature_columns
    )
    
    print("\n🏆 PIPELINE COMPLETED SUCCESSFULLY! 🏆")
    print(f"Model Performance: MAE={mae:.2f}, RMSE={rmse:.2f}, Accuracy={accuracy:.3f}")
    
    return model, predictions_2025

if __name__ == "__main__":
    model, predictions = main()