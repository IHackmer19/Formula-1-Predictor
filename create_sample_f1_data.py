#!/usr/bin/env python3
"""
Create sample F1 data that mimics the structure of the Kaggle Formula 1 dataset
This will allow us to build and test our neural network without requiring Kaggle API credentials
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

def create_sample_f1_data():
    """Create sample F1 data with realistic structure and relationships"""
    
    # Create data directory
    os.makedirs('f1_data', exist_ok=True)
    
    # Define realistic F1 teams and drivers
    teams = [
        'Mercedes', 'Red Bull Racing', 'Ferrari', 'McLaren', 'Alpine',
        'Aston Martin', 'AlphaTauri', 'Alfa Romeo', 'Haas', 'Williams'
    ]
    
    drivers_2024 = [
        'Lewis Hamilton', 'George Russell', 'Max Verstappen', 'Sergio Perez',
        'Charles Leclerc', 'Carlos Sainz', 'Lando Norris', 'Oscar Piastri',
        'Fernando Alonso', 'Lance Stroll', 'Yuki Tsunoda', 'Daniel Ricciardo',
        'Valtteri Bottas', 'Zhou Guanyu', 'Kevin Magnussen', 'Nico Hulkenberg',
        'Alexander Albon', 'Logan Sargeant', 'Esteban Ocon', 'Pierre Gasly'
    ]
    
    circuits = [
        'Bahrain International Circuit', 'Jeddah Corniche Circuit', 'Albert Park',
        'Baku City Circuit', 'Miami International Autodrome', 'Imola',
        'Monaco', 'Circuit de Barcelona-Catalunya', 'Circuit Gilles Villeneuve',
        'Red Bull Ring', 'Silverstone', 'Hungaroring', 'Spa-Francorchamps',
        'Zandvoort', 'Monza', 'Marina Bay Street Circuit', 'Suzuka',
        'Losail International Circuit', 'Circuit of the Americas', 'Mexico City',
        'Interlagos', 'Las Vegas Strip Circuit', 'Yas Marina Circuit'
    ]
    
    # Create circuits data
    circuits_data = []
    for i, circuit in enumerate(circuits):
        circuits_data.append({
            'circuitId': i + 1,
            'circuitRef': circuit.lower().replace(' ', '_').replace('-', '_'),
            'name': circuit,
            'location': f'Location_{i+1}',
            'country': f'Country_{i+1}',
            'lat': np.random.uniform(-90, 90),
            'lng': np.random.uniform(-180, 180),
            'alt': np.random.uniform(0, 2000)
        })
    
    circuits_df = pd.DataFrame(circuits_data)
    circuits_df.to_csv('f1_data/circuits.csv', index=False)
    
    # Create drivers data
    drivers_data = []
    for i, driver in enumerate(drivers_2024):
        name_parts = driver.split()
        drivers_data.append({
            'driverId': i + 1,
            'driverRef': driver.lower().replace(' ', '_'),
            'number': i + 1,
            'code': ''.join([part[:3].upper() for part in name_parts]),
            'forename': name_parts[0],
            'surname': name_parts[-1],
            'dob': f'199{np.random.randint(0, 9)}-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}',
            'nationality': f'Nationality_{i+1}'
        })
    
    drivers_df = pd.DataFrame(drivers_data)
    drivers_df.to_csv('f1_data/drivers.csv', index=False)
    
    # Create constructors data
    constructors_data = []
    for i, team in enumerate(teams):
        constructors_data.append({
            'constructorId': i + 1,
            'constructorRef': team.lower().replace(' ', '_'),
            'name': team,
            'nationality': f'Nationality_{i+1}'
        })
    
    constructors_df = pd.DataFrame(constructors_data)
    constructors_df.to_csv('f1_data/constructors.csv', index=False)
    
    # Create races data (2020-2024)
    races_data = []
    race_id = 1
    
    for year in range(2020, 2025):
        for round_num, circuit in enumerate(circuits[:23], 1):  # 23 races per season
            race_date = datetime(year, 3, 1) + timedelta(days=round_num * 14)
            races_data.append({
                'raceId': race_id,
                'year': year,
                'round': round_num,
                'circuitId': circuits.index(circuit) + 1,
                'name': f'{circuit} Grand Prix',
                'date': race_date.strftime('%Y-%m-%d'),
                'time': '14:00:00'
            })
            race_id += 1
    
    races_df = pd.DataFrame(races_data)
    races_df.to_csv('f1_data/races.csv', index=False)
    
    # Create results data with realistic correlations
    results_data = []
    result_id = 1
    
    # Driver skill levels (affects performance)
    driver_skills = {}
    for i, driver in enumerate(drivers_2024):
        # Top drivers get better base performance
        if i < 6:  # Top tier
            driver_skills[i + 1] = np.random.normal(0.8, 0.1)
        elif i < 12:  # Mid tier
            driver_skills[i + 1] = np.random.normal(0.5, 0.15)
        else:  # Lower tier
            driver_skills[i + 1] = np.random.normal(0.3, 0.1)
    
    # Team performance levels
    team_performance = {}
    for i, team in enumerate(teams):
        if i < 3:  # Top teams
            team_performance[i + 1] = np.random.normal(0.9, 0.05)
        elif i < 6:  # Mid teams
            team_performance[i + 1] = np.random.normal(0.6, 0.1)
        else:  # Lower teams
            team_performance[i + 1] = np.random.normal(0.4, 0.1)
    
    for race_id in range(1, len(races_data) + 1):
        race_year = races_df[races_df['raceId'] == race_id]['year'].iloc[0]
        
        # Assign drivers to teams (2 drivers per team)
        driver_team_assignments = {}
        for team_idx, team_id in enumerate(range(1, 11)):
            driver_team_assignments[team_idx * 2 + 1] = team_id
            driver_team_assignments[team_idx * 2 + 2] = team_id
        
        # Generate race results
        race_results = []
        for driver_id in range(1, 21):  # 20 drivers
            constructor_id = driver_team_assignments[driver_id]
            
            # Calculate performance score based on driver skill and team performance
            base_performance = (driver_skills[driver_id] + team_performance[constructor_id]) / 2
            race_performance = base_performance + np.random.normal(0, 0.2)  # Add race variability
            
            # Add some circuit-specific effects
            circuit_id = races_df[races_df['raceId'] == race_id]['circuitId'].iloc[0]
            circuit_effect = np.random.normal(0, 0.1)
            final_performance = race_performance + circuit_effect
            
            race_results.append((driver_id, final_performance))
        
        # Sort by performance and assign positions
        race_results.sort(key=lambda x: x[1], reverse=True)
        
        for position, (driver_id, performance) in enumerate(race_results, 1):
            constructor_id = driver_team_assignments[driver_id]
            
            # Calculate realistic lap times and points
            fastest_lap_time = 80 + np.random.normal(0, 5)  # Base lap time around 80 seconds
            total_time = fastest_lap_time * (50 + np.random.randint(0, 20))  # 50-70 laps
            
            # Points system (25, 18, 15, 12, 10, 8, 6, 4, 2, 1 for top 10)
            points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
            points = points_map.get(position, 0)
            
            results_data.append({
                'resultId': result_id,
                'raceId': race_id,
                'driverId': driver_id,
                'constructorId': constructor_id,
                'number': driver_id,
                'grid': np.random.randint(1, 21),  # Starting grid position
                'position': position,
                'positionText': str(position),
                'positionOrder': position,
                'points': points,
                'laps': 50 + np.random.randint(0, 20),
                'time': f"{int(total_time//60)}:{int(total_time%60):02d}.{np.random.randint(0, 999):03d}",
                'milliseconds': int(total_time * 1000),
                'fastestLap': np.random.randint(10, 60),
                'rank': np.random.randint(1, 21),
                'fastestLapTime': f"1:{int(fastest_lap_time%60):02d}.{np.random.randint(0, 999):03d}",
                'fastestLapSpeed': 200 + np.random.normal(0, 20),
                'statusId': 1  # Finished
            })
            result_id += 1
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv('f1_data/results.csv', index=False)
    
    # Create qualifying data
    qualifying_data = []
    qualifying_id = 1
    
    for race_id in range(1, len(races_data) + 1):
        # Generate qualifying times that correlate with race performance
        qual_performances = []
        for driver_id in range(1, 21):
            constructor_id = driver_team_assignments[driver_id]
            base_performance = (driver_skills[driver_id] + team_performance[constructor_id]) / 2
            qual_performance = base_performance + np.random.normal(0, 0.15)
            qual_performances.append((driver_id, qual_performance))
        
        # Sort by qualifying performance
        qual_performances.sort(key=lambda x: x[1], reverse=True)
        
        for position, (driver_id, performance) in enumerate(qual_performances, 1):
            constructor_id = driver_team_assignments[driver_id]
            
            # Generate realistic qualifying times
            base_time = 75 + np.random.normal(0, 2)  # Base qualifying time
            q1_time = base_time + np.random.normal(0, 0.5)
            q2_time = base_time - 0.5 + np.random.normal(0, 0.3) if position <= 15 else None
            q3_time = base_time - 1.0 + np.random.normal(0, 0.2) if position <= 10 else None
            
            qualifying_data.append({
                'qualifyId': qualifying_id,
                'raceId': race_id,
                'driverId': driver_id,
                'constructorId': constructor_id,
                'number': driver_id,
                'position': position,
                'q1': f"1:{int(q1_time%60):02d}.{np.random.randint(0, 999):03d}",
                'q2': f"1:{int(q2_time%60):02d}.{np.random.randint(0, 999):03d}" if q2_time else None,
                'q3': f"1:{int(q3_time%60):02d}.{np.random.randint(0, 999):03d}" if q3_time else None
            })
            qualifying_id += 1
    
    qualifying_df = pd.DataFrame(qualifying_data)
    qualifying_df.to_csv('f1_data/qualifying.csv', index=False)
    
    print("Sample F1 dataset created successfully!")
    print(f"Created {len(races_data)} races from 2020-2024")
    print(f"Created {len(results_data)} race results")
    print(f"Created {len(qualifying_data)} qualifying results")
    print(f"Data saved in 'f1_data' directory")

if __name__ == "__main__":
    create_sample_f1_data()