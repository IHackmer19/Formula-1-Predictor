#!/usr/bin/env python3
"""
Clean up 2025 F1 predictions to ensure correct driver lineup
"""

import pandas as pd
import json

def clean_2025_predictions():
    """Clean up the 2025 predictions to remove duplicates and ensure correct lineup"""
    
    # Load the current 2025 drivers configuration
    with open('current_2025_drivers.json', 'r') as f:
        config = json.load(f)
    
    # Create a clean list of unique drivers for 2025
    unique_drivers = {}
    
    for driver in config['2025_drivers']:
        driver_id = driver['id']
        if driver_id not in unique_drivers:
            unique_drivers[driver_id] = driver
        else:
            # If driver appears multiple times, keep the one with more races
            current_rounds = unique_drivers[driver_id]['rounds']
            new_rounds = driver['rounds']
            
            # Parse rounds to count them
            current_count = len(current_rounds.split(';')) if current_rounds else 0
            new_count = len(new_rounds.split(';')) if new_rounds else 0
            
            if new_count > current_count:
                unique_drivers[driver_id] = driver
    
    # Load the raw predictions
    df = pd.read_csv('f1db_focused_2025_predictions.csv')
    
    # Clean up predictions
    cleaned_predictions = []
    seen_drivers = set()
    
    for _, row in df.iterrows():
        driver_name = row['driver']
        
        # Find the driver in our unique drivers list
        found_driver = None
        for driver_id, driver_info in unique_drivers.items():
            if driver_info['name'] == driver_name:
                found_driver = driver_info
                break
        
        if found_driver and driver_name not in seen_drivers:
            seen_drivers.add(driver_name)
            
            # Use the constructor from our configuration
            constructor_name = found_driver['constructor_name']
            
            cleaned_predictions.append({
                'driver': driver_name,
                'constructor': constructor_name,
                'predicted_position': int(row['predicted_position']),
                'confidence': float(row['confidence'])
            })
    
    # Sort by predicted position
    cleaned_predictions.sort(key=lambda x: x['predicted_position'])
    
    # Reassign positions to be sequential
    for i, pred in enumerate(cleaned_predictions, 1):
        pred['position'] = i
    
    # Save cleaned predictions
    cleaned_df = pd.DataFrame(cleaned_predictions)
    cleaned_df.to_csv('f1db_focused_2025_predictions_cleaned.csv', index=False)
    
    # Create JSON for web app
    web_predictions = []
    for pred in cleaned_predictions:
        web_predictions.append({
            'position': pred['position'],
            'driver': pred['driver'],
            'team': pred['constructor']
        })
    
    with open('docs/data/f1db_focused_predictions.json', 'w') as f:
        json.dump(web_predictions, f, indent=2)
    
    print("🏁 CLEANED 2025 F1 PREDICTIONS")
    print("="*50)
    for pred in cleaned_predictions:
        print(f"🏁 P {pred['position']:2d}: {pred['driver']:<20} ({pred['constructor']})")
    
    print(f"\n✅ Cleaned predictions saved:")
    print(f"   - f1db_focused_2025_predictions_cleaned.csv")
    print(f"   - docs/data/f1db_focused_predictions.json")
    print(f"   - Total drivers: {len(cleaned_predictions)}")

if __name__ == "__main__":
    clean_2025_predictions()