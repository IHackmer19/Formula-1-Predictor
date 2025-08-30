#!/usr/bin/env python3
"""
Demo script showing different ways to use the F1 prediction system
"""

import pandas as pd
import numpy as np

def demo_usage():
    """Demonstrate different usage scenarios"""
    
    print("🎮 F1 NEURAL NETWORK PREDICTION SYSTEM - DEMO")
    print("="*50)
    
    # Load predictions
    predictions = pd.read_csv('f1_2025_predictions.csv')
    
    print("\n1️⃣ BASIC USAGE - View Predictions")
    print("-"*40)
    top_5 = predictions.head(5)
    for _, row in top_5.iterrows():
        print(f"P{row['predicted_position']}: {row['driver_name']} ({row['team']})")
    
    print("\n2️⃣ TEAM ANALYSIS - Constructor Performance")
    print("-"*40)
    team_stats = predictions.groupby('team').agg({
        'predicted_position': ['mean', 'min', 'max']
    }).round(1)
    team_stats.columns = ['Avg_Pos', 'Best_Pos', 'Worst_Pos']
    
    print("Top 3 Teams by Average Position:")
    top_teams = team_stats.sort_values('Avg_Pos').head(3)
    for i, (team, row) in enumerate(top_teams.iterrows(), 1):
        print(f"  {i}. {team}: Avg P{row['Avg_Pos']}, Best P{int(row['Best_Pos'])}")
    
    print("\n3️⃣ DRIVER ANALYSIS - Individual Performance")
    print("-"*40)
    
    # Championship contenders (top 6)
    contenders = predictions[predictions['predicted_position'] <= 6]
    print("Championship Contenders (Top 6):")
    for _, row in contenders.iterrows():
        print(f"  • {row['driver_name']} (P{row['predicted_position']}) - {row['team']}")
    
    print("\n4️⃣ BATTLE ANALYSIS - Close Fights")
    print("-"*40)
    
    # Mercedes vs Ferrari battle
    merc_ferrari = predictions[predictions['team'].isin(['Mercedes', 'Ferrari'])]
    print("Mercedes vs Ferrari Battle:")
    for _, row in merc_ferrari.iterrows():
        print(f"  {row['team']}: {row['driver_name']} - P{row['predicted_position']}")
    
    print("\n5️⃣ STATISTICAL INSIGHTS")
    print("-"*40)
    
    print(f"• Average predicted position: {predictions['predicted_position'].mean():.1f}")
    print(f"• Position standard deviation: {predictions['predicted_position'].std():.1f}")
    print(f"• Teams with both drivers in top 10: {len([team for team in predictions.groupby('team') if all(predictions[predictions['team'] == team]['predicted_position'] <= 10)])}")
    
    # Points prediction
    points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
    predictions['predicted_points'] = predictions['predicted_position'].map(points_map).fillna(0)
    
    total_points = predictions.groupby('team')['predicted_points'].sum().sort_values(ascending=False)
    print(f"\nConstructor Championship Points (Single Race):")
    for i, (team, points) in enumerate(total_points.head(5).items(), 1):
        print(f"  {i}. {team}: {int(points)} points")

if __name__ == "__main__":
    demo_usage()