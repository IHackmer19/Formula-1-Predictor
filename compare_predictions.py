#!/usr/bin/env python3
"""
Compare Sample vs Real Data Predictions

This script compares predictions made with sample data vs real Kaggle data
to demonstrate the improvement in accuracy and realism.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_predictions():
    """Load available prediction files"""
    predictions = {}
    
    # Check for sample data predictions
    sample_files = ['f1_2025_predictions.csv', 'f1_2025_predictions_sample.csv']
    for file in sample_files:
        if os.path.exists(file):
            predictions['sample'] = pd.read_csv(file)
            print(f"✅ Loaded sample predictions from {file}")
            break
    
    # Check for real data predictions
    real_files = ['f1_2025_predictions_kaggle.csv', 'f1_2025_real_predictions.csv']
    for file in real_files:
        if os.path.exists(file):
            predictions['real'] = pd.read_csv(file)
            print(f"✅ Loaded real data predictions from {file}")
            break
    
    return predictions

def compare_predictions(predictions):
    """Compare sample vs real data predictions"""
    print("\\n🔍 PREDICTION COMPARISON ANALYSIS")
    print("="*60)
    
    if 'sample' not in predictions:
        print("❌ Sample predictions not found")
        return
    
    if 'real' not in predictions:
        print("⚠️ Real data predictions not found")
        print("Real data predictions will be available after connecting to Kaggle dataset")
        print("\\nCurrent sample predictions:")
        display_single_prediction(predictions['sample'], "Sample Data")
        return
    
    sample_pred = predictions['sample']
    real_pred = predictions['real']
    
    print("\\n🏁 PODIUM COMPARISON")
    print("-"*40)
    
    print("📊 Sample Data Podium:")
    for i in range(3):
        row = sample_pred.iloc[i]
        print(f"   P{i+1}: {row['driver_name']} ({row['team']})")
    
    print("\\n🔥 Real Data Podium:")
    for i in range(3):
        row = real_pred.iloc[i]
        print(f"   P{i+1}: {row['driver_name']} ({row['team']})")
    
    # Compare team performance
    print("\\n🏎️ TEAM PERFORMANCE COMPARISON")
    print("-"*40)
    
    sample_teams = sample_pred.groupby('team')['predicted_position'].mean().sort_values()
    real_teams = real_pred.groupby('team')['predicted_position'].mean().sort_values()
    
    print("Sample Data Team Ranking:")
    for i, (team, avg_pos) in enumerate(sample_teams.head(5).items(), 1):
        print(f"   {i}. {team}: P{avg_pos:.1f}")
    
    print("\\nReal Data Team Ranking:")
    for i, (team, avg_pos) in enumerate(real_teams.head(5).items(), 1):
        print(f"   {i}. {team}: P{avg_pos:.1f}")
    
    # Analyze differences
    print("\\n📈 KEY DIFFERENCES")
    print("-"*30)
    
    # Winner comparison
    sample_winner = sample_pred.iloc[0]['driver_name']
    real_winner = real_pred.iloc[0]['driver_name']
    
    if sample_winner != real_winner:
        print(f"🏆 Winner changed: {sample_winner} → {real_winner}")
    else:
        print(f"🏆 Winner consistent: {sample_winner}")
    
    # Position changes for top drivers
    print("\\n🔄 Position Changes (Top 10):")
    for i in range(min(10, len(sample_pred))):
        sample_driver = sample_pred.iloc[i]['driver_name']
        sample_pos = i + 1
        
        # Find this driver in real predictions
        real_row = real_pred[real_pred['driver_name'] == sample_driver]
        if not real_row.empty:
            real_pos = real_row.iloc[0]['predicted_position']
            change = real_pos - sample_pos
            
            if change > 0:
                print(f"   📉 {sample_driver}: P{sample_pos} → P{real_pos} ({change:+d})")
            elif change < 0:
                print(f"   📈 {sample_driver}: P{sample_pos} → P{real_pos} ({change:+d})")
            else:
                print(f"   ➡️ {sample_driver}: P{sample_pos} (no change)")

def display_single_prediction(pred_df, data_type):
    """Display a single prediction set"""
    print(f"\\n🏁 2025 F1 PREDICTIONS ({data_type})")
    print("="*60)
    
    for _, row in pred_df.iterrows():
        pos_icon = "🥇" if row['predicted_position'] == 1 else "🥈" if row['predicted_position'] == 2 else "🥉" if row['predicted_position'] == 3 else "🏁" if row['predicted_position'] <= 10 else "  "
        print(f"{pos_icon} P{row['predicted_position']:2d}: {row['driver_name']:<25} ({row['team']:<15})")

def create_comparison_visualization(predictions):
    """Create visual comparison of predictions"""
    if len(predictions) < 2:
        print("⚠️ Need both sample and real predictions for comparison")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    sample_pred = predictions['sample']
    real_pred = predictions['real']
    
    # Podium comparison
    sample_podium = sample_pred.head(3)
    real_podium = real_pred.head(3)
    
    axes[0, 0].bar(range(3), [3, 2, 1], color=['gold', 'silver', '#CD7F32'], alpha=0.7)
    axes[0, 0].set_xticks(range(3))
    axes[0, 0].set_xticklabels([f"{row['driver_name']}\\n{row['team']}" for _, row in sample_podium.iterrows()], 
                              rotation=45, ha='right')
    axes[0, 0].set_title('Sample Data Podium')
    axes[0, 0].invert_yaxis()
    
    axes[0, 1].bar(range(3), [3, 2, 1], color=['gold', 'silver', '#CD7F32'], alpha=0.7)
    axes[0, 1].set_xticks(range(3))
    axes[0, 1].set_xticklabels([f"{row['driver_name']}\\n{row['team']}" for _, row in real_podium.iterrows()], 
                              rotation=45, ha='right')
    axes[0, 1].set_title('Real Data Podium')
    axes[0, 1].invert_yaxis()
    
    # Team performance comparison
    sample_teams = sample_pred.groupby('team')['predicted_position'].mean().sort_values()
    real_teams = real_pred.groupby('team')['predicted_position'].mean().sort_values()
    
    axes[1, 0].barh(range(len(sample_teams)), sample_teams.values, color='lightblue')
    axes[1, 0].set_yticks(range(len(sample_teams)))
    axes[1, 0].set_yticklabels(sample_teams.index, fontsize=8)
    axes[1, 0].set_title('Sample Data Team Performance')
    axes[1, 0].invert_yaxis()
    
    axes[1, 1].barh(range(len(real_teams)), real_teams.values, color='lightcoral')
    axes[1, 1].set_yticks(range(len(real_teams)))
    axes[1, 1].set_yticklabels(real_teams.index, fontsize=8)
    axes[1, 1].set_title('Real Data Team Performance')
    axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('sample_vs_real_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Comparison visualization saved as 'sample_vs_real_comparison.png'")

def main():
    """Main comparison function"""
    print("🔬 F1 PREDICTION COMPARISON TOOL")
    print("="*60)
    
    # Load available predictions
    predictions = load_predictions()
    
    if not predictions:
        print("❌ No prediction files found")
        print("\\nTo generate predictions:")
        print("1. Sample data: python3 f1_simple_predictor.py")
        print("2. Real data: python3 f1_predictor_with_kaggle.py (after Kaggle setup)")
        return
    
    # Perform comparison
    compare_predictions(predictions)
    
    # Create visualization if both available
    if len(predictions) == 2:
        create_comparison_visualization(predictions)
    
    # Summary
    print("\\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)
    
    if 'real' in predictions:
        print("🔥 REAL DATA PREDICTIONS AVAILABLE!")
        print("   ✅ Using authentic 70+ years of F1 history")
        print("   ✅ Enhanced accuracy and realism")
        print("   ✅ Ready for 2025 season predictions")
    else:
        print("📊 CURRENTLY USING SAMPLE DATA")
        print("   ⚠️ For best accuracy, connect to real Kaggle dataset")
        print("   📋 Follow instructions in KAGGLE_CONNECTION_GUIDE.md")
        print("   🎯 Real data will significantly improve predictions")
    
    print("\\n🏁 Neural network ready for F1 2025 season!")

if __name__ == "__main__":
    main()