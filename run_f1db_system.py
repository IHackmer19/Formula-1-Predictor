#!/usr/bin/env python3
"""
F1DB Live System Runner

Main entry point for the F1DB-powered neural network prediction system.
"""

import os
import subprocess
import sys
import json
from datetime import datetime

def check_f1db_data():
    """Check if F1DB data is available"""
    print("🔍 CHECKING F1DB DATA AVAILABILITY")
    print("="*50)
    
    required_files = [
        'f1db-races.csv',
        'f1db-races-race-results.csv',
        'f1db-drivers.csv',
        'f1db-constructors.csv'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing F1DB files: {missing_files}")
        print("\n📥 Downloading latest F1DB data...")
        
        # Try to download automatically
        try:
            from f1db_neural_predictor import F1DBNeuralPredictor
            predictor = F1DBNeuralPredictor()
            if predictor.check_f1db_update():
                print("✅ F1DB data downloaded successfully")
                return True
            else:
                print("❌ Failed to download F1DB data")
                return False
        except Exception as e:
            print(f"❌ Error downloading F1DB data: {e}")
            return False
    else:
        print("✅ F1DB data files available")
        return True

def show_current_predictions():
    """Show current F1DB predictions"""
    print("\n🏁 CURRENT F1DB PREDICTIONS")
    print("="*50)
    
    try:
        import pandas as pd
        
        if os.path.exists('f1db_2025_predictions.csv'):
            df = pd.read_csv('f1db_2025_predictions.csv')
            
            print("🏆 2025 Next Race Predictions:")
            for _, row in df.head(10).iterrows():
                pos_icon = "🥇" if row['predicted_position'] == 1 else "🥈" if row['predicted_position'] == 2 else "🥉" if row['predicted_position'] == 3 else "🏁"
                print(f"   {pos_icon} P{row['predicted_position']:2d}: {row['driver_name']:<25} ({row['team']:<15})")
            
            if len(df) > 10:
                print(f"   ... and {len(df) - 10} more drivers")
            
            return True
        else:
            print("❌ No predictions available - run the neural network first")
            return False
            
    except Exception as e:
        print(f"❌ Error loading predictions: {e}")
        return False

def show_f1db_status():
    """Show F1DB system status"""
    print("\n📊 F1DB SYSTEM STATUS")
    print("="*40)
    
    # Check version
    if os.path.exists('f1db_version.txt'):
        with open('f1db_version.txt', 'r') as f:
            version = f.read().strip()
        print(f"📋 F1DB Version: {version}")
    else:
        print("📋 F1DB Version: Not installed")
    
    # Check data files
    f1db_files = [f for f in os.listdir('.') if f.startswith('f1db-') and f.endswith('.csv')]
    print(f"📁 F1DB Files: {len(f1db_files)} CSV files")
    
    # Check model
    if os.path.exists('f1db_best_model.h5'):
        print("🧠 Model: Trained and saved")
    else:
        print("🧠 Model: Not trained yet")
    
    # Check predictions
    if os.path.exists('f1db_2025_predictions.csv'):
        print("🏁 Predictions: Available")
    else:
        print("🏁 Predictions: Not generated yet")
    
    # Check web app integration
    if os.path.exists('docs/data/f1db_predictions.json'):
        print("🌐 Web App: F1DB data integrated")
    else:
        print("🌐 Web App: Not updated with F1DB data")

def run_neural_network():
    """Run the F1DB neural network predictor"""
    print("\n🧠 RUNNING F1DB NEURAL NETWORK")
    print("="*50)
    
    try:
        result = subprocess.run(['python3', 'f1db_neural_predictor.py'], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n✅ Neural network completed successfully!")
            return True
        else:
            print(f"\n❌ Neural network failed with exit code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running neural network: {e}")
        return False

def start_web_app():
    """Start the web application"""
    print("\n🌐 STARTING F1DB WEB APPLICATION")
    print("="*50)
    
    try:
        print("🚀 Starting web server...")
        print("🔗 Web app will be available at: http://localhost:8000")
        print("🏁 Press Ctrl+C to stop")
        
        subprocess.run(['python3', 'run_web_demo.py'])
        
    except KeyboardInterrupt:
        print("\n🛑 Web app stopped")
    except Exception as e:
        print(f"❌ Error starting web app: {e}")

def setup_auto_updates():
    """Set up automatic F1DB updates"""
    print("\n⏰ SETTING UP AUTO-UPDATES")
    print("="*40)
    
    try:
        print("🔄 Starting F1DB monitoring service...")
        print("📅 Will check for updates after race weekends")
        print("🏁 Press Ctrl+C to stop")
        
        subprocess.run(['python3', 'f1db_update_service.py'])
        
    except KeyboardInterrupt:
        print("\n🛑 Auto-update service stopped")
    except Exception as e:
        print(f"❌ Error starting auto-updates: {e}")

def main():
    """Main F1DB system interface"""
    print("🏎️ F1DB LIVE DATA NEURAL NETWORK SYSTEM")
    print("="*60)
    print("🔥 Using live F1 database updated after every race weekend")
    print("📊 Source: https://github.com/f1db/f1db")
    print("="*60)
    
    # Check system status
    show_f1db_status()
    
    # Check if data is available
    if not check_f1db_data():
        print("\n❌ F1DB data not available")
        print("🔧 Please ensure internet connection and try again")
        return
    
    # Show current predictions if available
    show_current_predictions()
    
    # Interactive menu
    print("\n🎮 F1DB SYSTEM OPTIONS")
    print("="*40)
    print("1. 🧠 Run Neural Network Predictor")
    print("2. 🌐 Start Web Application")
    print("3. 🔄 Check for F1DB Updates")
    print("4. ⏰ Setup Auto-Updates")
    print("5. 📊 Show System Status")
    print("6. 🚪 Exit")
    
    try:
        while True:
            choice = input("\nEnter choice (1-6): ").strip()
            
            if choice == '1':
                run_neural_network()
                show_current_predictions()
            elif choice == '2':
                start_web_app()
            elif choice == '3':
                from f1db_auto_updater import F1DBAutoUpdater
                updater = F1DBAutoUpdater()
                updater.run_update_check()
            elif choice == '4':
                setup_auto_updates()
            elif choice == '5':
                show_f1db_status()
                show_current_predictions()
            elif choice == '6':
                print("🏁 Goodbye! Thanks for using F1DB Neural Predictor!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-6.")
                
    except KeyboardInterrupt:
        print("\n🏁 F1DB system stopped")

if __name__ == "__main__":
    main()