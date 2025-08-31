#!/usr/bin/env python3
"""
F1DB Automatic Updater

This script automatically checks for F1DB updates after race weekends
and retrains the neural network with the latest data.
"""

import requests
import json
import subprocess
import os
import pandas as pd
from datetime import datetime, timedelta
import schedule
import time

class F1DBAutoUpdater:
    def __init__(self):
        self.current_version = self._get_current_version()
        self.last_check = None
        
    def _get_current_version(self):
        """Get currently installed F1DB version"""
        try:
            with open('f1db_version.txt', 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            return None
    
    def check_for_updates(self):
        """Check if new F1DB version is available"""
        print("🔍 Checking for F1DB updates...")
        
        try:
            response = requests.get('https://api.github.com/repos/f1db/f1db/releases/latest')
            release_data = response.json()
            
            latest_version = release_data['tag_name']
            release_date = release_data['published_at']
            
            print(f"📊 Latest F1DB: {latest_version}")
            print(f"📅 Released: {release_date}")
            print(f"📋 Current: {self.current_version or 'None'}")
            
            if self.current_version != latest_version:
                print(f"🆕 New version available: {self.current_version} → {latest_version}")
                return True, latest_version, release_data
            else:
                print("✅ Already up to date")
                return False, latest_version, release_data
                
        except Exception as e:
            print(f"❌ Error checking updates: {e}")
            return False, None, None
    
    def download_latest_f1db(self, release_data):
        """Download latest F1DB data"""
        print("\n📥 DOWNLOADING LATEST F1DB DATA")
        print("="*50)
        
        # Find CSV asset
        csv_asset = None
        for asset in release_data['assets']:
            if asset['name'] == 'f1db-csv.zip':
                csv_asset = asset
                break
        
        if not csv_asset:
            print("❌ CSV data not found in release")
            return False
        
        try:
            # Download
            download_url = csv_asset['browser_download_url']
            print(f"📥 Downloading {csv_asset['name']} ({csv_asset['size'] / 1024 / 1024:.1f} MB)...")
            
            response = requests.get(download_url)
            with open('f1db-csv-latest.zip', 'wb') as f:
                f.write(response.content)
            
            # Extract
            import zipfile
            with zipfile.ZipFile('f1db-csv-latest.zip', 'r') as zip_ref:
                zip_ref.extractall('.')
            
            # Clean up
            os.remove('f1db-csv-latest.zip')
            
            # Update version
            with open('f1db_version.txt', 'w') as f:
                f.write(release_data['tag_name'])
            
            self.current_version = release_data['tag_name']
            print(f"✅ Downloaded F1DB {self.current_version}")
            return True
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    
    def retrain_model(self):
        """Retrain the neural network with updated data"""
        print("\n🎓 RETRAINING MODEL WITH LATEST DATA")
        print("="*50)
        
        try:
            # Run the F1DB neural predictor
            result = subprocess.run(['python3', 'f1db_neural_predictor.py'], 
                                  capture_output=True, text=True, timeout=1800)  # 30 min timeout
            
            if result.returncode == 0:
                print("✅ Model retrained successfully!")
                print("📊 Latest predictions generated")
                
                # Update web app data
                self._update_web_app_data()
                return True
            else:
                print(f"❌ Model training failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("⏰ Model training timed out")
            return False
        except Exception as e:
            print(f"❌ Error retraining model: {e}")
            return False
    
    def _update_web_app_data(self):
        """Update web app with latest predictions"""
        try:
            # Convert predictions to web format
            if os.path.exists('f1db_2025_predictions.csv'):
                df = pd.read_csv('f1db_2025_predictions.csv')
                predictions = df.to_dict('records')
                
                # Save to web app
                with open('docs/data/f1db_predictions.json', 'w') as f:
                    json.dump(predictions, f, indent=2)
                
                df.to_csv('docs/data/f1db_predictions.csv', index=False)
                
                print("✅ Web app data updated")
                return True
        except Exception as e:
            print(f"⚠️ Web app update failed: {e}")
            return False
    
    def run_update_check(self):
        """Run complete update check and retrain if needed"""
        print(f"🔄 F1DB AUTO-UPDATE CHECK - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        update_available, latest_version, release_data = self.check_for_updates()
        
        if update_available:
            print("🆕 Update available! Starting download and retrain...")
            
            if self.download_latest_f1db(release_data):
                if self.retrain_model():
                    print("🏆 Update complete! Latest F1 predictions ready.")
                    
                    # Log successful update
                    self._log_update(latest_version, True)
                    return True
                else:
                    print("❌ Retrain failed")
                    self._log_update(latest_version, False)
                    return False
            else:
                print("❌ Download failed")
                return False
        else:
            print("✅ No updates needed")
            return True
    
    def _log_update(self, version, success):
        """Log update attempts"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'version': version,
            'success': success
        }
        
        # Read existing log
        log_file = 'f1db_update_log.json'
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_data = json.load(f)
        else:
            log_data = []
        
        # Add new entry
        log_data.append(log_entry)
        
        # Keep only last 50 entries
        log_data = log_data[-50:]
        
        # Save log
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def setup_scheduled_updates(self):
        """Set up scheduled updates for race weekends"""
        print("⏰ SETTING UP SCHEDULED F1DB UPDATES")
        print("="*50)
        
        # Schedule updates for Monday after race weekends (when F1DB typically updates)
        schedule.every().monday.at("09:00").do(self.run_update_check)
        
        # Also check daily for any missed updates
        schedule.every().day.at("12:00").do(self.run_update_check)
        
        print("✅ Scheduled updates configured:")
        print("   📅 Every Monday at 09:00 (post-race weekend)")
        print("   📅 Daily at 12:00 (missed update check)")
        
        return True
    
    def run_scheduler(self):
        """Run the update scheduler"""
        print("🔄 F1DB AUTO-UPDATER RUNNING")
        print("="*40)
        print("Monitoring for F1DB updates after race weekends...")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(3600)  # Check every hour
        except KeyboardInterrupt:
            print("\n🛑 Auto-updater stopped")

def create_update_service():
    """Create a service script for continuous monitoring"""
    service_script = '''#!/usr/bin/env python3
"""
F1DB Update Service - Runs continuously to monitor for updates
"""

from f1db_auto_updater import F1DBAutoUpdater
import sys

def main():
    updater = F1DBAutoUpdater()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--check-now':
        # Manual update check
        updater.run_update_check()
    else:
        # Start scheduled service
        updater.setup_scheduled_updates()
        updater.run_scheduler()

if __name__ == "__main__":
    main()
'''
    
    with open('f1db_update_service.py', 'w') as f:
        f.write(service_script)
    
    print("✅ Update service created: f1db_update_service.py")

def main():
    """Main function for manual updates"""
    updater = F1DBAutoUpdater()
    
    print("🏎️ F1DB AUTO-UPDATER")
    print("="*40)
    print("Options:")
    print("1. Check for updates now")
    print("2. Set up scheduled updates")
    print("3. View update log")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            updater.run_update_check()
        elif choice == '2':
            updater.setup_scheduled_updates()
            updater.run_scheduler()
        elif choice == '3':
            if os.path.exists('f1db_update_log.json'):
                with open('f1db_update_log.json', 'r') as f:
                    log_data = json.load(f)
                print("\n📋 Update Log:")
                for entry in log_data[-10:]:  # Last 10 entries
                    status = "✅" if entry['success'] else "❌"
                    print(f"   {status} {entry['timestamp']}: {entry['version']}")
            else:
                print("📋 No update log found")
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n⏭️ Cancelled")

if __name__ == "__main__":
    # Install required package
    try:
        import schedule
    except ImportError:
        print("📦 Installing schedule package...")
        subprocess.run(['pip3', 'install', '--break-system-packages', 'schedule'], check=True)
        import schedule
    
    create_update_service()
    main()