#!/usr/bin/env python3
"""
Deploy F1 Neural Network Predictor to GitHub Pages

This script helps set up and deploy the web application to GitHub Pages.
"""

import os
import json
import subprocess
import shutil
from pathlib import Path

def check_git_repository():
    """Check if we're in a git repository"""
    print("🔍 CHECKING GIT REPOSITORY")
    print("="*40)
    
    if os.path.exists('.git'):
        print("✅ Git repository detected")
        
        # Check remote origin
        try:
            result = subprocess.run(['git', 'remote', 'get-url', 'origin'], 
                                  capture_output=True, text=True, check=True)
            remote_url = result.stdout.strip()
            print(f"📡 Remote origin: {remote_url}")
            return True
        except subprocess.CalledProcessError:
            print("⚠️ No remote origin configured")
            return False
    else:
        print("❌ Not a git repository")
        return False

def initialize_git_repository():
    """Initialize git repository if needed"""
    print("\n🔧 INITIALIZING GIT REPOSITORY")
    print("="*40)
    
    try:
        # Initialize git if not already done
        if not os.path.exists('.git'):
            subprocess.run(['git', 'init'], check=True)
            print("✅ Git repository initialized")
        
        # Create .gitignore
        gitignore_content = """# F1 Neural Network Predictor - Git Ignore

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/

# Jupyter Notebook
.ipynb_checkpoints

# Data files (large)
*.h5
*.pkl
*.joblib

# Model files
best_f1_model.h5

# Large image files
*.png
!docs/images/*.png

# Sample data backup
f1_sample_data_backup/
f1_data_modern/

# Temporary files
*.tmp
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Keep docs folder for GitHub Pages
!docs/
!docs/**/*
"""
        
        with open('.gitignore', 'w') as f:
            f.write(gitignore_content)
        print("✅ .gitignore created")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error initializing git: {e}")
        return False

def prepare_web_assets():
    """Prepare web assets for deployment"""
    print("\n📦 PREPARING WEB ASSETS")
    print("="*40)
    
    # Copy latest predictions to web app
    prediction_files = [
        'f1_2025_predictions.csv',
        'f1_2025_predictions_sample.csv',
        'f1_2025_predictions_kaggle.csv'
    ]
    
    copied_files = []
    for file in prediction_files:
        if os.path.exists(file):
            shutil.copy(file, f'docs/data/{os.path.basename(file)}')
            copied_files.append(file)
            print(f"✅ Copied {file}")
    
    if not copied_files:
        print("⚠️ No prediction files found to copy")
        return False
    
    # Convert to JSON format
    try:
        import pandas as pd
        
        for file in copied_files:
            df = pd.read_csv(file)
            json_file = file.replace('.csv', '.json')
            
            predictions = df.to_dict('records')
            with open(f'docs/data/{os.path.basename(json_file)}', 'w') as f:
                json.dump(predictions, f, indent=2)
            
            print(f"✅ Converted {file} to JSON")
    
    except Exception as e:
        print(f"⚠️ Error converting to JSON: {e}")
    
    # Copy visualization images if they exist
    image_files = [
        'model_evaluation.png',
        '2025_race_predictions.png',
        'f1_data_exploration.png',
        'quick_predictions_summary.png'
    ]
    
    for img_file in image_files:
        if os.path.exists(img_file):
            shutil.copy(img_file, f'docs/images/{img_file}')
            print(f"✅ Copied {img_file}")
    
    print(f"✅ Web assets prepared successfully")
    return True

def create_github_pages_config():
    """Create GitHub Pages specific configuration"""
    print("\n⚙️ CREATING GITHUB PAGES CONFIG")
    print("="*40)
    
    # Create GitHub workflow for automated deployment
    workflow_dir = Path('.github/workflows')
    workflow_dir.mkdir(parents=True, exist_ok=True)
    
    workflow_content = """name: Deploy F1 Predictor to GitHub Pages

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        
      - name: Setup Pages
        uses: actions/configure-pages@v4
        
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: './docs'
          
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
"""
    
    with open(workflow_dir / 'deploy.yml', 'w') as f:
        f.write(workflow_content)
    
    print("✅ GitHub Actions workflow created")
    
    # Create CNAME file if custom domain is needed
    # Uncomment and modify if you have a custom domain
    # with open('docs/CNAME', 'w') as f:
    #     f.write('your-custom-domain.com')
    # print("✅ CNAME file created")
    
    return True

def create_deployment_guide():
    """Create deployment guide"""
    print("\n📝 CREATING DEPLOYMENT GUIDE")
    print("="*40)
    
    guide_content = """# 🚀 GitHub Pages Deployment Guide

## Quick Deployment Steps

### 1. Repository Setup
```bash
# If not already a git repository
git init
git add .
git commit -m "Initial F1 Neural Network Predictor"

# Add GitHub remote (replace with your repository)
git remote add origin https://github.com/yourusername/f1-neural-predictor.git
git push -u origin main
```

### 2. Enable GitHub Pages
1. Go to your GitHub repository
2. Click **Settings** tab
3. Scroll to **Pages** section
4. Under **Source**, select **Deploy from a branch**
5. Choose **main** branch and **/ (root)** folder
6. Click **Save**

### 3. Configure for /docs folder (Alternative)
If you want to serve from /docs folder:
1. In Pages settings, select **main** branch and **/docs** folder
2. Click **Save**
3. Your site will be available at: `https://yourusername.github.io/repository-name`

### 4. Custom Domain (Optional)
1. Add CNAME file to docs/ folder with your domain
2. Configure DNS settings with your domain provider
3. Enable HTTPS in GitHub Pages settings

## 🔄 Updating Predictions

### Automatic Updates
When you update predictions:
```bash
# Generate new predictions
python3 f1_predictor_with_kaggle.py

# Convert to web format
python3 deploy_to_github_pages.py

# Commit and push
git add .
git commit -m "Update 2025 F1 predictions"
git push origin main
```

### Manual Updates
1. Update CSV files in repository root
2. Run: `python3 deploy_to_github_pages.py`
3. Commit and push changes

## 🎨 Customization

### Update Repository URL
Replace `yourusername` and `f1-neural-predictor` in:
- `docs/_config.yml`
- `docs/index.html` (GitHub links)
- This deployment guide

### Custom Styling
- Modify `docs/css/style.css` for visual changes
- Update `docs/js/app.js` for functionality changes
- Add new visualizations in `docs/js/charts.js`

## 🔧 Troubleshooting

### Common Issues
1. **404 Error**: Check GitHub Pages source settings
2. **CSS Not Loading**: Verify file paths in HTML
3. **Data Not Loading**: Check JSON/CSV file accessibility
4. **Charts Not Displaying**: Verify Chart.js CDN availability

### Debug Mode
Add to browser console:
```javascript
// Check if data loaded
console.log('Predictions:', window.F1App.predictionsData());

// Check data source
console.log('Data source:', window.F1App.currentDataSource());
```

## 🌟 Features

✅ **Responsive Design** - Works on all devices  
✅ **Interactive Charts** - Powered by Chart.js  
✅ **Real Data Ready** - Automatically uses Kaggle data when available  
✅ **Fast Loading** - Optimized for GitHub Pages  
✅ **SEO Optimized** - Proper meta tags and structure  

## 🏁 Go Live!

Your F1 Neural Network Predictor web app is ready for deployment!
"""
    
    with open('GITHUB_PAGES_DEPLOYMENT.md', 'w') as f:
        f.write(guide_content)
    
    print("✅ Deployment guide created: GITHUB_PAGES_DEPLOYMENT.md")
    return True

def create_web_app_demo():
    """Create a local demo of the web app"""
    print("\n🎮 CREATING LOCAL WEB APP DEMO")
    print("="*40)
    
    demo_script = """#!/usr/bin/env python3
'''
Local F1 Web App Demo Server
'''

import http.server
import socketserver
import webbrowser
import os
import threading
import time

def start_demo_server():
    PORT = 8000
    
    # Change to docs directory
    os.chdir('docs')
    
    # Create server
    Handler = http.server.SimpleHTTPRequestHandler
    
    # Add CORS headers for local development
    class CORSRequestHandler(Handler):
        def end_headers(self):
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', '*')
            super().end_headers()
    
    with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
        print(f"🌐 F1 Web App Demo Server running at: http://localhost:{PORT}")
        print("🏎️ Opening web browser...")
        
        # Open browser after short delay
        def open_browser():
            time.sleep(2)
            webbrowser.open(f'http://localhost:{PORT}')
        
        threading.Thread(target=open_browser, daemon=True).start()
        
        print("\\n🔧 Press Ctrl+C to stop the server")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\\n🛑 Server stopped")

if __name__ == "__main__":
    start_demo_server()
"""
    
    with open('run_web_demo.py', 'w') as f:
        f.write(demo_script)
    
    print("✅ Demo server script created: run_web_demo.py")
    return True

def main():
    """Main deployment preparation function"""
    print("🚀 F1 NEURAL NETWORK - GITHUB PAGES DEPLOYMENT")
    print("="*60)
    
    # Check current setup
    git_ready = check_git_repository()
    
    # Initialize git if needed
    if not git_ready:
        initialize_git_repository()
    
    # Prepare web assets
    if not prepare_web_assets():
        print("❌ Failed to prepare web assets")
        return False
    
    # Create GitHub Pages configuration
    create_github_pages_config()
    
    # Create deployment guide
    create_deployment_guide()
    
    # Create demo server
    create_web_app_demo()
    
    print("\n🎉 GITHUB PAGES DEPLOYMENT READY!")
    print("="*50)
    print("\\n📋 Next Steps:")
    print("1. 🌐 Test locally: python3 run_web_demo.py")
    print("2. 📤 Push to GitHub: git add . && git commit -m 'Deploy F1 web app' && git push")
    print("3. ⚙️ Enable GitHub Pages in repository settings")
    print("4. 🏁 Visit your live web app!")
    
    print("\\n🔗 Your web app will be available at:")
    print("   https://yourusername.github.io/f1-neural-predictor")
    
    print("\\n💡 For detailed instructions, see: GITHUB_PAGES_DEPLOYMENT.md")
    
    return True

if __name__ == "__main__":
    main()