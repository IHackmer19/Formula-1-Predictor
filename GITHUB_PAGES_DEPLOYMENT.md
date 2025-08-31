# 🚀 GitHub Pages Deployment Guide

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
