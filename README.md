# 🏎️ F1 Neural Network Predictor - Live F1DB System

Advanced Formula 1 driver position prediction using deep learning neural networks powered by **live F1DB data** updated after every race weekend.

## 🔥 **Live Data Integration**

**✅ F1DB Database**: https://github.com/f1db/f1db  
**✅ Real-time Updates**: Data refreshed after every race weekend  
**✅ Authentic Data**: 75+ years of genuine F1 history (1950-2025)  
**✅ Current Season**: 2025 data through Hungarian Grand Prix  

---

## 🏁 **2025 Race Predictions**

### 🏆 **Predicted Podium (F1DB v2025.14.0)**
1. **🥇 Alexander Albon** (Williams)
2. **🥈 Andrea Kimi Antonelli** (Mercedes)
3. **🥉 Carlos Sainz** (Williams)

### 📊 **Model Performance**
- **Training Data**: 8,195 authentic race results (2000-2025)
- **Mean Absolute Error**: 4.50 positions
- **Within 5 positions**: 62.0% accuracy
- **Neural Network**: 320,001 parameters

---

## 🚀 **Quick Start**

### **Run F1DB System**
```bash
# Install dependencies
pip3 install --break-system-packages -r requirements.txt

# Run the complete F1DB system
python3 run_f1db_system.py

# Or run neural network directly
python3 f1db_neural_predictor.py
```

### **Web Application**
```bash
# Start local web app
python3 run_web_demo.py
# Visit: http://localhost:8000

# Deploy to GitHub Pages
git add . && git commit -m "F1DB live system" && git push
# Visit: https://ihackmer19.github.io/Formula-1-Predictor
```

---

## 🧠 **Neural Network Architecture**

### **Enhanced F1DB Model**
- **Input**: 37 F1DB-specific features
- **Architecture**: 6-layer deep network (256→512→256→128→64→32→1)
- **Regularization**: Dropout + Batch Normalization
- **Optimizer**: Adam with Huber loss (robust to outliers)
- **Parameters**: 320,001 trainable parameters

### **F1DB Features**
- **Historical Performance**: Rolling averages (3, 5, 10 races)
- **Career Statistics**: Wins, podiums, points, race count
- **Circuit-Specific**: Track performance history
- **Constructor Metrics**: Team performance indicators
- **Era Features**: Current regulations, turbo-hybrid era
- **Circuit Data**: Length, turns, type characteristics
- **Form Indicators**: Recent performance trends

---

## 📊 **F1DB Data Overview**

### **Dataset Scale**
- **Years**: 1950 - 2025 (76 years)
- **Races**: 1,149 Grand Prix events
- **Results**: 27,091 race results
- **Drivers**: 912 F1 drivers in history
- **Constructors**: 185 teams/constructors
- **Circuits**: 77 F1 circuits

### **Current Season (2025)**
- **Races Completed**: 24 (through Hungarian GP)
- **Active Drivers**: 27 drivers in recent data
- **Active Teams**: 11 constructors
- **Latest Update**: Hungarian Grand Prix 2025

---

## 🔄 **Automatic Updates**

### **F1DB Auto-Updater**
```bash
# Manual update check
python3 f1db_auto_updater.py

# Continuous monitoring
python3 f1db_update_service.py
```

### **Update Schedule**
- **📅 Monday 09:00**: Post-race weekend updates
- **📅 Daily 12:00**: Missed update checks
- **🔄 Automatic**: Download, retrain, deploy

### **Update Process**
1. **Monitor**: Check F1DB GitHub releases
2. **Download**: Latest CSV data automatically
3. **Retrain**: Neural network with new data
4. **Deploy**: Update web app predictions
5. **Log**: Track all update attempts

---

## 🌐 **Web Application**

### **Live Features**
- **🔥 F1DB Integration**: Real-time data loading
- **📊 Interactive Charts**: Team and driver analysis
- **📱 Mobile Optimized**: Responsive design
- **🏁 Live Predictions**: Updated after each race

### **GitHub Pages Deployment**
```bash
# Enable GitHub Pages in repository settings
# Source: Deploy from 'docs' folder
# URL: https://ihackmer19.github.io/Formula-1-Predictor
```

---

## 📁 **Project Structure**

```
├── 🔥 F1DB CORE SYSTEM
│   ├── f1db_neural_predictor.py    # Main F1DB neural network
│   ├── f1db_auto_updater.py       # Automatic update system
│   ├── run_f1db_system.py         # System interface
│   └── f1db_version.txt           # Current F1DB version
│
├── 📊 LIVE F1DB DATA
│   ├── f1db-races.csv             # Race information
│   ├── f1db-races-race-results.csv # Race results (27K+ results)
│   ├── f1db-drivers.csv           # Driver database (912 drivers)
│   ├── f1db-constructors.csv      # Constructor data (185 teams)
│   └── f1db-*.csv                 # Additional F1DB datasets
│
├── 🏁 PREDICTIONS & MODELS
│   ├── f1db_2025_predictions.csv  # Latest 2025 predictions
│   ├── f1db_best_model.h5         # Trained neural network
│   ├── f1db_model_evaluation.png  # Performance analysis
│   └── f1db_update_log.json       # Update history
│
├── 🌐 WEB APPLICATION
│   └── docs/                      # GitHub Pages app
│       ├── index.html             # F1DB-integrated interface
│       ├── data/f1db_predictions.* # Live prediction data
│       └── js/app.js              # F1DB data loading
│
└── 📖 DOCUMENTATION
    ├── F1DB_LIVE_SYSTEM.md        # F1DB integration guide
    ├── README.md                  # This file
    └── requirements.txt           # Dependencies
```

---

## 🎯 **Key Features**

### **🔥 Live Data**
- **Real F1 Database**: Authentic data from 1950-2025
- **Race Weekend Updates**: New data after every Sunday race
- **No Synthetic Data**: 100% authentic F1 information
- **Complete History**: 75+ years of Formula 1 evolution

### **🧠 Advanced AI**
- **Deep Neural Network**: 6-layer architecture
- **F1-Specific Features**: 37 racing-focused inputs
- **Temporal Modeling**: Historical performance analysis
- **Robust Training**: Handles real-world F1 variability

### **🌐 Professional Web App**
- **GitHub Pages Ready**: Instant deployment
- **Live Data Integration**: F1DB predictions in real-time
- **Mobile Responsive**: Perfect on all devices
- **Interactive Charts**: Team and driver analysis

### **🔄 Automated System**
- **Auto-Updates**: Monitors F1DB for new releases
- **Smart Retraining**: Updates model with latest data
- **Web Integration**: Automatically updates predictions
- **Error Handling**: Robust failure recovery

---

## 🎮 **Usage Examples**

### **Basic Prediction**
```bash
# Run F1DB neural network
python3 f1db_neural_predictor.py

# View predictions
cat f1db_2025_predictions.csv
```

### **Web Interface**
```bash
# Start web app locally
python3 run_web_demo.py

# Deploy to GitHub Pages
git push origin main
```

### **System Management**
```bash
# Interactive system interface
python3 run_f1db_system.py

# Check for F1DB updates
python3 f1db_auto_updater.py

# Start monitoring service
python3 f1db_update_service.py
```

---

## 🏆 **Advantages Over Static Data**

### **🔥 F1DB vs Static Datasets**
| Feature | F1DB Live | Static Kaggle |
|---------|-----------|---------------|
| **Updates** | After every race | Fixed (2020) |
| **Current Data** | 2025 season | Outdated |
| **Accuracy** | High (real patterns) | Limited |
| **Relevance** | Current drivers/teams | Historical only |
| **Maintenance** | Community supported | Unmaintained |

### **🎯 Real-World Benefits**
- **Current Grid**: Actual 2025 F1 drivers and teams
- **Recent Form**: Latest performance data included
- **Rule Changes**: Accounts for current F1 regulations
- **Team Evolution**: Real constructor changes and mergers
- **Driver Transfers**: Actual 2025 driver lineup

---

## 🔧 **Technical Requirements**

### **Python Dependencies**
```bash
pip3 install pandas numpy scikit-learn tensorflow matplotlib seaborn requests schedule
```

### **System Requirements**
- **Python**: 3.8+ (tested with 3.13)
- **Memory**: 4GB+ RAM for neural network training
- **Storage**: 100MB for F1DB data files
- **Network**: Internet connection for F1DB updates

---

## 🌟 **Future Enhancements**

### **Planned Features**
- **Live Timing Integration**: Real-time race data
- **Weather Data**: Track conditions and forecasts
- **Strategy Analysis**: Pit stop and tire strategy factors
- **Ensemble Models**: Multiple prediction algorithms

### **Community Contributions**
- **Feature Requests**: GitHub issues welcome
- **Data Improvements**: F1DB community contributions
- **Model Enhancements**: Neural network optimizations
- **Web App Features**: UI/UX improvements

---

## 📞 **Support & Community**

### **Resources**
- **F1DB Database**: https://github.com/f1db/f1db
- **Project Repository**: https://github.com/IHackmer19/Formula-1-Predictor
- **Live Web App**: https://ihackmer19.github.io/Formula-1-Predictor
- **Documentation**: F1DB_LIVE_SYSTEM.md

### **Getting Help**
- **GitHub Issues**: Report bugs or request features
- **F1DB Community**: Join F1 data discussions
- **Documentation**: Comprehensive guides provided

---

## 🏁 **Ready for 2025 F1 Season!**

**🔥 Your F1 Neural Network Predictor now uses live, authentic Formula 1 data!**

- ✅ **Real Data**: F1DB live database integration
- ✅ **Auto-Updates**: Monitors race weekend updates
- ✅ **Web Ready**: Professional GitHub Pages deployment
- ✅ **Production Quality**: 320K parameter neural network

**🏎️ Experience the future of Formula 1 predictions with live data! 🏆**