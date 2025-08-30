#!/bin/bash

echo "🏎️ FORMULA 1 KAGGLE DATASET SETUP"
echo "=================================="

echo ""
echo "📋 MANUAL DOWNLOAD INSTRUCTIONS:"
echo ""
echo "1. 🌐 Visit the dataset page:"
echo "   https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020"
echo ""
echo "2. 📥 Download the dataset:"
echo "   - Click the 'Download' button (requires Kaggle account)"
echo "   - This downloads a ZIP file (~50MB)"
echo ""
echo "3. 📂 Extract the dataset:"
echo "   - Extract all CSV files to this directory"
echo "   - You should see files like: circuits.csv, drivers.csv, results.csv, etc."
echo ""
echo "4. 🔄 Run the real data processor:"
echo "   python3 real_f1_neural_network.py"
echo ""
echo "=================================="
echo ""

# Check if dataset files already exist
if [ -f "results.csv" ] && [ -f "drivers.csv" ] && [ -f "races.csv" ]; then
    echo "✅ Real dataset files found!"
    echo "🚀 Running real data analysis..."
    python3 real_f1_neural_network.py
else
    echo "❌ Real dataset files not found in current directory"
    echo ""
    echo "🔧 ALTERNATIVE: Set up Kaggle API"
    echo "================================"
    echo ""
    echo "If you prefer using the API:"
    echo "1. Get your API token from https://www.kaggle.com/account"
    echo "2. Create ~/.config/kaggle/kaggle.json with your credentials"
    echo "3. Run: kaggle datasets download -d rohanrao/formula-1-world-championship-1950-2020 --unzip"
    echo ""
    echo "💡 For now, the system will continue with sample data for demonstration."
    echo "   Real data will provide much more accurate predictions!"
fi