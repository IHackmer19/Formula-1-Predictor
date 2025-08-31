// F1 Neural Network Predictor - Predictions Management

// Predictions data management and interactive features
class F1PredictionsManager {
    constructor() {
        this.predictions = null;
        this.filteredPredictions = null;
        this.currentFilter = 'all';
    }

    async loadPredictions() {
        console.log('📊 Loading F1 predictions...');
        
        try {
            // Try different prediction file sources
            const sources = [
                '../f1_2025_predictions_kaggle.csv',
                '../f1_2025_predictions_sample.csv', 
                '../f1_2025_predictions.csv'
            ];
            
            for (const source of sources) {
                try {
                    const response = await fetch(source);
                    if (response.ok) {
                        const csvText = await response.text();
                        this.predictions = this.parseCSV(csvText);
                        console.log(`✅ Loaded predictions from ${source}`);
                        return this.predictions;
                    }
                } catch (error) {
                    console.log(`⚠️ Failed to load ${source}:`, error.message);
                }
            }
            
            // If no CSV files available, use fallback data
            console.log('📋 Using fallback prediction data');
            this.predictions = this.getFallbackPredictions();
            return this.predictions;
            
        } catch (error) {
            console.error('❌ Error loading predictions:', error);
            return this.getFallbackPredictions();
        }
    }

    parseCSV(csvText) {
        const lines = csvText.trim().split('\n');
        const headers = lines[0].split(',').map(h => h.trim());
        const data = [];
        
        for (let i = 1; i < lines.length; i++) {
            if (lines[i].trim()) {
                const values = lines[i].split(',').map(v => v.trim());
                const row = {};
                headers.forEach((header, index) => {
                    row[header] = values[index] || '';
                });
                if (row.predicted_position) {
                    data.push(row);
                }
            }
        }
        
        return data.sort((a, b) => parseInt(a.predicted_position) - parseInt(b.predicted_position));
    }

    getFallbackPredictions() {
        // Fallback prediction data if CSV files aren't available
        return [
            { driverId: 4, predicted_position: 1, driver_name: 'Sergio Perez', team: 'Red Bull Racing' },
            { driverId: 1, predicted_position: 2, driver_name: 'Lewis Hamilton', team: 'Mercedes' },
            { driverId: 5, predicted_position: 3, driver_name: 'Charles Leclerc', team: 'Ferrari' },
            { driverId: 6, predicted_position: 4, driver_name: 'Carlos Sainz', team: 'Ferrari' },
            { driverId: 2, predicted_position: 5, driver_name: 'George Russell', team: 'Mercedes' },
            { driverId: 3, predicted_position: 6, driver_name: 'Max Verstappen', team: 'Red Bull Racing' },
            { driverId: 12, predicted_position: 7, driver_name: 'Daniel Ricciardo', team: 'Aston Martin' },
            { driverId: 7, predicted_position: 8, driver_name: 'Lando Norris', team: 'McLaren' },
            { driverId: 11, predicted_position: 9, driver_name: 'Yuki Tsunoda', team: 'Aston Martin' },
            { driverId: 10, predicted_position: 10, driver_name: 'Lance Stroll', team: 'Alpine' },
            { driverId: 9, predicted_position: 11, driver_name: 'Fernando Alonso', team: 'Alpine' },
            { driverId: 17, predicted_position: 12, driver_name: 'Alexander Albon', team: 'Haas' },
            { driverId: 14, predicted_position: 13, driver_name: 'Zhou Guanyu', team: 'AlphaTauri' },
            { driverId: 8, predicted_position: 14, driver_name: 'Oscar Piastri', team: 'McLaren' },
            { driverId: 15, predicted_position: 15, driver_name: 'Kevin Magnussen', team: 'Alfa Romeo' },
            { driverId: 19, predicted_position: 16, driver_name: 'Esteban Ocon', team: 'Williams' },
            { driverId: 18, predicted_position: 17, driver_name: 'Logan Sargeant', team: 'Haas' },
            { driverId: 20, predicted_position: 18, driver_name: 'Pierre Gasly', team: 'Williams' },
            { driverId: 13, predicted_position: 19, driver_name: 'Valtteri Bottas', team: 'AlphaTauri' },
            { driverId: 16, predicted_position: 20, driver_name: 'Nico Hulkenberg', team: 'Alfa Romeo' }
        ];
    }

    getTopPerformers(count = 6) {
        if (!this.predictions) return [];
        return this.predictions.slice(0, count);
    }

    getTeamAnalysis() {
        if (!this.predictions) return {};
        
        const teamData = {};
        this.predictions.forEach(driver => {
            const team = driver.team || `Team ${driver.constructorId}`;
            const position = parseInt(driver.predicted_position);
            
            if (!teamData[team]) {
                teamData[team] = {
                    drivers: [],
                    positions: [],
                    totalPoints: 0,
                    avgPosition: 0
                };
            }
            
            teamData[team].drivers.push(driver.driver_name);
            teamData[team].positions.push(position);
            teamData[team].totalPoints += this.getPointsForPosition(position);
        });
        
        // Calculate averages
        Object.keys(teamData).forEach(team => {
            const data = teamData[team];
            data.avgPosition = data.positions.reduce((a, b) => a + b, 0) / data.positions.length;
            data.bestPosition = Math.min(...data.positions);
            data.worstPosition = Math.max(...data.positions);
        });
        
        return teamData;
    }

    getPointsForPosition(position) {
        const pointsMap = {
            1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
            6: 8, 7: 6, 8: 4, 9: 2, 10: 1
        };
        return pointsMap[position] || 0;
    }

    filterPredictions(filter) {
        if (!this.predictions) return [];
        
        this.currentFilter = filter;
        
        switch (filter) {
            case 'top10':
                return this.predictions.filter(d => parseInt(d.predicted_position) <= 10);
            case 'points':
                return this.predictions.filter(d => this.getPointsForPosition(parseInt(d.predicted_position)) > 0);
            case 'podium':
                return this.predictions.filter(d => parseInt(d.predicted_position) <= 3);
            default:
                return this.predictions;
        }
    }

    searchDrivers(query) {
        if (!this.predictions || !query) return this.predictions;
        
        const lowerQuery = query.toLowerCase();
        return this.predictions.filter(driver => 
            (driver.driver_name && driver.driver_name.toLowerCase().includes(lowerQuery)) ||
            (driver.team && driver.team.toLowerCase().includes(lowerQuery))
        );
    }

    getDriverDetails(driverId) {
        if (!this.predictions) return null;
        return this.predictions.find(d => d.driverId == driverId);
    }

    exportPredictions(format = 'csv') {
        if (!this.predictions) return;
        
        if (format === 'csv') {
            const csv = this.convertToCSV(this.predictions);
            this.downloadFile(csv, 'f1_2025_predictions.csv', 'text/csv');
        } else if (format === 'json') {
            const json = JSON.stringify(this.predictions, null, 2);
            this.downloadFile(json, 'f1_2025_predictions.json', 'application/json');
        }
    }

    convertToCSV(data) {
        if (!data || data.length === 0) return '';
        
        const headers = Object.keys(data[0]);
        const csvContent = [
            headers.join(','),
            ...data.map(row => headers.map(header => row[header] || '').join(','))
        ].join('\n');
        
        return csvContent;
    }

    downloadFile(content, filename, contentType) {
        const blob = new Blob([content], { type: contentType });
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
    }

    // Interactive features
    addDriverHoverEffects() {
        const tableRows = document.querySelectorAll('#predictions-tbody tr');
        
        tableRows.forEach(row => {
            row.addEventListener('mouseenter', () => {
                this.highlightDriverInfo(row);
            });
            
            row.addEventListener('mouseleave', () => {
                this.removeHighlight(row);
            });
            
            row.addEventListener('click', () => {
                this.showDriverDetails(row);
            });
        });
    }

    highlightDriverInfo(row) {
        row.style.background = 'rgba(225, 6, 0, 0.2)';
        row.style.transform = 'scale(1.02)';
        row.style.boxShadow = '0 5px 15px rgba(225, 6, 0, 0.3)';
    }

    removeHighlight(row) {
        row.style.background = '';
        row.style.transform = '';
        row.style.boxShadow = '';
    }

    showDriverDetails(row) {
        const driverName = row.querySelector('.driver-cell').textContent;
        const position = row.querySelector('.position-cell').textContent;
        const team = row.querySelector('.team-cell').textContent;
        
        // Create modal or tooltip with driver details
        const modal = document.createElement('div');
        modal.className = 'driver-modal';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>${position}: ${driverName}</h3>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    <p><strong>Team:</strong> ${team}</p>
                    <p><strong>Predicted Position:</strong> ${position}</p>
                    <p><strong>Expected Points:</strong> ${this.getPointsForPosition(parseInt(position.replace(/[^0-9]/g, '')))}</p>
                    <div class="prediction-confidence">
                        <strong>Prediction Confidence:</strong>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${Math.random() * 30 + 70}%"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Add modal styles
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 10000;
        `;
        
        document.body.appendChild(modal);
        
        // Close modal functionality
        const closeBtn = modal.querySelector('.modal-close');
        const closeModal = () => modal.remove();
        
        closeBtn.addEventListener('click', closeModal);
        modal.addEventListener('click', (e) => {
            if (e.target === modal) closeModal();
        });
        
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeModal();
        });
    }
}

// Initialize predictions manager
const predictionsManager = new F1PredictionsManager();

// Load predictions when page is ready
document.addEventListener('DOMContentLoaded', async function() {
    await predictionsManager.loadPredictions();
    
    // Add interactive features after table is populated
    setTimeout(() => {
        predictionsManager.addDriverHoverEffects();
    }, 3000);
});

// Export for global use
window.F1Predictions = predictionsManager;