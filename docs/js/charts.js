// F1 Neural Network Predictor - Charts and Visualizations

// Chart configurations and data
let teamChart, driverChart, accuracyChart, championshipChart;

// Initialize charts when predictions data is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Wait for predictions data to load
    setTimeout(() => {
        if (window.F1App && window.F1App.predictionsData()) {
            initializeCharts(window.F1App.predictionsData());
        }
    }, 3000);
});

function initializeCharts(predictionsData) {
    console.log('📊 Initializing charts with predictions data');
    
    if (!predictionsData || predictionsData.length === 0) {
        console.warn('No predictions data available for charts');
        return;
    }
    
    createTeamPerformanceChart(predictionsData);
    createDriverDistributionChart(predictionsData);
    createAccuracyChart();
    createChampionshipChart(predictionsData);
}

function createTeamPerformanceChart(data) {
    const ctx = document.getElementById('team-chart');
    if (!ctx) return;
    
    // Calculate team average positions
    const teamData = {};
    data.forEach(driver => {
        const team = driver.team || `Team ${driver.constructorId}`;
        const position = parseInt(driver.predicted_position);
        
        if (!teamData[team]) {
            teamData[team] = { positions: [], drivers: [] };
        }
        teamData[team].positions.push(position);
        teamData[team].drivers.push(driver.driver_name || `Driver ${driver.driverId}`);
    });
    
    // Calculate averages and sort
    const teamAverages = Object.entries(teamData)
        .map(([team, data]) => ({
            team,
            avgPosition: data.positions.reduce((a, b) => a + b, 0) / data.positions.length,
            driverCount: data.drivers.length
        }))
        .sort((a, b) => a.avgPosition - b.avgPosition);
    
    const labels = teamAverages.map(t => t.team);
    const positions = teamAverages.map(t => t.avgPosition);
    const colors = labels.map(team => window.F1App.getTeamColor(team));
    
    teamChart = new Chart(ctx, {
        type: 'horizontalBar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Average Predicted Position',
                data: positions,
                backgroundColor: colors,
                borderColor: colors,
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Team Performance Ranking',
                    color: '#ffffff',
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    max: 20,
                    reverse: true, // Lower position numbers are better
                    grid: {
                        color: '#333'
                    },
                    ticks: {
                        color: '#cccccc'
                    },
                    title: {
                        display: true,
                        text: 'Average Position (Lower is Better)',
                        color: '#cccccc'
                    }
                },
                y: {
                    grid: {
                        color: '#333'
                    },
                    ticks: {
                        color: '#cccccc'
                    }
                }
            }
        }
    });
}

function createDriverDistributionChart(data) {
    const ctx = document.getElementById('driver-chart');
    if (!ctx) return;
    
    // Create position distribution
    const positionCounts = {};
    for (let i = 1; i <= 20; i++) {
        positionCounts[i] = 0;
    }
    
    data.forEach(driver => {
        const position = parseInt(driver.predicted_position);
        if (position >= 1 && position <= 20) {
            positionCounts[position]++;
        }
    });
    
    const positions = Object.keys(positionCounts);
    const counts = Object.values(positionCounts);
    
    // Color positions differently
    const colors = positions.map(pos => {
        const p = parseInt(pos);
        if (p === 1) return '#FFD700';
        if (p === 2) return '#C0C0C0';
        if (p === 3) return '#CD7F32';
        if (p <= 10) return '#00ff00';
        return '#666666';
    });
    
    driverChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: positions.map(p => `P${p}`),
            datasets: [{
                label: 'Number of Drivers',
                data: counts,
                backgroundColor: colors,
                borderColor: colors,
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Position Distribution',
                    color: '#ffffff',
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: '#333'
                    },
                    ticks: {
                        color: '#cccccc'
                    },
                    title: {
                        display: true,
                        text: 'Predicted Position',
                        color: '#cccccc'
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: '#333'
                    },
                    ticks: {
                        color: '#cccccc'
                    },
                    title: {
                        display: true,
                        text: 'Number of Drivers',
                        color: '#cccccc'
                    }
                }
            }
        }
    });
}

function createAccuracyChart() {
    const ctx = document.getElementById('accuracy-chart');
    if (!ctx) return;
    
    // Model accuracy metrics
    const accuracyData = {
        'Exact Position': 10.0,
        'Within 1 Position': 25.9,
        'Within 2 Positions': 44.3,
        'Within 3 Positions': 58.5,
        'Within 5 Positions': 77.8
    };
    
    const labels = Object.keys(accuracyData);
    const values = Object.values(accuracyData);
    
    accuracyChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: [
                    '#ff4444',
                    '#ff8800',
                    '#ffaa00',
                    '#88ff00',
                    '#00ff00'
                ],
                borderColor: '#333',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#cccccc',
                        padding: 20,
                        usePointStyle: true
                    }
                },
                title: {
                    display: true,
                    text: 'Prediction Accuracy',
                    color: '#ffffff',
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                }
            }
        }
    });
}

function createChampionshipChart(data) {
    const ctx = document.getElementById('championship-chart');
    if (!ctx) return;
    
    // Calculate constructor championship points
    const teamPoints = {};
    
    data.forEach(driver => {
        const team = driver.team || `Team ${driver.constructorId}`;
        const position = parseInt(driver.predicted_position);
        const points = getPointsForPosition(position);
        
        if (!teamPoints[team]) {
            teamPoints[team] = 0;
        }
        teamPoints[team] += points;
    });
    
    // Sort teams by points
    const sortedTeams = Object.entries(teamPoints)
        .sort(([,a], [,b]) => b - a)
        .slice(0, 10); // Top 10 teams
    
    const teamNames = sortedTeams.map(([team]) => team);
    const points = sortedTeams.map(([, points]) => points);
    const colors = teamNames.map(team => window.F1App.getTeamColor(team));
    
    championshipChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: teamNames,
            datasets: [{
                label: 'Predicted Points',
                data: points,
                backgroundColor: colors,
                borderColor: colors,
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Constructor Championship (Single Race)',
                    color: '#ffffff',
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: '#333'
                    },
                    ticks: {
                        color: '#cccccc',
                        maxRotation: 45
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: '#333'
                    },
                    ticks: {
                        color: '#cccccc'
                    },
                    title: {
                        display: true,
                        text: 'Points',
                        color: '#cccccc'
                    }
                }
            }
        }
    });
}

// Update charts when window resizes
window.addEventListener('resize', function() {
    if (teamChart) teamChart.resize();
    if (driverChart) driverChart.resize();
    if (accuracyChart) accuracyChart.resize();
    if (championshipChart) championshipChart.resize();
});

// Export for use in other scripts
window.F1Charts = {
    initializeCharts,
    teamChart: () => teamChart,
    driverChart: () => driverChart,
    accuracyChart: () => accuracyChart,
    championshipChart: () => championshipChart
};