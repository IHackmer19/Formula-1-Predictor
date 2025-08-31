// F1 Neural Network Predictor - Main App JavaScript

// Global variables
let predictionsData = null;
let currentDataSource = 'sample';

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    console.log('🏎️ Initializing F1 Neural Network Predictor');
    
    // Show loading screen
    showLoadingScreen();
    
    // Initialize components
    setTimeout(() => {
        setupNavigation();
        loadPredictionsData();
        setupEventListeners();
        hideLoadingScreen();
        animateOnScroll();
    }, 2000);
}

function showLoadingScreen() {
    const loadingScreen = document.getElementById('loading-screen');
    if (loadingScreen) {
        loadingScreen.style.display = 'flex';
    }
}

function hideLoadingScreen() {
    const loadingScreen = document.getElementById('loading-screen');
    if (loadingScreen) {
        loadingScreen.style.opacity = '0';
        setTimeout(() => {
            loadingScreen.style.display = 'none';
        }, 500);
    }
}

function setupNavigation() {
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');
    const navLinks = document.querySelectorAll('.nav-link');

    // Mobile menu toggle
    if (hamburger && navMenu) {
        hamburger.addEventListener('click', () => {
            hamburger.classList.toggle('active');
            navMenu.classList.toggle('active');
        });
    }

    // Smooth scrolling for navigation links
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href').substring(1);
            scrollToSection(targetId);
            
            // Close mobile menu
            if (navMenu) {
                navMenu.classList.remove('active');
            }
            if (hamburger) {
                hamburger.classList.remove('active');
            }
        });
    });

    // Update active nav link on scroll
    window.addEventListener('scroll', updateActiveNavLink);
}

function scrollToSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        const offsetTop = section.offsetTop - 80; // Account for fixed navbar
        window.scrollTo({
            top: offsetTop,
            behavior: 'smooth'
        });
    }
}

function updateActiveNavLink() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-link');
    
    let currentSection = '';
    const scrollPos = window.scrollY + 100;

    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.offsetHeight;
        
        if (scrollPos >= sectionTop && scrollPos < sectionTop + sectionHeight) {
            currentSection = section.getAttribute('id');
        }
    });

    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${currentSection}`) {
            link.classList.add('active');
        }
    });
}

function loadPredictionsData() {
    console.log('📊 Loading predictions data...');
    
    // Try to load real predictions first, then fall back to sample
    loadCSVData('../f1_2025_predictions_kaggle.csv')
        .then(data => {
            predictionsData = data;
            currentDataSource = 'kaggle';
            console.log('✅ Loaded real Kaggle predictions');
            populatePredictions(data);
        })
        .catch(() => {
            // Fall back to sample data
            loadCSVData('../f1_2025_predictions_sample.csv')
                .then(data => {
                    predictionsData = data;
                    currentDataSource = 'sample';
                    console.log('✅ Loaded sample predictions');
                    populatePredictions(data);
                })
                .catch(() => {
                    // Fall back to default sample data
                    loadCSVData('../f1_2025_predictions.csv')
                        .then(data => {
                            predictionsData = data;
                            currentDataSource = 'sample';
                            console.log('✅ Loaded default predictions');
                            populatePredictions(data);
                        })
                        .catch(error => {
                            console.error('❌ Failed to load any predictions data:', error);
                            showErrorMessage('Failed to load predictions data');
                        });
                });
        });
}

function loadCSVData(url) {
    return fetch(url)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.text();
        })
        .then(csvText => {
            return parseCSV(csvText);
        });
}

function parseCSV(csvText) {
    const lines = csvText.trim().split('\n');
    const headers = lines[0].split(',');
    const data = [];
    
    for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',');
        const row = {};
        headers.forEach((header, index) => {
            row[header.trim()] = values[index] ? values[index].trim() : '';
        });
        if (row.predicted_position) {
            data.push(row);
        }
    }
    
    return data;
}

function populatePredictions(data) {
    if (!data || data.length === 0) {
        showErrorMessage('No predictions data available');
        return;
    }

    console.log(`📊 Populating predictions with ${data.length} drivers`);
    
    // Update podium
    updatePodium(data);
    
    // Update results table
    updateResultsTable(data);
    
    // Update data source indicator
    updateDataSourceIndicator();
    
    // Initialize charts
    initializeCharts(data);
}

function updatePodium(data) {
    // Sort by predicted position
    const sortedData = data.sort((a, b) => parseInt(a.predicted_position) - parseInt(b.predicted_position));
    
    // Update podium positions
    for (let i = 0; i < 3 && i < sortedData.length; i++) {
        const driver = sortedData[i];
        const driverElement = document.getElementById(`p${i+1}-driver`);
        const teamElement = document.getElementById(`p${i+1}-team`);
        
        if (driverElement && teamElement) {
            driverElement.textContent = driver.driver_name || `Driver ${driver.driverId}`;
            teamElement.textContent = driver.team || `Team ${driver.constructorId}`;
        }
    }
}

function updateResultsTable(data) {
    const tbody = document.getElementById('predictions-tbody');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    // Sort by predicted position
    const sortedData = data.sort((a, b) => parseInt(a.predicted_position) - parseInt(b.predicted_position));
    
    sortedData.forEach(driver => {
        const row = createResultsRow(driver);
        tbody.appendChild(row);
    });
    
    // Setup table filtering
    setupTableFiltering();
}

function createResultsRow(driver) {
    const row = document.createElement('tr');
    const position = parseInt(driver.predicted_position);
    
    // Position cell
    const positionCell = document.createElement('td');
    positionCell.className = 'position-cell';
    if (position <= 3) positionCell.classList.add(`position-${position}`);
    if (position <= 10) positionCell.classList.add('position-points');
    
    let positionIcon = '';
    if (position === 1) positionIcon = '🥇 ';
    else if (position === 2) positionIcon = '🥈 ';
    else if (position === 3) positionIcon = '🥉 ';
    else if (position <= 10) positionIcon = '🏁 ';
    
    positionCell.innerHTML = `${positionIcon}P${position}`;
    
    // Driver cell
    const driverCell = document.createElement('td');
    driverCell.className = 'driver-cell';
    driverCell.textContent = driver.driver_name || `Driver ${driver.driverId}`;
    
    // Team cell
    const teamCell = document.createElement('td');
    teamCell.className = 'team-cell';
    teamCell.textContent = driver.team || `Team ${driver.constructorId}`;
    
    // Confidence cell
    const confidenceCell = document.createElement('td');
    const confidence = parseFloat(driver.prediction_confidence) || Math.random() * 0.3 + 0.7;
    confidenceCell.innerHTML = `
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: ${confidence * 100}%"></div>
        </div>
        <span style="font-size: 0.8rem; color: var(--text-gray);">${(confidence * 100).toFixed(1)}%</span>
    `;
    
    // Points cell
    const pointsCell = document.createElement('td');
    const points = getPointsForPosition(position);
    pointsCell.textContent = points;
    if (points > 0) pointsCell.style.color = 'var(--success-color)';
    
    row.appendChild(positionCell);
    row.appendChild(driverCell);
    row.appendChild(teamCell);
    row.appendChild(confidenceCell);
    row.appendChild(pointsCell);
    
    // Add data attributes for filtering
    row.setAttribute('data-position', position);
    row.setAttribute('data-points', points > 0 ? 'yes' : 'no');
    
    return row;
}

function getPointsForPosition(position) {
    const pointsMap = {
        1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
        6: 8, 7: 6, 8: 4, 9: 2, 10: 1
    };
    return pointsMap[position] || 0;
}

function setupTableFiltering() {
    const filterButtons = document.querySelectorAll('.filter-btn');
    const tableRows = document.querySelectorAll('#predictions-tbody tr');
    
    filterButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            // Update active button
            filterButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            const filter = btn.getAttribute('data-filter');
            
            // Filter table rows
            tableRows.forEach(row => {
                const position = parseInt(row.getAttribute('data-position'));
                const hasPoints = row.getAttribute('data-points') === 'yes';
                
                let show = true;
                
                if (filter === 'top10') {
                    show = position <= 10;
                } else if (filter === 'points') {
                    show = hasPoints;
                }
                
                row.style.display = show ? '' : 'none';
            });
        });
    });
}

function updateDataSourceIndicator() {
    // Add data source indicator to the page
    const indicator = document.createElement('div');
    indicator.className = 'data-source-indicator';
    indicator.innerHTML = `
        <div class="indicator-content">
            <span class="indicator-icon">${currentDataSource === 'kaggle' ? '🔥' : '📊'}</span>
            <span class="indicator-text">
                ${currentDataSource === 'kaggle' ? 'Real Kaggle Data' : 'Sample Data'}
            </span>
        </div>
    `;
    
    // Add styles
    indicator.style.cssText = `
        position: fixed;
        top: 80px;
        right: 20px;
        background: ${currentDataSource === 'kaggle' ? 'var(--success-color)' : 'var(--warning-color)'};
        color: var(--dark-bg);
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 0.8rem;
        z-index: 999;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    `;
    
    document.body.appendChild(indicator);
}

function setupEventListeners() {
    // Smooth scrolling for buttons
    const scrollButtons = document.querySelectorAll('[onclick*="scrollToSection"]');
    scrollButtons.forEach(button => {
        button.addEventListener('click', (e) => {
            e.preventDefault();
            const onclick = button.getAttribute('onclick');
            const sectionId = onclick.match(/scrollToSection\('(.+)'\)/)[1];
            scrollToSection(sectionId);
        });
    });
    
    // Add scroll animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    }, observerOptions);
    
    // Observe sections for animations
    document.querySelectorAll('section').forEach(section => {
        observer.observe(section);
    });
}

function animateOnScroll() {
    window.addEventListener('scroll', () => {
        const scrolled = window.pageYOffset;
        const parallax = document.querySelector('.hero-visual');
        
        if (parallax) {
            const speed = scrolled * 0.5;
            parallax.style.transform = `translateY(${speed}px)`;
        }
    });
}

function showErrorMessage(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.innerHTML = `
        <div class="error-content">
            <i class="fas fa-exclamation-triangle"></i>
            <span>${message}</span>
        </div>
    `;
    
    errorDiv.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: var(--error-color);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        z-index: 10000;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    `;
    
    document.body.appendChild(errorDiv);
    
    // Remove after 5 seconds
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
}

// Utility functions
function formatDriverName(name) {
    if (!name) return 'Unknown Driver';
    return name.split(' ').map(word => 
        word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
    ).join(' ');
}

function formatTeamName(team) {
    if (!team) return 'Unknown Team';
    return team.replace(/_/g, ' ');
}

function getTeamColor(teamName) {
    const teamColors = {
        'Mercedes': '#00D2BE',
        'Red Bull Racing': '#0600EF',
        'Ferrari': '#DC143C',
        'McLaren': '#FF8700',
        'Alpine': '#0090FF',
        'Aston Martin': '#006F62',
        'AlphaTauri': '#2B4562',
        'Alfa Romeo': '#900000',
        'Haas': '#FFFFFF',
        'Williams': '#005AFF'
    };
    
    return teamColors[teamName] || '#666666';
}

// Export functions for use in other scripts
window.F1App = {
    scrollToSection,
    formatDriverName,
    formatTeamName,
    getTeamColor,
    predictionsData: () => predictionsData,
    currentDataSource: () => currentDataSource
};