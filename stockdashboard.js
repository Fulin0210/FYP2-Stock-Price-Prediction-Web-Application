// Global variables
let currentTimeframe = '1y';
let stockChart = null;
let candlestickChart = null;
let currentChartType = 'line';

// Initialize Select2 for searchable dropdowns
$(document).ready(function() {
    $('#search-bar').select2();
    loadSectors();
    initializeTimeframeButtons();
    initializeChartToggle();
    initializeViewToggle();
});

document.getElementById('search-button').addEventListener('click', function() {
    const selectedSector = document.getElementById('sector-dropdown').value;
    const selectedStock = document.getElementById('search-bar').value;

    if (selectedSector) {
        loadSectorPerformance(selectedSector);
        updateStockInfo(selectedStock);
    } else if (selectedStock) {
        updateStockInfo(selectedStock);
    }
});

// Initialize timeframe buttons
function initializeTimeframeButtons() {
    const buttons = document.querySelectorAll('.timeframe-btn');
    buttons.forEach(button => {
        button.addEventListener('click', function() {
            buttons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');
            currentTimeframe = this.dataset.timeframe;
            
            // Refresh both sector performance and stock data
            const selectedSector = document.getElementById('sector-dropdown').value;
            const selectedStock = document.getElementById('search-bar').value;
            
            if (selectedSector) {
                loadSectorPerformance(selectedSector);
            }
            if (selectedStock) {
                updateStockInfo(selectedStock);
            }
        });
    });
    // Set default active timeframe
    document.querySelector('[data-timeframe="1y"]').classList.add('active');
}

// Initialize chart toggle
function initializeChartToggle() {
    const buttons = document.querySelectorAll('.chart-toggle button');
    buttons.forEach(button => {
        button.addEventListener('click', function() {
            buttons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');
            toggleChartType(this.dataset.chart);
        });
    });
}

// Toggle between chart types
function toggleChartType(type) {
    currentChartType = type;
    const stockChartCanvas = document.getElementById('stockChart');
    const candlestickChartDiv = document.getElementById('candlestickChart');
    
    if (type === 'line') {
        stockChartCanvas.style.display = 'block';
        candlestickChartDiv.style.display = 'none';
    } else {
        stockChartCanvas.style.display = 'none';
        candlestickChartDiv.style.display = 'block';
    }
    
    const selectedStock = document.getElementById('search-bar').value;
    if (selectedStock) {
        updateStockInfo(selectedStock);
    }
}

// Load sectors into dropdown
function loadSectors() {
    fetch('/get_sectors')
        .then(response => response.json())
        .then(sectors => {
            const sectorDropdown = document.getElementById('sector-dropdown');
            sectors.forEach(sector => {
                const option = document.createElement('option');
                option.value = sector;
                option.textContent = sector;
                sectorDropdown.appendChild(option);
            });
        })
        .catch(error => console.error('Error loading sectors:', error));
}

// Load stocks when sector is selected
document.getElementById('sector-dropdown').addEventListener('change', function() {
    const sector = this.value;
    if (sector) {
        // Load sector performance data
        loadSectorPerformance(sector);
        
        // Load stocks for the selected sector
        fetch(`/get_stocks/${sector}`)
            .then(response => response.json())
            .then(stocks => {
                const stockDropdown = $('#search-bar');
                stockDropdown.empty();
                stockDropdown.append('<option disabled selected>Select Stock</option>');
                
                stocks.forEach(stock => {
                    stockDropdown.append(new Option(
                        `${stock['Ticker Name']} (${stock['Symbol']})`,
                        stock['Symbol']
                    ));
                });
            })
            .catch(error => console.error('Error loading stocks:', error));
    }
});

// Initialize view toggle
function initializeViewToggle() {
    const buttons = document.querySelectorAll('.view-toggle button');
    buttons.forEach(button => {
        button.addEventListener('click', function() {
            buttons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');
            togglePerformanceView(this.dataset.view);
        });
    });
}

// Toggle between performance views
function togglePerformanceView(view) {
    const cardView = document.getElementById('stocksCardView');
    const chartView = document.getElementById('stocksChartView');
    
    if (view === 'cards') {
        cardView.style.display = 'flex';
        chartView.style.display = 'none';
    } else {
        cardView.style.display = 'none';
        chartView.style.display = 'block';
        if (window.topPerformersChart) {
            window.topPerformersChart.update();
        }
    }
}

// Load sector performance data
function loadSectorPerformance(sector) {
    // Show loading indicator
    document.getElementById('loading-indicator').style.display = 'block';

    // Get selected stock
    const selectedStock = document.getElementById('search-bar').value;
    
    // Fetch data including selected_stock parameter
    fetch(`/get_sector_performance?sector=${encodeURIComponent(sector)}&timeframe=${currentTimeframe}&selected_stock=${encodeURIComponent(selectedStock)}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Get selected stock data from the returned data list
                let found = null;
                if (selectedStock) {
                    found = data.data.find(stock => stock.symbol === selectedStock);
                }

                // Update market cap (main card) with selected stock's market cap
                let mainMarketCap = '-';
                if (selectedStock) {
                    if (found && typeof found.market_cap !== 'undefined' && found.market_cap !== null) {
                        mainMarketCap = 'RM' + formatMarketCap(found.market_cap);
                    }
                }
                // Assuming 'total-market-cap' is where the main card market cap is displayed
                const mainMarketCapElement = document.getElementById('total-market-cap');
                if(mainMarketCapElement) { // Check if element exists
                    mainMarketCapElement.textContent = mainMarketCap;
                }

                // Update previous close (replace trading volume)
                let prevClose = '-';
                if (found && typeof found.previous_close !== 'undefined' && found.previous_close !== null) {
                    prevClose = 'RM' + Number(found.previous_close).toFixed(2);
                }
                const prevCloseElement = document.getElementById('previous-close');
                if (prevCloseElement) { // Check if the element exists
                    prevCloseElement.textContent = prevClose;
                }

                // Update timeframe label
                document.querySelector('.timeframe-label').textContent = `(${currentTimeframe.toUpperCase()})`;

                // Update cards view
                updatePerformanceCards(data.data);
                                
                // Update performance bars
                updatePerformanceBars(data.data);

                // Update main metric cards with selected stock's data
                if(selectedStock && found) {
                     // Assuming elements with these IDs exist for the main metrics
                    const changePercentElement = document.getElementById('change-percentage');
                    const tradingVolumeElement = document.getElementById('total-trading-volume'); // Assuming this ID for volume

                    if(changePercentElement) {
                        let calculatedChangePercent = 0;
                        if (found.start_price !== null && typeof found.start_price !== 'undefined' && found.start_price > 0) {
                            calculatedChangePercent = ((found.end_price - found.start_price) / found.start_price) * 100;
                        }
                        changePercentElement.textContent = `${calculatedChangePercent >= 0 ? '+' : ''}${calculatedChangePercent.toFixed(2)}%`;
                        changePercentElement.className = calculatedChangePercent >= 0 ? 'metric-value positive' : 'metric-value negative';
                    }

                    if(tradingVolumeElement) {
                         // The backend provides avg_volume for the stock, which might be suitable
                         tradingVolumeElement.textContent = found.avg_volume ? found.avg_volume.toLocaleString() : '-';
                    }
                     // Market cap is already updated above using total-market-cap ID

                } else { // If no stock is selected or found, clear main metrics
                     const mainMetricElements = ['total-market-cap', 'previous-close', 'change-percentage', 'total-trading-volume'];
                     mainMetricElements.forEach(id => {
                         const el = document.getElementById(id);
                         if(el) el.textContent = '-';
                     });
                }

            } else {
                console.error('Error loading sector performance:', data.error);
                 // Clear all performance data and show error
                document.getElementById('stocksCardView').innerHTML = '<p>' + data.error + '</p>';
                document.getElementById('stocksChartView').style.display = 'none';
                 // Also clear main metric cards
                 const mainMetricElements = ['total-market-cap', 'previous-close', 'change-percentage', 'total-trading-volume'];
                     mainMetricElements.forEach(id => {
                         const el = document.getElementById(id);
                         if(el) el.textContent = '-';
                     });
            }
        })
        .catch(error => {
            console.error('Error loading sector performance:', error);
            // Show a generic error message on the frontend
            document.getElementById('stocksCardView').innerHTML = '<p>Error loading sector performance data.</p>';
            document.getElementById('stocksChartView').style.display = 'none';
             // Also clear main metric cards
                 const mainMetricElements = ['total-market-cap', 'previous-close', 'change-percentage', 'total-trading-volume'];
                     mainMetricElements.forEach(id => {
                         const el = document.getElementById(id);
                         if(el) el.textContent = '-';
                     });
        })
        .finally(() => {
            document.getElementById('loading-indicator').style.display = 'none';
        });
}

// Update performance cards
function updatePerformanceCards(stocks) {
    const stocksList = document.getElementById('stocksCardView');
    stocksList.innerHTML = '';

    if (!stocks || stocks.length === 0) {
        stocksList.innerHTML = '<p>No top performers found for this sector and timeframe.</p>';
        return;
    }

    // Recalculate change percentage for consistent display
    const processedStocks = stocks.map(stock => {
        let calculatedChangePercent = 0;
        if (stock.start_price !== null && typeof stock.start_price !== 'undefined' && stock.start_price > 0) {
             calculatedChangePercent = ((stock.end_price - stock.start_price) / stock.start_price) * 100;
        }
        return {
            ...stock,
            calculatedChangePercent: calculatedChangePercent
        };
    });

    // Sort by the newly calculated change percentage for consistency (optional, but good practice)
    processedStocks.sort((a, b) => b.calculatedChangePercent - a.calculatedChangePercent);

    processedStocks.forEach((stock, index) => {
        const stockCard = document.createElement('div');
        stockCard.className = 'stock-card';

        // Format metrics using the calculated change percentage
        const formattedChange = `${stock.calculatedChangePercent >= 0 ? '+' : ''}${stock.calculatedChangePercent.toFixed(2)}%`;
        const formattedMarketCap = stock.market_cap !== null && typeof stock.market_cap !== 'undefined' ? `RM${formatMarketCap(stock.market_cap)}` : '-'; // Using existing formatMarketCap function

        stockCard.innerHTML = `
            <div class="rank-badge">${index + 1}</div>
            <div class="stock-info">
                <div class="stock-name">${stock.symbol}</div>
                <div class="stock-details">${stock.name}</div>
            </div>
            <div class="stock-metrics">
                <div class="metric">
                    <div class="metric-label">Change %</div>
                    <div class="metric-value ${stock.calculatedChangePercent >= 0 ? 'positive' : 'negative'}">
                        ${formattedChange}
                    </div>
                </div>
                <div class="metric">
                    <div class="metric-label">Market Cap</div>
                    <div class="metric-value">${formattedMarketCap}</div>
                </div>
            </div>
        `;
        stocksList.appendChild(stockCard);
    });
}

// Update performance bars
function updatePerformanceBars(stocks) {
    const barsContainer = document.querySelector('.performance-bars');
    barsContainer.innerHTML = '';

    if (!stocks || stocks.length === 0) {
        barsContainer.innerHTML = '<p>No performance data to display.</p>';
        return;
    }

    // Recalculate change percentage for consistent display
    const processedStocks = stocks.map(stock => {
        let calculatedChangePercent = 0;
         if (stock.start_price !== null && typeof stock.start_price !== 'undefined' && stock.start_price > 0) {
             calculatedChangePercent = ((stock.end_price - stock.start_price) / stock.start_price) * 100;
        }
        return {
            ...stock,
            calculatedChangePercent: calculatedChangePercent
        };
    });

    // Sort by the newly calculated change percentage for consistency (optional)
     processedStocks.sort((a, b) => b.calculatedChangePercent - a.calculatedChangePercent);


    // Find the max calculated change for scaling the bars
    const maxChange = Math.max(...processedStocks.map(stock => Math.abs(stock.calculatedChangePercent)));
    
    if (maxChange === 0) { // Avoid division by zero if all changes are 0
         barsContainer.innerHTML = '<p>No significant price change in this period.</p>';
         return;
    }

    processedStocks.forEach((stock, index) => {
        const barWidth = (Math.abs(stock.calculatedChangePercent) / maxChange) * 100;
        const bar = document.createElement('div');
        bar.className = 'performance-bar';

        const formattedChange = `${stock.calculatedChangePercent >= 0 ? '+' : ''}${stock.calculatedChangePercent.toFixed(2)}%`;

        bar.innerHTML = `
            <div class="bar-label">${stock.symbol}</div>
            <div class="bar-container">
                <div class="bar-fill" style="width: ${barWidth}%; 
                    background: ${stock.calculatedChangePercent >= 0 ? 'var(--success-color)' : 'var(--danger-color)'}">
                </div>
            </div>
            <div class="bar-value ${stock.calculatedChangePercent >= 0 ? 'positive' : 'negative'}">
                ${formattedChange}
            </div>
        `;
        barsContainer.appendChild(bar);
    });
}

// Format large numbers
function formatNumber(num) {
    if (num >= 1000000000) {
        return (num / 1000000000).toFixed(1) + 'B';
    }
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    }
    if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toLocaleString();
}

// Format market cap
function formatMarketCap(marketCap) {
    if (marketCap >= 1000) {
        return (marketCap / 1000).toFixed(2) + 'T';
    }
    if (marketCap >= 1) {
        return marketCap.toFixed(2) + 'B';
    }
    return marketCap.toFixed(2) + 'M';
}

// Update timeframe
function setTimeframe(timeframe) {
    currentTimeframe = timeframe;
    const selectedStock = document.getElementById('search-bar').value;
    if (selectedStock) {
        updateStockInfo(selectedStock);
    }
}

// Calculate YTD date
function getYTDDate() {
    const now = new Date();
    return new Date(now.getFullYear(), 0, 1);
}

function filterDataByTimeframe(data, timeframe) {
    const now = new Date(); 
    let startDate;

    switch (timeframe) {
        case '1w':
            startDate = new Date(now);
            startDate.setDate(startDate.getDate() - 7);
            break;
        case '6m':
            startDate = new Date(now);
            startDate.setMonth(startDate.getMonth() - 6);
            break;
        case '1m':
            startDate = new Date(now);
            startDate.setMonth(startDate.getMonth() - 1);
            break;
        case '1y':
            startDate = new Date(now);
            startDate.setFullYear(startDate.getFullYear() - 1);
            break;
        case '5y':
            startDate = new Date(now);
            startDate.setFullYear(startDate.getFullYear() - 5);
            break;
        case '10y':
            startDate = new Date(now);
            startDate.setFullYear(startDate.getFullYear() - 10);
            break;
        case 'ytd':
            startDate = getYTDDate();
            startDate.setDate(startDate.getDate() + 1);
            break;
        default:
            startDate = new Date(now);
            startDate.setFullYear(startDate.getFullYear() - 1);
    }
    startDate.setDate(startDate.getDate() - 1);

    return data.filter(item => new Date(item.Date) >= startDate);
}

function updateStockInfo(ticker) {
    document.getElementById('loading-indicator').style.display = 'block';
    document.getElementById('error-message').style.display = 'none';

    fetch(`/get_stock_data?ticker=${ticker}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }

            // Update market cap and previous close
            if (data.market_cap !== undefined) {
                document.getElementById('market-cap').textContent = `${data.market_cap}B`;
            }
            if (data.previous_close !== undefined) {
                document.getElementById('previous-close').textContent = data.previous_close.toFixed(2);
            }

            // Update current price and percentage change if desired
            if (data.current_price !== undefined) {
                document.getElementById('current-price').textContent = data.current_price.toFixed(2);
            }
            if (data.change_percent !== undefined) {
                const changeEl = document.getElementById('price-change');
                changeEl.textContent = `${data.change_percent.toFixed(2)}%`;
                changeEl.className = data.change_percent >= 0 ? 'text-green' : 'text-red';
            }

            // Update line chart
            const stockData = filterDataByTimeframe(data.data, currentTimeframe);
            if (stockData.length > 0) {
                updateCharts(stockData);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('error-message').style.display = 'block';
            document.getElementById('error-message').querySelector('span').textContent = 'Error loading data: ' + error.message;
        })
        .finally(() => {
            document.getElementById('loading-indicator').style.display = 'none';
        });
}


// Update charts
function updateCharts(stockData) {
    if (currentChartType === 'line') {
        updateLineChart(stockData);
    } else {
        updateCandlestickChart(stockData);
    }
}

function updateLineChart(stockData) {
    const ctx = document.getElementById('stockChart').getContext('2d');
    if (stockChart) {
        stockChart.destroy();
    }

    const priceData = stockData.map(d => Number(d.Close));
    const dates = stockData.map(d => {
        const dateObj = new Date(d.Date);
        return dateObj.toLocaleDateString('en-CA');
    });

    const startPrice = priceData[0];
    const endPrice = priceData[priceData.length - 1];

    let changePercent = 0;
    if (startPrice > 0) {
        changePercent = ((endPrice - startPrice) / startPrice * 100);
    }

    const isPositive = changePercent >= 0;
    const changePercentStr = `${isPositive ? '+' : ''}${changePercent.toFixed(2)}%`;

    stockChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [{
                label: `Price (${changePercentStr})`,
                data: priceData,
                borderColor: isPositive ? 'rgb(40, 167, 69)' : 'rgb(220, 53, 69)',
                backgroundColor: isPositive ? 'rgba(40, 167, 69, 0.1)' : 'rgba(220, 53, 69, 0.1)',
                fill: true,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Stock Price History'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            return `Price: RM${context.raw.toFixed(2)}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: {
                        callback: function(value) {
                            return 'RM' + value.toFixed(2);
                        }
                    }
                }
            },
            interaction: {
                intersect: false,
                mode: 'index'
            }
        }
    });
}

// Update candlestick chart
function updateCandlestickChart(stockData) {
    const candlestickData = stockData.map(d => ({
        x: new Date(d.Date),
        open: d.Open,
        high: d.High,
        low: d.Low,
        close: d.Close
    }));

    const trace = {
        x: candlestickData.map(d => d.x),
        open: candlestickData.map(d => d.open),
        high: candlestickData.map(d => d.high),
        low: candlestickData.map(d => d.low),
        close: candlestickData.map(d => d.close),
        type: 'candlestick',
        xaxis: 'x',
        yaxis: 'y',
        increasing: {line: {color: '#28a745'}},
        decreasing: {line: {color: '#dc3545'}}
    };

    const layout = {
        title: 'Candlestick Chart',
        yaxis: {
            title: 'Price (RM)',
            autorange: true,
            tickformat: 'RM.2f'
        },
        xaxis: {
            title: 'Date',
            rangeslider: {visible: false}
        },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)'
    };

    Plotly.newPlot('candlestickChart', [trace], layout);
}