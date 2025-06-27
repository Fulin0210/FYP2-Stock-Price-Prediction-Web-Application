document.addEventListener("DOMContentLoaded", () => {
    loadSectors();
    updateSectors();
    evaluateModel();
});

function loadSectors() {
    fetch('/get_sectors')
        .then(response => response.json())
        .then(sectors => {
            const sectorDropdown = document.getElementById("sector-dropdown");
            sectorDropdown.innerHTML = '<option value="">Select Sector</option>';
            sectors.forEach(sector => {
                const option = document.createElement("option");
                option.value = sector;
                option.textContent = sector;
                sectorDropdown.appendChild(option);
            });
        })
        .catch(error => console.error('Error loading sectors:', error));
}

function updateSectors() {
    const sectorDropdown = document.getElementById("sector-dropdown");
    const stockDropdown = document.getElementById("stock-dropdown");
    stockDropdown.disabled = true;

    sectorDropdown.addEventListener("change", () => {
        const selectedSector = sectorDropdown.value;
        stockDropdown.innerHTML = '<option disabled selected>Select A Stock</option>';
        document.getElementById("stock-error").style.display = "none";

        if (selectedSector) {
            stockDropdown.disabled = false;
            fetchStocks(selectedSector);
        } else {
            stockDropdown.disabled = true;
        }
    });
}

function fetchStocks(sector) {
    fetch(`/get_stocks/${sector}`)
        .then(response => response.json())
        .then(stocks => {
            const stockDropdown = document.getElementById("stock-dropdown");
            stockDropdown.innerHTML = '<option disabled selected>Select A Stock</option>';

            if (stocks.length === 0) {
                document.getElementById("stock-error").textContent = "No stocks available for this sector.";
                document.getElementById("stock-error").style.display = "inline";
                return;
            }

            stocks.forEach(stock => {
                const option = document.createElement("option");
                option.value = stock.Symbol;
                option.textContent = `${stock['Ticker Name']} (${stock.Symbol})`;
                stockDropdown.appendChild(option);
            });
        })
        .catch(error => {
            console.error("Error loading stocks:", error);
            alert('Error loading stock data.');
        });
}

function hideErrorMessages() {
    ["sector-error", "stock-error", "model-error", "date-error", "timeframe-error"].forEach(id => {
        document.getElementById(id).style.display = "none";
    });
}

function validateFormInputs(sector, stock, model, date, timeframe) {
    let hasError = false;
    if (!sector) showError("sector-error", "Please select a sector."), hasError = true;
    if (!stock) showError("stock-error", "Please select a stock."), hasError = true;
    if (!model) showError("model-error", "Please select a machine learning model."), hasError = true;
    if (!date) showError("date-error", "Please select a prediction start date."), hasError = true;
    if (!timeframe) showError("timeframe-error", "Please select a timeframe."), hasError = true;
    return hasError;
}

function showError(id, message) {
    const el = document.getElementById(id);
    if (el) {
        el.textContent = message;
        el.style.display = "inline";
    } else {
        console.error(`Error element with id ${id} not found`);
    }
}

function clearGraphs() {
    // Clear the container div but recreate canvas
    const container = document.getElementById('future-price-chart-container');
    if (container) {
        container.innerHTML = `
            <p class="no-data-message">
                <i class="fas fa-chart-line"></i>
                Select parameters and run prediction to see results
                <small>The graph will show historical data and future predictions</small>
            </p>
            <canvas id="futurePriceChart"></canvas>
        `;
    }

    if (maeChart) maeChart.destroy();
    if (rmseChart) rmseChart.destroy();
    if (r2Chart) r2Chart.destroy();
}


function evaluateModel() {
    document.getElementById("predictionForm").addEventListener("submit", async (event) => {
        event.preventDefault();

        const sector = document.getElementById("sector-dropdown").value;
        const stock = document.getElementById("stock-dropdown").value;
        const model = document.getElementById("model").value;
        const date = document.getElementById("prediction-date").value;
        const timeframe = document.getElementById("timeframe").value;

        hideErrorMessages();
        if (validateFormInputs(sector, stock, model, date, timeframe)) return;

        const requestData = { sector, stock_symbol: stock, model, prediction_date: date, timeframe };

        clearGraphs();
        showLoadingState(true);

        try {
            const [evalRes, predRes] = await Promise.all([
                fetch('/evaluate_model', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(requestData)
                }),
                fetch('/predict_future_price', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(requestData)
                })
            ]);

            const evalData = await evalRes.json();
            const predData = await predRes.json();

            if (!evalRes.ok) throw new Error(evalData.error || `Model evaluation failed: ${evalRes.statusText}`);
            if (!predRes.ok) throw new Error(predData.error || `Price prediction failed: ${predRes.statusText}`);

            validateResponseData(evalData, predData);
            handleModelEvaluationResponse(evalData);
            handleFuturePricePredictionResponse(predData);

        } catch (error) {
            console.error('Error during prediction:', error);
            showError("prediction-error", error.message || "An error occurred during prediction");
        } finally {
            showLoadingState(false);
        }
    });
}

function validateResponseData(evalData, predData) {
    if (!evalData || typeof evalData.mae !== 'number' || typeof evalData.rmse !== 'number' || typeof evalData.r2 !== 'number') {
        throw new Error('Invalid evaluation metrics received');
    }
    if (!predData || !Array.isArray(predData.predictions) || !Array.isArray(predData.dates) || predData.predictions.length === 0) {
        throw new Error('Invalid prediction data received');
    }
}

function showLoadingState(isLoading) {
    const btn = document.querySelector('.submit-btn');
    const loader = document.getElementById('loading-indicator');
    btn.disabled = isLoading;
    btn.innerHTML = isLoading ? '<span class="spinner"></span> Processing...' : 'Run Prediction';
    if (loader) loader.style.display = isLoading ? 'block' : 'none';
}

let maeChart, rmseChart, r2Chart;

function handleModelEvaluationResponse(data) {
    const { mae, rmse, r2, dates, predictions } = data;
    document.getElementById("mae-value").textContent = mae.toFixed(4);
    document.getElementById("rmse-value").textContent = rmse.toFixed(4);
    document.getElementById("r2-value").textContent = r2.toFixed(4);

    if (maeChart) maeChart.destroy();
    if (rmseChart) rmseChart.destroy();
    if (r2Chart) r2Chart.destroy();

    maeChart = createSpeedometerGauge('mae-chart', Math.min(mae * 2, 100), { colorStart: '#4ecdc4', colorEnd: '#45b7af', label: 'MAE', inverse: true });
    rmseChart = createSpeedometerGauge('rmse-chart', Math.min(rmse * 2, 100), { colorStart: '#ff6b6b', colorEnd: '#ee5253', label: 'RMSE', inverse: true });
    r2Chart = createSpeedometerGauge('r2-chart', r2 * 100, { colorStart: '#54a0ff', colorEnd: '#2e86de', label: 'R²', inverse: false });
}

function createSpeedometerGauge(id, value, options) {
    const ctx = document.getElementById(id).getContext('2d');
    const gradient = ctx.createLinearGradient(0, 0, 0, 170);
    gradient.addColorStop(0, options.colorStart);
    gradient.addColorStop(1, options.colorEnd);

    const total = 50;
    const active = Math.round((value / 100) * total);
    const data = Array(total).fill(1);
    const colors = data.map((_, i) => (options.inverse ? (i >= total - active) : (i < active)) ? gradient : 'rgba(236, 240, 241, 0.6)');

    return new Chart(ctx, {
        type: 'doughnut',
        data: { datasets: [{ data, backgroundColor: colors, borderWidth: 0, spacing: 0.03 }] },
        options: {
            circumference: 180,
            rotation: -90,
            cutout: '75%',
            plugins: { tooltip: { enabled: false }, legend: { display: false } },
            layout: { padding: { top: 20 } },
            animation: { duration: 1500, easing: 'easeInOutQuart' }
        }
    });
}

let retryCount = 0;
const maxRetries = 5;

function handleFuturePricePredictionResponse(response) {
    const container = document.getElementById('futurePriceChart');

    if (!container) {
        console.error("Graph container not found at prediction render time");

        if (retryCount < maxRetries) {
            retryCount++;
            setTimeout(() => handleFuturePricePredictionResponse(response), 200);
        } else {
            console.error("Max retries reached. Could not find graph container.");
        }
        return;
    }
    retryCount = 0;  // reset retry count

    // Remove or hide the <p> message above the canvas
    const parent = container.parentElement;
    if (parent) {
        const messageParagraph = parent.querySelector('p.no-data-message');
        if (messageParagraph) {
            messageParagraph.style.display = 'none'; // or messageParagraph.remove();
        }
    }

    if (window.futureChartInstance) {
        window.futureChartInstance.destroy();
    }

    const ctx = container.getContext('2d');
    window.futureChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: response.dates,
            datasets: [{
                label: 'Predicted Price',
                data: response.predictions,
                borderColor: 'rgba(75, 192, 192, 1)',
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { display: true, title: { display: true, text: 'Date' } },
                y: { display: true, title: { display: true, text: 'Price' }, beginAtZero: false }
            },
            plugins: {
                legend: { display: true, position: 'top' },
                tooltip: { enabled: true }
            }
        }
    });
}