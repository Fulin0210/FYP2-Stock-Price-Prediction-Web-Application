from flask import url_for, Blueprint, request, render_template, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Initialize Blueprint
views = Blueprint('views', __name__)

# Load stock data
stocks_df = pd.read_csv('StocksBySector.csv')

# ============= Route Handlers =============

@views.route('/')
def home():
    return render_template("index.html")

@views.route('/info')
def info():
    return render_template('info.html')

@views.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@views.route('/get_sectors', methods=['GET'])
def get_sectors():
    sectors = stocks_df['Sector'].unique().tolist()
    print("Available sectors:", sectors) 
    return jsonify(sectors)

@views.route('/get_stocks/<sector>', methods=['GET'])
def get_stocks(sector):
    print(f"Sector selected: {sector}")
    stocks = stocks_df[stocks_df['Sector'].str.lower() == sector.lower()][['Ticker Name', 'Symbol']]
    print(stocks) 
    return jsonify(stocks.to_dict(orient='records')) 

@views.route('/get_stock_data', methods=['GET'])
def get_stock_data():
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({"error": "No ticker name provided"}), 400

    try:
        stock = yf.Ticker(ticker)
        historical_data = stock.history(period="10y", auto_adjust=False).reset_index()
        
        num_cols = historical_data.select_dtypes(include=np.number).columns
        historical_data[num_cols] = historical_data[num_cols].round(2)

        print(f"Historical data for {ticker} fetched with shape: {historical_data.shape}")
        print(historical_data.head(10)) 
        print(historical_data.describe())
        
        # Get the previous day's closing price
        prev_close = historical_data['Close'].iloc[-2] if len(historical_data) > 1 else None

        return jsonify({"data": historical_data.to_dict(orient="records"), "prev_close": prev_close})

    except Exception as e:
        print(f"Error fetching historical data for {ticker}: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ============= Model Training and Evaluation =============

def preprocess_data(stock_data, scale=False):
    """Preprocess stock data for model training and testing."""
    
    # Shift Close price by -1 to predict the next day's close
    stock_data['Close_Price_Target'] = stock_data['Close'].shift(-1)
    stock_data = stock_data.dropna(subset=['Close_Price_Target'])
    
    X = stock_data[['Open', 'High', 'Low', 'Adj Close', 'Close', 'Volume']]
    y = stock_data['Close_Price_Target']
    
    scaler = None
    if scale:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    return X_train, X_test, y_train, y_test, scaler

def build_model(model_type, input_dim=None):
    if model_type == 'RF':
        return RandomForestRegressor(
            n_estimators=100,
            max_features=None,
            random_state=42
        )
    elif model_type == 'SVR':
        return SVR(
            C=100,
            epsilon=0.01,
            gamma=0.01,
            cache_size=1000,
            kernel='rbf'
        )
    elif model_type == 'ANN':
        num_of_layers = 4
        num_of_nodes = 64
        learning_rate = 0.001
        dropout_rate = 0.2

        model = Sequential()
        model.add(Dense(units=num_of_nodes, input_dim=input_dim, activation='relu'))

        for _ in range(num_of_layers - 1):
            model.add(Dense(units=num_of_nodes, activation='relu'))
            model.add(Dropout(dropout_rate))

        model.add(Dense(1))
        model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
        return model
    else:
        raise ValueError("Invalid model type")

def calc_metrics(model, X_test, y_test):
    """Calculate model performance metrics."""
    if isinstance(model, Sequential):
        y_pred = model.predict(X_test, verbose=0)
        y_pred = y_pred.flatten()
    else:  # RF or SVR models
        y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return mae, rmse, r2, y_pred

# ============= Prediction Functions =============
def find_last_trading_day(df, prediction_date):
    print("Inside find_last_trading_day()")
    print("Original prediction_date:", prediction_date, type(prediction_date))

    # Ensure prediction_date is pd.Timestamp
    if not isinstance(prediction_date, pd.Timestamp):
        prediction_date = pd.to_datetime(prediction_date)
        print("Converted prediction_date to pd.Timestamp:", prediction_date)

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        print("df['Date'] dtype after conversion:", df['Date'].dtype)

        trading_days = df['Date'].sort_values().unique()
        print(f"Number of trading_days found: {len(trading_days)}")
        print("Sample trading_days:", trading_days[:5])
    else:
        trading_days = df.index.sort_values().unique()
        print(f"Number of trading_days from index: {len(trading_days)}")
        print("Sample trading_days from index:", trading_days[:5])

    trading_tz = trading_days[0].tzinfo
    print("Trading days timezone:", trading_tz)

    if prediction_date.tzinfo is None and trading_tz is not None:
        prediction_date = prediction_date.tz_localize(trading_tz)
        print("Localized prediction_date to trading days tz:", prediction_date)
    # If prediction_date has tz but differs from trading_days tz, convert it
    elif prediction_date.tzinfo != trading_tz:
        prediction_date = prediction_date.tz_convert(trading_tz)
        print("Converted prediction_date tz to trading days tz:", prediction_date)

    try:
        valid_days = trading_days[trading_days <= prediction_date]
        print(f"Number of valid_days <= prediction_date: {len(valid_days)}")
    except Exception as e:
        print("Error filtering valid_days:", e)
        valid_days = []

    if len(valid_days) == 0:
        print("No valid trading days found before or equal to prediction_date")
        return None  

    last_day = valid_days[-1]
    print("Last trading day found:", last_day, type(last_day))
    return last_day

def estimate_next_features(last_input, predicted_close, scaler=None):
    if scaler is None:
        next_features = last_input.copy()
        next_features[0] = predicted_close * 0.995
        next_features[1] = predicted_close * 1.005
        next_features[2] = predicted_close * 0.99
        next_features[3] = predicted_close
        next_features[4] = predicted_close
        return next_features
    else:
         # Scaled: inverse transform to get raw values
        last_input_raw = scaler.inverse_transform(last_input.reshape(1, -1))[0]

        close_price = last_input_raw[3]
        if close_price == 0:
            # Prevent division by zero - fallback to fixed multipliers
            open_to_close = 0.995
            high_to_close = 1.005
            low_to_close = 0.99
            adjclose_to_close = 1.0
        else:
            open_to_close = last_input_raw[0] / close_price
            high_to_close = last_input_raw[1] / close_price
            low_to_close = last_input_raw[2] / close_price
            adjclose_to_close = last_input_raw[4] / close_price

        last_input_raw[3] = predicted_close               # Close
        last_input_raw[0] = predicted_close * open_to_close  # Open
        last_input_raw[1] = predicted_close * high_to_close  # High
        last_input_raw[2] = predicted_close * low_to_close   # Low
        last_input_raw[4] = predicted_close * adjclose_to_close  # Adj Close

        # Return scaled features
        return scaler.transform(last_input_raw.reshape(1, -1))[0]

@views.route('/evaluate_model', methods=['POST'])
def evaluate_model():
    """Evaluate model performance on historical data."""
    try:
        data = request.get_json()
        print("Received evaluate_model POST data:", data)

        stock_symbol = data['stock_symbol']
        prediction_date = pd.to_datetime(data['prediction_date'])
        ml_model = data['model']

        print(f"Evaluating {ml_model} for {stock_symbol} until {prediction_date}")

        start_date = prediction_date - pd.DateOffset(years=10)
        stock_data = yf.Ticker(stock_symbol).history(start=start_date, end=prediction_date, auto_adjust=False).reset_index()
        print("Downloaded stock data shape:", stock_data.shape)

        if stock_data.empty:
            print("Error: No historical data found.")
            return jsonify({"success": False, "error": "No historical data found."})

        num_cols = stock_data.select_dtypes(include=np.number).columns
        stock_data[num_cols] = stock_data[num_cols].round(2)

        scale = ml_model in ['SVR', 'ANN']
        X_train, X_test, y_train, y_test, scaler = preprocess_data(stock_data, scale=scale)
        print(f"Data split: X_train={X_train.shape}, X_test={X_test.shape}, y_train={y_train.shape}, y_test={y_test.shape}")

        model = build_model(ml_model, input_dim=X_train.shape[1])
        print(f"Model {ml_model} built successfully.")

        if ml_model == 'ANN':
            print("Training ANN model...")
            model.fit(X_train, y_train, epochs=100, batch_size=32,
                      validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=3)], verbose=0)
        else:
            print(f"Training {ml_model} model...")
            model.fit(X_train, y_train)

        print("Model training complete. Calculating metrics...")
        mae, rmse, r2, y_pred = calc_metrics(model, X_test, y_test)
        print(f"Metrics: MAE={mae}, RMSE={rmse}, R2={r2}")

        results = {
            'success': True,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred.tolist(),
            'dates': stock_data['Date'].iloc[-len(y_test):].tolist(),
        }

        return jsonify(results)

    except Exception as e:
        print("Error during evaluation:", str(e))
        return jsonify({"success": False, "error": str(e)}), 500

@views.route('/predict_future_price', methods=['POST'])
def predict_future_price():
    try:
        data = request.get_json()
        print("Received predict_future_price POST data:", data)

        stock_symbol = data['stock_symbol']
        prediction_date = pd.to_datetime(data['prediction_date'])

        ml_model = data['model']
        scale = ml_model in ['SVR', 'ANN']  # scale only for these models

        timeframe_str = data['timeframe']

        unit = timeframe_str[-1]
        amount = int(timeframe_str[:-1])
        if unit == 'd':
            timeframe = amount
        elif unit == 'w':
            timeframe = amount * 7
        elif unit == 'm':
            timeframe = amount * 30
        else:
            timeframe = amount
        print(f"Timeframe (days): {timeframe}")

        # Download historical data ending before prediction_date
        start_date = prediction_date - pd.DateOffset(years=10)
        stock_data = yf.Ticker(stock_symbol).history(
            start=start_date, end=prediction_date, auto_adjust=False
        ).reset_index()

        if stock_data.empty:
            return jsonify({"success": False, "error": "No historical data found."})

        # Round numeric columns
        num_cols = stock_data.select_dtypes(include=np.number).columns
        stock_data[num_cols] = stock_data[num_cols].round(2)

        # Preprocess data and scaling info
        X_train, _, y_train, _, scaler = preprocess_data(stock_data, scale=scale)

        # Build and train model
        model = build_model(ml_model, input_dim=X_train.shape[1])
        if ml_model == 'ANN':
            model.fit(
                X_train,
                y_train,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=[EarlyStopping(monitor='val_loss', patience=3)],
                verbose=0,
            )
        else:
            model.fit(X_train, y_train)

        # Find last trading day before prediction_date
        prev_day = prediction_date - pd.Timedelta(days=1)
        last_trading_day = find_last_trading_day(stock_data, prev_day)

        print("stock_data['Date'] dtype:", stock_data['Date'].dtype)
        print("last_trading_day:", last_trading_day, type(last_trading_day), last_trading_day.tzinfo)
        print("stock_data['Date'] sample values:", stock_data['Date'].head())

        # Ensure last_trading_day has same timezone as stock_data['Date']
        if stock_data['Date'].dt.tz is not None and last_trading_day is not None:
            if last_trading_day.tzinfo is None:
                last_trading_day = last_trading_day.tz_localize(stock_data['Date'].dt.tz)
                print("Localized last_trading_day timezone:", last_trading_day.tzinfo)
            else:
                last_trading_day = last_trading_day.tz_convert(stock_data['Date'].dt.tz)
                print("Converted last_trading_day timezone:", last_trading_day.tzinfo)

        # Safely filter last day data with try-except to catch errors
        try:
            last_day_data = stock_data[stock_data['Date'] == last_trading_day]
        except Exception as e:
            print("Error in filtering last_day_data:", e)
            print("stock_data['Date'].dt.tz:", stock_data['Date'].dt.tz)
            print("Is last_trading_day tz-aware?", last_trading_day.tzinfo is not None)
            last_day_data = pd.DataFrame()  # fallback to empty DataFrame

        if last_day_data.empty:
            return jsonify({"success": False, "error": "No data found for last trading day."})

        # Prepare input features for prediction
        features_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        if scale and scaler:
            last_input = scaler.transform(last_day_data[features_cols].values)[0]
        else:
            last_input = last_day_data[features_cols].values.flatten()

        predictions = []
        for i in range(timeframe):
            input_array = np.array(last_input).reshape(1, -1)

            if ml_model == 'ANN':
                next_pred = model.predict(input_array, verbose=0)[0][0]
            else:
                next_pred = model.predict(input_array)[0]


            predictions.append(float(next_pred))
            print(f"Day {i + 1} prediction: {next_pred}")

            if scale and scaler:
                last_input = estimate_next_features(last_input, next_pred, scaler)
            else:
                last_input = estimate_next_features(last_input, next_pred, None)

        prediction_dates = []
        current_date = prediction_date + pd.Timedelta(days=1)

        while len(prediction_dates) < timeframe:
            if current_date.weekday() < 5: 
                prediction_dates.append(current_date)
            current_date += pd.Timedelta(days=1)

        future_predictions_df = pd.DataFrame({'Date': prediction_dates, 'Predicted Price': predictions})

        return jsonify({
            'success': True,
            'predictions': predictions,
            'dates': future_predictions_df['Date'].dt.strftime('%Y-%m-%d').tolist(),
        })

    except Exception as e:
        print("Error during future prediction:", str(e))
        return jsonify({"success": False, "error": str(e)}), 500

# ============= Sector Analysis =============

@views.route('/get_sector_performance', methods=['GET'])
def get_sector_performance():
    """Analyze and return sector performance metrics."""
    try:
        sector = request.args.get('sector')
        timeframe = request.args.get('timeframe', '1y')
        selected_stock = request.args.get('selected_stock')
        if not sector:
            return jsonify({"error": "No sector provided"}), 400

        # Get stocks in the sector
        sector_stocks = stocks_df[stocks_df['Sector'].str.lower() == sector.lower()]
        
        performance_data = []
        valid_stocks = 0
        
        for _, stock in sector_stocks.iterrows():
            try:
                ticker = yf.Ticker(stock['Symbol'])

                # Define period mapping for yfinance history call
                period_map = {
                    '1d': '2d',
                    '1w': '5d',
                    '1m': '1mo',
                    '6m': '6mo',
                    '1y': '1y',
                    '5y': '5y',
                    '10y': '10y',
                    'ytd': 'ytd'
                }
                period = period_map.get(timeframe, '1y')
                interval = '1d'

                hist = ticker.history(period=period, interval=interval, auto_adjust=False)

                if hist.empty or len(hist) < 2:
                    continue

                start_date = hist.index[0].strftime('%Y-%m-%d')
                end_date = hist.index[-1].strftime('%Y-%m-%d')

                start_price = round(hist['Close'].iloc[0], 2)
                end_price = round(hist['Close'].iloc[-1], 2)
                previous_close = hist['Close'].iloc[-2]
                percent_change = ((end_price - start_price) / start_price * 100) if start_price > 0 else 0

                market_cap = ticker.info.get('marketCap', 0)
                market_cap = round(market_cap / 1e9, 2) if market_cap else 0

                valid_stocks += 1
                
                performance_data.append({
                    'symbol': stock['Symbol'],
                    'name': stock['Ticker Name'],
                    'percent_change': round(percent_change, 2),
                    'current_price': round(end_price, 2),
                    'market_cap': market_cap,
                    'previous_close': round(previous_close, 2),
                    'start_price': round(start_price, 2),
                    'end_price': round(end_price, 2),
                    'start_date': start_date,
                    'end_date': end_date,
                })

                print(f"[{stock['Symbol']}] {start_date}: {start_price} -> {end_date}: {end_price} | Change: {percent_change:.2f}%")

            except Exception as e:
                print(f"Error fetching data for {stock['Symbol']}: {str(e)}")
                continue
        
        if not performance_data:
            return jsonify({
                "success": False,
                "error": "No valid performance data found for the sector"
            }), 404

        # Sort by percent change to get top performers
        performance_data.sort(key=lambda x: x['percent_change'], reverse=True)

        # Always include the selected stock in the returned data
        top5 = performance_data[:5]
        if selected_stock:
            found = next((x for x in performance_data if x['symbol'] == selected_stock), None)
            if found and found not in top5:
                top5.append(found)

        sector_stats = {
            "total_stocks": len(sector_stocks),
            "valid_stocks": valid_stocks,
            "top_performer": performance_data[0] if performance_data else None,
        }

        return jsonify({
            "success": True,
            "data": top5,
            "sector_stats": sector_stats
        })

    except Exception as e:
        print(f"Error in get_sector_performance: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
