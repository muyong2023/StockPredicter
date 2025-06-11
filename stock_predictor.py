import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def fetch_stock_data(ticker_symbol: str, period: str, interval: str) -> pd.DataFrame | None:
    """
    Fetches historical stock data from Yahoo Finance.

    Args:
        ticker_symbol: The stock ticker symbol (e.g., "AAPL").
        period: The period for which to fetch data (e.g., "1mo", "7d").
        interval: The data interval (e.g., "1m", "5m", "1d").

    Returns:
        A pandas DataFrame containing the stock data (OHLC, Volume, etc.),
        or None if an error occurs during data fetching.
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        data = stock.history(period=period, interval=interval)
        if data.empty:
            print(f"No data found for {ticker_symbol} with period {period} and interval {interval}.")
            return None
        print(f"Successfully fetched data for {ticker_symbol}")
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker_symbol}: {e}")
        return None

def preprocess_data(data: pd.DataFrame | None) -> pd.Series | None:
    """
    Preprocesses the raw stock data DataFrame to extract and clean 'Close' prices.

    Args:
        data: A pandas DataFrame obtained from fetch_stock_data.

    Returns:
        A pandas Series containing cleaned 'Close' prices, or None if processing fails.
    """
    if data is None or data.empty:
        print("No data to preprocess.")
        return None

    if 'Close' not in data.columns:
        print("Error: 'Close' column not found in the data.")
        return None

    close_prices = data['Close']

    # Handle missing values
    if close_prices.isnull().any():
        print(f"Found {close_prices.isnull().sum()} missing values in 'Close' prices.")
        close_prices = close_prices.fillna(method='ffill') # Forward-fill first
        if close_prices.isnull().any(): # If ffill didn't catch all (e.g. leading NaNs)
            close_prices = close_prices.fillna(method='bfill') # Back-fill second
        print("Missing values handled using forward-fill and back-fill.")

    if close_prices.isnull().any():
        # This might happen if the entire series was NaNs or still has NaNs after filling
        print("Warning: Could not fill all missing values. Returning series with NaNs.")

    return close_prices

def predict_prices_arima(series: pd.Series | None, n_predictions: int) -> pd.Series | None:
    """
    Predicts future stock prices using an ARIMA model.

    Args:
        series: A pandas Series of preprocessed 'Close' prices.
        n_predictions: The number of steps to forecast ahead.

    Returns:
        A pandas Series containing the price predictions, or None if prediction fails.
    """
    if series is None:
        print("Cannot make predictions: input series is None.")
        return None

    if len(series) < 10: # ARIMA needs sufficient data to fit
        print(f"Cannot fit ARIMA model: Time series too short ({len(series)} points). Needs at least 10.")
        return None

    # Define ARIMA model order (p,d,q). These are common starting points
    # but may need tuning (e.g., using auto_arima or ACF/PACF plots for specific data).
    order = (5, 1, 0)

    try:
        # Create and fit the ARIMA model
        # Suppress convergence warnings for this basic example.
        # In a real application, these warnings should be investigated.
        model = ARIMA(series, order=order)
        # Using method_kwargs to suppress convergence warnings.
        # In a production system, monitor these warnings.
        model_fit = model.fit()

        # Make predictions
        predictions = model_fit.forecast(steps=n_predictions)
        return predictions
    except Exception as e:
        print(f"Error during ARIMA model fitting or prediction: {e}")
        return None

if __name__ == '__main__':
    try:
        # --- Configuration ---
        ticker_symbol = "MSFT"  # Example: Microsoft. Others: "AAPL", "GOOGL", "SPY"
        data_fetch_period = "1mo" # Fetch 1 month of data. Examples: "7d", "3mo", "1y"
        base_interval = "1m"    # Base granularity. Examples: "5m", "15m", "1h", "1d"
                                # Note: yfinance has limitations on very short intervals for long periods.

        # We want to predict the price at the END of these future intervals (in minutes)
        forecast_target_minutes = [1, 5, 20, 60]

        print(f"Starting stock price prediction for {ticker_symbol}...")
        print(f"Fetching {base_interval} data for {ticker_symbol} over the last {data_fetch_period}...")

        # --- Data Fetching ---
        raw_data_df = fetch_stock_data(ticker_symbol, period=data_fetch_period, interval=base_interval)

        if raw_data_df is None:
            print(f"Could not fetch data for {ticker_symbol}. Exiting.")
            exit()

        # --- Data Preprocessing ---
        base_close_series = preprocess_data(raw_data_df)

        if base_close_series is None:
            print(f"Could not preprocess data for {ticker_symbol}. Exiting.")
            exit()

        print(f"--- Predictions for {ticker_symbol} ---")

        # Ensure index is datetime for correct resampling and context
        if not isinstance(base_close_series.index, pd.DatetimeIndex):
            try:
                base_close_series.index = pd.to_datetime(base_close_series.index, utc=True) # Assuming UTC if not specified by yfinance
            except Exception as e:
                print(f"Could not convert index to DatetimeIndex: {e}. Exiting.")
                exit()

        # yfinance usually returns timezone-aware timestamps if possible (e.g., for recent data).
        # If not, it might be naive. Standardizing to UTC if naive helps consistency.
        if base_close_series.index.tz is None:
            print("Index is timezone-naive, localizing to UTC for consistency.")
            base_close_series = base_close_series.tz_localize('UTC')
        else:
            # Convert to UTC if it's a different timezone
            base_close_series = base_close_series.tz_convert('UTC')

        current_time = base_close_series.index[-1]
        print(f"Based on data up to (UTC): {current_time}")

        # --- Forecasting Loop ---
        for minutes_ahead in forecast_target_minutes:
            print(f"\nPredicting price {minutes_ahead} minute(s) ahead...")

            series_to_predict_on = None

            if minutes_ahead == 1 and base_interval == "1m":
                # If base interval is already 1m and we want 1m prediction, use base series directly
                series_to_predict_on = base_close_series
            elif base_interval.endswith('m') and minutes_ahead % int(base_interval[:-1]) == 0 and minutes_ahead // int(base_interval[:-1]) == 1:
                # General case: if base_interval is Xm and minutes_ahead is also X, use base_series
                # This handles cases like base_interval="5m", minutes_ahead=5
                 if int(base_interval[:-1]) == minutes_ahead:
                     series_to_predict_on = base_close_series

            if series_to_predict_on is None: # If not covered by above direct use cases, then resample
                resample_interval = f'{minutes_ahead}T' # 'T' is pandas offset alias for minutes
                print(f"Resampling data to {resample_interval} interval...")
                try:
                    # Resample the base_close_series to the target interval, taking the last known price
                    resampled_series = base_close_series.resample(resample_interval).last()
                except Exception as e:
                    print(f"Error during resampling to {resample_interval}: {e}. Skipping this interval.")
                    continue

                # Handle NaNs that might be introduced by resampling
                if resampled_series.isnull().any():
                    nan_count = resampled_series.isnull().sum()
                    print(f"Found {nan_count} NaNs after resampling to {resample_interval}. Filling...")
                    resampled_series = resampled_series.fillna(method='ffill').fillna(method='bfill')

                if resampled_series.isnull().any():
                    print(f"Warning: Could not fill all NaNs in resampled series for {minutes_ahead} min interval. Skipping.")
                    continue

                series_to_predict_on = resampled_series

            if series_to_predict_on is None or len(series_to_predict_on) < 10:
                actual_len = len(series_to_predict_on) if series_to_predict_on is not None else 0
                print(f"Not enough data points after resampling/processing for {minutes_ahead} min interval (need >=10, got {actual_len}). Skipping.")
                continue

            # Predict 1 step ahead using the model trained on the (potentially resampled) series
            prediction = predict_prices_arima(series_to_predict_on, n_predictions=1)

            if prediction is not None and not prediction.empty:
                predicted_price = prediction.iloc[0]
                predicted_timestamp = prediction.index[0] # This will be the start of the next interval
                # Ensure predicted_timestamp is also in UTC for consistent display
                if predicted_timestamp.tz is None:
                    predicted_timestamp = predicted_timestamp.tz_localize('UTC')
                else:
                    predicted_timestamp = predicted_timestamp.tz_convert('UTC')
                print(f"Predicted price at {predicted_timestamp} (UTC) (for ~{minutes_ahead} min ahead): {predicted_price:.2f}")
            else:
                print(f"Failed to make a prediction for {minutes_ahead} minute(s) ahead.")

        print("\nPrediction process_data completed.")

    except Exception as e:
        print(f"An unexpected error occurred in the main execution block: {e}")
        import traceback
        traceback.print_exc()

```
