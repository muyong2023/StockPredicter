# Simple Stock Price Predictor

This script uses Python and the ARIMA time series model to provide basic stock price predictions for 1, 5, 20, and 60 minutes into the future.

## Description

The script performs the following steps:
1. Fetches historical stock data (defaulting to 1-minute intervals for the past month for a specified ticker) using the `yfinance` library.
2. Preprocesses the data, focusing on 'Close' prices and handling missing values by forward and backward filling.
3. For each target forecast interval (1, 5, 20, 60 minutes):
    a. Resamples the historical 'Close' price series to the respective interval if needed (e.g., from 1-minute data to 5-minute data). It takes the last available price in each new interval.
    b. Trains an ARIMA(5,1,0) model on the (potentially resampled) data. The (5,1,0) order means it uses 5 lagged values, 1 differencing order to make the series stationary, and 0 lagged forecast errors.
    c. Predicts the next single price point for that interval.
4. Prints the predictions, including the timestamp for which the prediction is valid (in UTC).

## Limitations

*   **Educational Purposes Only**: This is a highly simplified model and should NOT be used for actual trading or as financial advice. Stock market prediction is extremely complex and influenced by numerous factors not captured here.
*   **ARIMA Model Simplification**: The ARIMA model uses fixed parameters (p=5, d=1, q=0). For better accuracy, these parameters would need to be tuned for specific datasets (e.g., using techniques like ACF/PACF analysis, grid search, or auto-ARIMA).
*   **Data Dependent**: The quality, quantity, and stationarity of the historical data significantly impact predictions. The chosen `data_fetch_period` should be appropriate for the `base_interval`.
*   **No External Factors**: The model does not consider news, market sentiment, economic indicators, corporate actions, or other external factors that heavily influence stock prices.
*   **Resampling Effects**: Resampling data (e.g., from 1-minute to 5-minute intervals) can smooth out noise but also lead to loss of information. The choice of `last()` price during resampling is one of many possibilities (e.g., `ohlc()`, `mean()`).
*   **Single Point Forecast**: The script forecasts only the next price point for each interval.

## Requirements

Make sure you have Python 3 (preferably 3.7+) installed. Install the necessary libraries using `pip` and the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```
The `requirements.txt` file includes:
- yfinance
- pandas
- numpy
- statsmodels
- matplotlib (currently optional, but good for future plotting extensions)

## Usage

1.  **Modify Script Parameters (Optional but Recommended)**:
    You can change the stock and data parameters directly within the `if __name__ == '__main__':` block in `stock_predictor.py`:
    *   `ticker_symbol`: The stock ticker you want to analyze (e.g., "AAPL", "GOOGL", "MSFT", "SPY").
    *   `data_fetch_period`: The amount of historical data to fetch (e.g., "1mo", "7d", "3mo"). Be mindful of `yfinance` limitations for very fine intervals over long periods.
    *   `base_interval`: The granularity of the initially fetched data (e.g., "1m", "5m", "15m", "1h"). Note that predictions are made for 1, 5, 20, and 60 minutes ahead, and data is resampled from this base interval if necessary.

2.  **Run the Script**:
    Execute the script from your terminal:
    ```bash
    python stock_predictor.py
    ```

3.  **Interpret Output**:
    The script will print messages indicating its progress: data fetching, preprocessing, and then the predicted price for each specified future interval (1, 5, 20, and 60 minutes). Timestamps are shown in UTC. For example:
    ```
    Starting stock price prediction for MSFT...
    Fetching 1m data for MSFT over the last 1mo...
    Successfully fetched data for MSFT
    --- Predictions for MSFT ---
    Based on data up to (UTC): YYYY-MM-DD HH:MM:SS+00:00

    Predicting price 1 minute(s) ahead...
    Predicted price at YYYY-MM-DD HH:MM:SS+00:00 (UTC) (for ~1 min ahead): PREDICTED_PRICE

    Predicting price 5 minute(s) ahead...
    Resampling data to 5T interval...
    Predicted price at YYYY-MM-DD HH:MM:SS+00:00 (UTC) (for ~5 min ahead): PREDICTED_PRICE
    ...and so on for 20 and 60 minutes.
    Prediction process_data completed.
    ```
    If errors occur (e.g., invalid ticker, network issues, insufficient data for ARIMA), informative messages will be printed.

## Future Enhancements (Ideas)
*   Implement `pmdarima.auto_arima` for automatic ARIMA order selection.
*   Add plotting of historical data and predictions using `matplotlib`.
*   Incorporate more sophisticated time series models (e.g., GARCH for volatility, Prophet, or LSTM neural networks).
*   Implement a walk-forward validation strategy to evaluate model performance more rigorously.
*   Allow configuration of parameters via command-line arguments instead of direct script modification.
*   Add basic technical indicators as potential exogenous variables for ARIMA (ARIMAX).