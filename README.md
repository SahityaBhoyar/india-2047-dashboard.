# India 2047 Electricity Generation Forecast & Dashboard

An interactive dashboard and modeling suite forecasting India's electricity generation (in TWh) through 2047.

## Project Files
- **`index.html`**: Interactive web dashboard visualizing 2047 generation scenarios, fuel mix, and regional zone splits.
- **`model.py`**: Deterministic scenario and sensitivity (tornado) model estimating 2047 generation based on per-capita growth, urbanization, and population drivers.
- **`ml_forecast.py`**: Multi-method ML forecasting script combining log-linear trend, damped Holt exponential smoothing, Ridge driver regression, and logistic S-curve models into a backtest-weighted ensemble.
- **`renewable_energy_dataset.csv`**: Historical input dataset used for training and backtesting.

## How to Run Locally
1. Run the deterministic scenario model:
   ```bash
   python model.py
