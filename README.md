# Time Series Forecasting & Anomaly Detection for Energy Analytics

Advanced data mining project for MCDA 5580 analyzing Nova Scotia Power electricity consumption data.

## Project Structure
```text
📦 Project Root
├── 📂 data/
│   └── nsp_electricity_dataset.csv          # Original dataset
├── 📂 output/
│   ├── engineered_features.csv              # Preprocessed data with lag & rolling features
│   ├── forecast_results.csv                 # Prophet + LSTM forecasts (for Tableau)
│   ├── anomaly_results.csv                  # Anomaly detection results (for Tableau)
│   ├── forecast_metrics_all_regions.csv     # Model performance metrics
│   └── *.png                                # Visualizations
├── 📂 src/
│   ├── eda.ipynb                             # Exploratory Data Analysis
│   ├── forecasting.ipynb                    # Prophet & LSTM forecasting models
│   └── anomaly_detection.ipynb              # Z-Score, IQR, Isolation Forest detection
└── README.md
```

## What’s Included
- EDA and feature engineering with lags, rolling stats, and HDD
- Prophet forecasting with external regressors
- Multivariate LSTM forecasting in PyTorch
- Anomaly detection with Z-Score, IQR, and Isolation Forest
- Tableau-ready exports in `output/`

## How to Run

1. Install dependencies:
```bash
uv sync
```

2. Run the notebooks in order:
```bash
jupyter notebook src/eda.ipynb
jupyter notebook src/forecasting.ipynb
jupyter notebook src/anomaly_detection.ipynb
```

## Outputs
- `output/engineered_features.csv`
- `output/forecast_results.csv`
- `output/anomaly_results.csv`

## Course Info
- MCDA 5580 - Data and Text Mining
- Instructor: Pranay Malusare
- Saint Mary's University
