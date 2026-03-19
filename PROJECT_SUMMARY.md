# Time Series Forecasting & Anomaly Detection for Energy Analytics
## NSP Dataset Project - Complete Work Summary

---

## 1. PROJECT OVERVIEW

### Objective
Build forecasting models (Prophet & LSTM) and anomaly detection systems for Nova Scotia Power electricity consumption data across 5 regions.

### Dataset
- **Source:** NSP Electricity Analytics Dataset
- **Size:** 438,240 rows × 30+ columns
- **Time Range:** January 2, 2015 - December 31, 2024
- **Regions:** Annapolis Valley, Cape Breton, Halifax, Pictou County, South Shore
- **Granularity:** Hourly measurements

---

## 2. FEATURE ENGINEERING (EDA Phase)

### Features Created
1. **Lag Features:**
   - `consumption_kwh_lag_1h` - 1-hour lag
   - `consumption_kwh_lag_24h` - 24-hour (1 day) lag

2. **Rolling Statistics:**
   - `rolling_mean_168h` - 7-day (168-hour) rolling mean
   - `rolling_std_24h` - 24-hour rolling standard deviation

3. **Weather Interaction:**
   - `hdd` - Heating Degree Days: (18 - temperature_c).clip(lower=0)

4. **Categorical Encoding:**
   - One-hot encoded: `region` and `customer_type`

### Existing Time Features
- hour, day_of_week, month, year, week
- is_weekend, season, is_holiday

### Output
- **File:** `output/engineered_features.csv` (73MB, not tracked in git)
- **Columns:** 38 features total

---

## 3. FORECASTING MODELS

### Data Split Strategy
- **Train:** 60%
- **Validation:** 20%
- **Test:** 20%
- Applied to full dataset (all 10 years of data)

### Model 1: Prophet (Facebook Prophet)

#### Configuration
- Seasonality: yearly, weekly, daily (multiplicative mode)
- **Regressors used:**
  - temperature_c
  - humidity_pct
  - hdd (Heating Degree Days)
  - consumption_kwh_lag_1h
  - consumption_kwh_lag_24h
  - rolling_mean_168h
  - rolling_std_24h
  - is_holiday

#### Results (MAE by Region)
| Region | MAE | RMSE | MAPE |
|--------|-----|------|------|
| Annapolis Valley | 21.75 | 48.54 | 29.29 |
| Cape Breton | 38.53 | 83.65 | 32.51 |
| Halifax | 97.52 | 214.45 | 66.20 |
| Pictou County | 13.55 | 28.65 | 43.04 |
| South Shore | 17.93 | 39.14 | 39.30 |

### Model 2: LSTM (PyTorch)

#### Architecture
- **Type:** Multivariate LSTM
- **Layers:** 1 LSTM layer (64 hidden units) + Dense output
- **Sequence Length:** 24 hours
- **Features used (12 total):**
  - consumption_kwh (target)
  - consumption_kwh_lag_1h
  - consumption_kwh_lag_24h
  - rolling_mean_168h
  - rolling_std_24h
  - temperature_c, humidity_pct
  - hour, day_of_week, is_weekend
  - grid_load_pct, renewable_pct

#### Training Configuration
- Device: CUDA (GPU)
- Batch size: 16
- Epochs: 20
- Optimizer: Adam (lr=0.001)
- Loss: MSE
- Memory optimization: Batch processing with cache clearing

#### Results (MAE by Region)
| Region | MAE | RMSE | MAPE |
|--------|-----|------|------|
| Annapolis Valley | 21.35 | 49.62 | 28.39 |
| Cape Breton | 36.15 | 81.54 | 30.72 |
| Halifax | 101.81 | 217.17 | 67.98 |
| Pictou County | 11.33 | 28.10 | 38.42 |
| South Shore | 15.98 | 38.44 | 38.66 |

### Model Comparison
**Best Model by Region (Lowest MAE):**
- Annapolis Valley: LSTM (21.35)
- Cape Breton: LSTM (36.15)
- Halifax: Prophet (97.52)
- Pictou County: LSTM (11.33)
- South Shore: LSTM (15.98)

**Winner:** LSTM outperforms Prophet in 4 out of 5 regions

---

## 4. OUTPUTS FOR TABLEAU DASHBOARD

### File 1: forecast_results.csv
**Columns:** ds, yhat, yhat_lower, yhat_upper, model, region

**Content:**
- Combined Prophet + LSTM forecasts
- Total rows: 175,275
- All 5 regions
- Both models for comparison

**Purpose:** Visualize and compare forecast performance in Tableau

### File 2: anomaly_results.csv
**Columns:** timestamp, region, consumption_kwh, anomaly_flag, anomaly_type, anomaly_method

**Content:**
- Combined Z-Score, IQR, and Isolation Forest anomaly results
- Total rows: 438,240
- All 5 regions
- High-confidence anomalies flagged for Tableau comparison

**Purpose:** Visualize anomaly patterns and method overlap in Tableau

---

## 5. TECHNICAL IMPLEMENTATION DETAILS

### Branch Structure
- `main` - Production-ready code
- `feature/eda` - Exploratory Data Analysis and feature engineering
- `feature/forecasting-with-engineered-features` - Forecast model development
- `feature/anomaly-detection` - Anomaly detection (current)

### Key Files
```text
project/
├── data/
│   └── nsp_electricity_dataset.csv (original data)
├── output/
│   ├── engineered_features.csv (73MB, generated from EDA)
│   ├── forecast_results.csv (175K rows, both models)
│   ├── forecast_metrics_all_regions.csv
│   ├── forecast_comparison_all_regions.png
│   └── [various analysis plots]
├── src/
│   ├── eda.ipynb (feature engineering)
│   ├── forecasting.ipynb (Prophet + LSTM implementation)
│   └── anomaly_detection.ipynb (in progress)
└── tools/
    └── [helper scripts - deprecated, functionality moved to notebooks]
```

### Technology Stack
- **Python Libraries:**
  - pandas, numpy - Data manipulation
  - matplotlib, seaborn, plotly - Visualization
  - scikit-learn - Preprocessing, metrics, Isolation Forest
  - statsmodels - Statistical methods
  - prophet - Facebook Prophet forecasting
  - torch - PyTorch for LSTM
  - scipy - Statistical tests

---

## 6. CHALLENGES SOLVED

### Challenge 1: GPU Memory Issues with LSTM
**Problem:** CUDA out of memory (tried to allocate 14.28 GiB on 7.62 GiB GPU)

**Solution:**
- Implemented batch training with DataLoader
- Reduced model complexity (64 hidden units, 1 layer)
- Added memory clearing after each batch
- Used batch_size=16

### Challenge 2: Feature Engineering Consistency
**Problem:** Need to ensure features created in EDA match forecasting requirements

**Solution:**
- Created `engineered_features.csv` as single source of truth
- Both forecasting and anomaly detection load from same file
- Added rolling statistics as per assignment requirements (168h mean, 24h std)

### Challenge 3: Multi-Region Processing
**Problem:** Need to train separate models for 5 regions efficiently

**Solution:**
- Created reusable functions: `forecast_region_prophet()`, `forecast_region_lstm()`
- Loop through regions with progress tracking
- Store results in combined dataframe for comparison

---

## 7. KEY DECISIONS MADE

1. **60/20/20 Split:** Chose train/validation/test split instead of 80/20 for better evaluation
2. **Full Dataset:** Used complete 10-year dataset instead of subsampling for production-quality results
3. **Multivariate LSTM:** Included 12 features instead of univariate for better performance
4. **Per-Region Processing:** Train separate models per region instead of single global model
5. **CPU Fallback:** Keep CPU training option for reproducibility without GPU

---

## 8. RESULTS SUMMARY

### Forecasting Performance
- **LSTM** is the recommended model for 4/5 regions
- **Halifax** is the exception where Prophet performs better (larger, more complex patterns)
- Error metrics are reasonable for production use
- Uncertainty intervals provided for both models

### Anomaly Detection Performance
- **Z-Score** identified 3,373 anomalies
- **IQR** identified 6,596 anomalies
- **Isolation Forest** identified 8,765 anomalies
- **High-confidence overlap** flagged 4,412 anomalies detected by 2+ methods
- `anomaly_results.csv` is exported and ready for Tableau

### Next Steps
1. Create final presentation with:
   - Model comparison visualizations
   - Regional performance analysis
   - Recommendations for deployment

---

## 9. ASSIGNMENT REQUIREMENTS MET

✅ Feature Engineering (lags, rolling statistics, weather interactions)
✅ Prophet Forecasting (with multiple regressors)
✅ LSTM Forecasting (multivariate, PyTorch)
✅ Multiple regions (all 5 regions)
✅ Model comparison (metrics, visualizations)
✅ Tableau exports (`forecast_results.csv` and `anomaly_results.csv`)
✅ Anomaly Detection (Z-Score, IQR, Isolation Forest)
✅ Anomaly export for Tableau

---

## 10. REPOSITORY STATE

### Current Branch: `main`
### Latest Commits:
```text
- Update README and project summary
- Merge feature/eda into main
- Add Isolation Forest and create Tableau anomaly export
```

### Files Ready for Presentation:
- All forecast metrics and visualizations in `output/`
- Jupyter notebooks with complete workflow
- Combined forecast export for Tableau dashboard
- Final anomaly export for Tableau dashboard
