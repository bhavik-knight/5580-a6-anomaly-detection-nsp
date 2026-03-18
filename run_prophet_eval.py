#!/usr/bin/env python3
"""Evaluate Prophet model on Halifax test set and save metrics for comparison."""
from pathlib import Path
import pandas as pd
import numpy as np
from prophet import Prophet
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / 'data'
OUTPUT_DIR = ROOT_DIR / 'output'

# Load data
df = pd.read_csv(DATA_DIR / 'nsp_electricity_dataset.csv', parse_dates=['timestamp'], index_col='timestamp', low_memory=False,
                 dtype={'region': 'category', 'consumption_kwh':'float32'})
halifax = df[df['region']=='Halifax']
prophet_df = halifax[['consumption_kwh']].reset_index().rename(columns={'timestamp':'ds','consumption_kwh':'y'})

# Fit a quick Prophet model (or load existing forecast if available)
model = Prophet()
model.fit(prophet_df)

# Create train/test split by time: last 20% as test
n = len(prophet_df)
train_end = int(n*0.8)
train = prophet_df.iloc[:train_end]
test = prophet_df.iloc[train_end:]

future = model.make_future_dataframe(periods=len(test), freq='h')
forecast = model.predict(future)

# Extract forecast for test period
forecast_test = forecast.set_index('ds').loc[test['ds']]

# Compute metrics
y_true = test['y'].values.reshape(-1,1)
y_pred = forecast_test['yhat'].values.reshape(-1,1)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
print(f"Prophet Test MAE: {mae:.3f}, RMSE: {rmse:.3f}, MAPE: {mape:.3f}%")

# Save to the same results CSV
results_path = OUTPUT_DIR / 'model_results.csv'
row = {'Model': 'Prophet', 'MAE': float(mae), 'RMSE': float(rmse), 'MAPE': float(mape)}
if results_path.exists():
    df_res = pd.read_csv(results_path)
    df_res = pd.concat([df_res, pd.DataFrame([row])], ignore_index=True)
else:
    df_res = pd.DataFrame([row])
df_res.to_csv(results_path, index=False)
print(f"Saved Prophet metrics to: {results_path}")

