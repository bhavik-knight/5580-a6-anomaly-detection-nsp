#!/usr/bin/env python3
"""Create forecast_results.csv combining Prophet and saved PyTorch LSTM predictions.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import os
import joblib

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
OUTPUT_DIR = ROOT / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
data_path = DATA_DIR / 'nsp_electricity_dataset.csv'
df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp', low_memory=False,
                 dtype={'region': 'category', 'consumption_kwh':'float32'})
halifax = df[df['region']=='Halifax']
prophet_df = halifax[['consumption_kwh']].reset_index().rename(columns={'timestamp':'ds','consumption_kwh':'y'})

# Train/test split for Prophet (time-based)
n = len(prophet_df)
train_end = int(n * 0.8)
train = prophet_df.iloc[:train_end]
test = prophet_df.iloc[train_end:]

# Fit Prophet on full data (we only need forecasts aligned to test dates)
from prophet import Prophet
model = Prophet()
model.fit(prophet_df)

# Produce forecast (includes historical dates)
future = model.make_future_dataframe(periods=len(test), freq='h')
forecast = model.predict(future)

# Extract forecast for test period
forecast_test = forecast.set_index('ds').loc[test['ds']]
prophet_export = forecast_test[['yhat','yhat_lower','yhat_upper']].copy()
prophet_export = prophet_export.reset_index()
prophet_export['model'] = 'Prophet'
prophet_export['region'] = 'Halifax'

# LSTM: load scaler and model weights, recreate sequences and predict
scaler_path = OUTPUT_DIR / 'lstm_scaler.pkl'
if not scaler_path.exists():
    raise SystemExit(f"Scaler not found at {scaler_path}; run LSTM training first")
scaler = joblib.load(scaler_path)

# Prepare series and scaled values
values = halifax['consumption_kwh'].sort_index().values.reshape(-1,1).astype('float32')
values_scaled = scaler.transform(values)

# Create sequences
window_size = 24
X = []
for i in range(0, len(values_scaled) - window_size):
    X.append(values_scaled[i:i+window_size])
X = np.array(X)

# Split 60/20/20
n_seq = X.shape[0]
train_end_seq = int(n_seq * 0.6)
val_end_seq = int(n_seq * 0.8)
X_test = X[val_end_seq:]

# Load PyTorch model
import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden1=100, hidden2=50, dropout=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(input_size=hidden1, hidden_size=hidden2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden2, 1)
    def forward(self, x):
        out1, _ = self.lstm1(x)
        out1 = self.dropout1(out1)
        out2, _ = self.lstm2(out1)
        out = out2[:, -1, :]
        out = self.dropout2(out)
        out = self.fc(out)
        return out

model_path = OUTPUT_DIR / 'lstm_model_halifax_torch.pt'
if not model_path.exists():
    raise SystemExit(f"LSTM model weights not found at {model_path}; run LSTM training first")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMForecaster().to(device)
state = torch.load(str(model_path), map_location=device)
model.load_state_dict(state)
model.eval()

with torch.no_grad():
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    preds = model(X_test_t).cpu().numpy()

# Align test dates: for sequence split, the y targets correspond to timestamps starting at index window_size
# The test sequences start at index val_end_seq, which corresponds to original timestamps at position val_end_seq + window_size
start_idx = val_end_seq + window_size
end_idx = start_idx + preds.shape[0]
all_dates = halifax.reset_index()['timestamp'].values
test_dates = all_dates[start_idx:end_idx]

# Build LSTM export
lstm_export = pd.DataFrame({
    'ds': test_dates,
    'yhat': preds.flatten(),
    'yhat_lower': preds.flatten() - (2 * np.std(preds)),
    'yhat_upper': preds.flatten() + (2 * np.std(preds)),
    'model': 'LSTM',
    'region': 'Halifax'
})

# For LSTM, inverse-transform yhat to original scale
lstm_export['yhat'] = scaler.inverse_transform(lstm_export['yhat'].values.reshape(-1,1)).flatten()
lstm_export['yhat_lower'] = scaler.inverse_transform(lstm_export['yhat_lower'].values.reshape(-1,1)).flatten()
lstm_export['yhat_upper'] = scaler.inverse_transform(lstm_export['yhat_upper'].values.reshape(-1,1)).flatten()

# Combine both models
forecast_results = pd.concat([prophet_export[['ds','yhat','yhat_lower','yhat_upper','model','region']], lstm_export[['ds','yhat','yhat_lower','yhat_upper','model','region']]], ignore_index=True)

# Save to CSV
out_path = OUTPUT_DIR / 'forecast_results.csv'
forecast_results.to_csv(out_path, index=False)
print(f"Saved forecast results: {len(forecast_results)} rows to {out_path}")
print(forecast_results.head())

