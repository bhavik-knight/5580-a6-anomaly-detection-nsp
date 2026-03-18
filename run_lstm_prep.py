#!/usr/bin/env python3
"""Run LSTM data preparation steps from the notebook in a lightweight script.
This script:
- reads the dataset
- filters Halifax
- scales consumption_kwh with MinMaxScaler
- creates 24-hour sliding windows
- splits into train/val/test (60/20/20)
- saves scaler and a small diagnostic plot to output/
- prints shapes and sample values
"""
from pathlib import Path
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

data_path = DATA_DIR / "nsp_electricity_dataset.csv"
if not data_path.exists():
    print(f"ERROR: data file not found at {data_path}")
    raise SystemExit(1)

print(f"Reading data from: {data_path}")

# Read CSV
df = pd.read_csv(
    data_path,
    parse_dates=['timestamp'],
    index_col='timestamp',
    dayfirst=False,
    low_memory=False,
    na_values=['?', 'NA', ''],
    dtype={
        'region': 'category',
        'consumption_kwh': 'float32'
    }
)

halifax_data = df[df['region'] == 'Halifax']
series = halifax_data['consumption_kwh'].sort_index()
print(f"Halifax series shape: {series.shape}")

# Convert to numpy array
values = series.values.reshape(-1, 1).astype('float32')

# Normalize
scaler = MinMaxScaler(feature_range=(0, 1))
values_scaled = scaler.fit_transform(values)
scaler_path = OUTPUT_DIR / 'lstm_scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"Saved MinMaxScaler to: {scaler_path}")

# Create windows
window_size = 24
step = 1
X = []
 y = []
for i in range(0, len(values_scaled) - window_size, step):
    X.append(values_scaled[i:i+window_size])
    y.append(values_scaled[i+window_size])
X = np.array(X)
y = np.array(y)
print(f"Created sequences -> X: {X.shape}, y: {y.shape}")

# Split
n_samples = X.shape[0]
train_end = int(n_samples * 0.6)
val_end = int(n_samples * 0.8)
X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Diagnostic plot
plt.figure(figsize=(10, 4))
plt.plot(values[:200], label='original')
plt.plot(scaler.inverse_transform(values_scaled[:200]), label='scaled->inv')
plt.title('Original vs Scaled (first 200 points)')
plt.legend()
plot_path = OUTPUT_DIR / 'lstm_data_scaled.png'
plt.savefig(plot_path, bbox_inches='tight')
plt.close()
print(f"Saved diagnostic plot to: {plot_path}")

# Sample outputs
if X_train.shape[0] > 0:
    print('Sample X_train[0] first 5 values (scaled):')
    print(X_train[0][:5].flatten())
    print('Sample y_train[0] inverse-scaled:')
    print(scaler.inverse_transform(y_train[0].reshape(1, -1)))
else:
    print('No training samples created (check window/series length).')

