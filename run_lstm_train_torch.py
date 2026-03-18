#!/usr/bin/env python3
"""Run LSTM training using PyTorch (short validation run, 5 epochs).
Saves model and plots to output/ and prints model summary and size.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import os
import joblib
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Read data
data_path = DATA_DIR / "nsp_electricity_dataset.csv"
df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp', low_memory=False,
                 dtype={'region': 'category', 'consumption_kwh':'float32'})
halifax_data = df[df['region']=='Halifax']
series = halifax_data['consumption_kwh'].sort_index()
values = series.values.reshape(-1,1).astype('float32')

# Scale
scaler = MinMaxScaler(feature_range=(0,1))
values_scaled = scaler.fit_transform(values)
joblib.dump(scaler, OUTPUT_DIR / 'lstm_scaler.pkl')

# Windows
window_size = 24
X = []
y = []
for i in range(0, len(values_scaled)-window_size):
    X.append(values_scaled[i:i+window_size])
    y.append(values_scaled[i+window_size])
X = np.array(X)
y = np.array(y)

n = X.shape[0]
train_end = int(n*0.6)
val_end = int(n*0.8)
X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

# Convert to torch tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

train_ds = TensorDataset(X_train_t, y_train_t)
val_ds = TensorDataset(X_val_t, y_val_t)
test_ds = TensorDataset(X_test_t, y_test_t)

batch_size = 32
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# Model
class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden1=100, hidden2=50, dropout=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(input_size=hidden1, hidden_size=hidden2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden2, 1)
    def forward(self, x):
        # x: (batch, seq_len, features)
        out1, _ = self.lstm1(x)  # out1: (batch, seq_len, hidden1)
        out1 = self.dropout1(out1)
        out2, _ = self.lstm2(out1)  # out2: (batch, seq_len, hidden2)
        out = out2[:, -1, :]  # take last time-step
        out = self.dropout2(out)
        out = self.fc(out)
        return out

model = LSTMForecaster().to(device)
print(model)

# Training setup
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop with validation and checkpointing
epochs = 5
best_val_loss = float('inf')
checkpoint_path = OUTPUT_DIR / 'lstm_model_torch_checkpoint.pt'
model_path = OUTPUT_DIR / 'lstm_model_halifax_torch.pt'

train_losses = []
val_losses = []

for epoch in range(1, epochs+1):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    # validation
    model.eval()
    val_running = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            val_running += loss.item() * xb.size(0)
    val_loss = val_running / len(val_loader.dataset)
    val_losses.append(val_loss)

    print(f"Epoch {epoch}/{epochs} - train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}")

    # checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), checkpoint_path)

# Save final model
torch.save(model.state_dict(), model_path)
print('Saved model to', model_path)

# Load best model
best_model = LSTMForecaster().to(device)
best_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
best_model.eval()

# Predict on test set
with torch.no_grad():
    X_test_t = X_test_t.to(device)
    preds_t = best_model(X_test_t).cpu().numpy()

y_pred = scaler.inverse_transform(preds_t)
y_true = scaler.inverse_transform(y_test)

mae = mean_absolute_error(y_true, y_pred)
rmse = math.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
print(f"Test MAE: {mae:.3f}, RMSE: {rmse:.3f}, MAPE: {mape:.3f}%")

# Save metrics to CSV for comparison
import pandas as pd
results_path = OUTPUT_DIR / 'model_results.csv'
row = {'Model': 'LSTM (PyTorch)', 'MAE': float(mae), 'RMSE': float(rmse), 'MAPE': float(mape)}
if results_path.exists():
    df_res = pd.read_csv(results_path)
    df_res = pd.concat([df_res, pd.DataFrame([row])], ignore_index=True)
else:
    df_res = pd.DataFrame([row])
df_res.to_csv(results_path, index=False)
print(f"Appended LSTM metrics to: {results_path}")

# Save loss plot
plt.figure(figsize=(8,4))
plt.plot(train_losses, label='train_loss')
plt.plot(val_losses, label='val_loss')
plt.legend(); plt.title('LSTM Loss (short run - PyTorch)')
plt.savefig(OUTPUT_DIR / 'lstm_loss_curves_halifax_torch_short.png')
plt.close()

# Save predictions plot (test)
plt.figure(figsize=(12,5))
plt.plot(y_true.flatten(), label='actual')
plt.plot(y_pred.flatten(), label='predicted', alpha=0.7)
plt.legend(); plt.title('LSTM Predictions (short run - PyTorch)')
plt.savefig(OUTPUT_DIR / 'lstm_predictions_halifax_torch_short.png')
plt.close()

# Check model file size
size_mb = os.path.getsize(model_path) / (1024*1024)
print(f"Model file size: {size_mb:.2f} MB")

print('Done')

