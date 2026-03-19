import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path.cwd() / 'output'
path = OUTPUT_DIR / 'engineered_features.csv'
print('Loading:', path)
df = pd.read_csv(path, parse_dates=['timestamp'])

# Helper metrics
model_results = []

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return mae, rmse, mape

# LSTM class
class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=100, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        prediction = self.fc(last_output)
        return prediction


def forecast_region_lstm(region_name, df_full, sequence_length=24, epochs=20, batch_size=32, subsample_years=2):
    region_df = df_full[df_full['region'] == region_name].copy()
    region_df = region_df.sort_values('timestamp').reset_index(drop=True)

    # Optional: subsample to last N years for speed
    if subsample_years is not None:
        region_df = region_df.tail(365 * 24 * int(subsample_years))
        print(f"Using last {subsample_years} years ({len(region_df)} rows) for training speed")

    if 'hour' not in region_df.columns:
        region_df['hour'] = region_df['timestamp'].dt.hour
    if 'day_of_week' not in region_df.columns:
        region_df['day_of_week'] = region_df['timestamp'].dt.dayofweek
    if 'is_weekend' not in region_df.columns:
        region_df['is_weekend'] = region_df['day_of_week'].isin([5,6]).astype(int)

    feature_cols = ['consumption_kwh', 'consumption_kwh_lag_1h', 'consumption_kwh_lag_24h',
                   'rolling_mean_168h', 'rolling_std_24h',
                   'temperature_c', 'humidity_pct',
                   'hour', 'day_of_week', 'is_weekend',
                   'grid_load_pct', 'renewable_pct']

    region_df = region_df.dropna(subset=feature_cols).reset_index(drop=True)
    if len(region_df) < sequence_length + 10:
        print(f"{region_name}: insufficient data after dropna; skipping")
        return None

    data = region_df[feature_cols].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(data_scaled) - sequence_length):
        X.append(data_scaled[i:i+sequence_length])
        y.append(data_scaled[i+sequence_length, 0])
    X, y = np.array(X), np.array(y).reshape(-1,1)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Force CPU to avoid CUDA OOM and reduce model size
    device = torch.device('cpu')
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)

    model = LSTMForecaster(input_size=len(feature_cols), hidden_size=50, num_layers=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Batch training with DataLoader
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.4f}")

    # Evaluate in batches
    test_dataset = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_pred = model(batch_X)
            predictions.append(batch_pred)

    if predictions:
        y_pred_t = torch.cat(predictions, dim=0)
    else:
        y_pred_t = model(X_test_t)

    y_test_actual = scaler.inverse_transform(np.column_stack([y_test, np.zeros((len(y_test), len(feature_cols)-1))]))[:,0]
    y_pred_actual = scaler.inverse_transform(np.column_stack([y_pred_t.numpy(), np.zeros((len(y_pred_t), len(feature_cols)-1))]))[:,0]

    mae, rmse, mape = calculate_metrics(y_test_actual, y_pred_actual)
    print(f"{region_name}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")
    return {'region': region_name, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


regions = sorted(df['region'].unique())
results = []
for r in regions:
    try:
        res = forecast_region_lstm(r, df, epochs=5)
        if res:
            results.append(res)
    except Exception as e:
        print(f"Error for {r}: {e}")

if results:
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_DIR / 'lstm_metrics_summary.csv', index=False)
    print('Saved:', OUTPUT_DIR / 'lstm_metrics_summary.csv')
else:
    print('No LSTM results generated')
