from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / 'data'
path = DATA_DIR / 'nsp_electricity_dataset.csv'

print(f"Loading dataset from: {path}")

df = pd.read_csv(path, parse_dates=['timestamp'], low_memory=False)

print(f"shape: {df.shape}")
print('columns:')
print(list(df.columns))
print('\nanomaly_flag present:', 'anomaly_flag' in df.columns)
print('\nfirst 5 rows:')
print(df.head().to_string())

