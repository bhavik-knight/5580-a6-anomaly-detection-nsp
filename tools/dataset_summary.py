from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'nsp_electricity_dataset.csv'
print(f"Dataset path: {DATA}")

# Count lines (fast, streaming)
line_count = 0
with open(DATA, 'rb') as f:
    for _ in f:
        line_count += 1
print(f"Total lines (including header): {line_count}")

# Read small sample
sample = pd.read_csv(DATA, nrows=5, parse_dates=['timestamp'], low_memory=False)
print(f"Sample shape: {sample.shape}")
print('Columns:')
for c in sample.columns:
    print(f' - {c} (dtype: {sample[c].dtype})')

print('\nFirst 5 rows:')
print(sample.to_string(index=False))

print('\nContains column anomaly_flag:', 'anomaly_flag' in sample.columns)

