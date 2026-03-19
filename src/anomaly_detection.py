"""
Anomaly detection notebook converted to a readable Python script.

Sections:
 - Loading & Feature Engineering (ensure processed CSV exists)
 - Understanding existing anomalies (use anomaly_flag and anomaly_type)
 - Z-score method
 - IQR method
 - IsolationForest
 - Forecast-residual based anomalies (instructions)

Run these as notebook cells for guided execution and documentation.
"""
from pathlib import Path
import pandas as pd
import numpy as np

# Paths
ROOT = Path.cwd().parent
DATA = ROOT / 'data' / 'processed_nsp.csv'
if not DATA.exists():
    DATA = ROOT / 'data' / 'processed_nsp_sample.csv'

if not DATA.exists():
    raise SystemExit('Processed dataset not found. Run processing cells in forecasting notebook first.')

df = pd.read_csv(DATA, parse_dates=['timestamp'])

# 1) Inspect existing anomaly labels
print('anomaly_flag present:', 'anomaly_flag' in df.columns)
if 'anomaly_flag' in df.columns:
    print(df['anomaly_flag'].value_counts())

# 2) Z-score per region (example cell)
def zscore_per_region(df):
    out = []
    for r, g in df.groupby('region'):
        vals = pd.to_numeric(g['consumption_kwh'], errors='coerce').dropna()
        mean = vals.mean()
        std = vals.std()
        g = g.copy()
        if std == 0 or np.isnan(std):
            g['zscore'] = 0.0
        else:
            g['zscore'] = (g['consumption_kwh'] - mean) / std
        g['z_anomaly'] = g['zscore'].abs() > 3
        out.append(g)
    return pd.concat(out, ignore_index=True)

# 3) IQR method per region (example cell)
def iqr_per_region(df):
    df = df.copy()
    df['iqr_anomaly'] = False
    for r, g in df.groupby('region'):
        q1 = g['consumption_kwh'].quantile(0.25)
        q3 = g['consumption_kwh'].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        idx = g[(g['consumption_kwh'] < low) | (g['consumption_kwh'] > high)].index
        df.loc[idx, 'iqr_anomaly'] = True
    return df

# 4) IsolationForest example (cell)
def isoforest_detect(df, features=None):
    features = features or ['consumption_kwh', 'consumption_kwh_lag_1h', 'rolling_mean_24h', 'hdd']
    features = [f for f in features if f in df.columns]
    from sklearn.ensemble import IsolationForest
    X = df[features].fillna(0).values
    clf = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    clf.fit(X)
    df['isoforest_score'] = clf.decision_function(X)
    df['isoforest_anomaly'] = clf.predict(X) == -1
    return df

print('Anomaly detection script ready. Run each function in a notebook cell to inspect results and save outputs.')

