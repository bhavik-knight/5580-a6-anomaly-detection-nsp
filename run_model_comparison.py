#!/usr/bin/env python3
"""Load model_results.csv, print table, save comparison plots."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / 'output'
results_path = OUTPUT_DIR / 'model_results.csv'

if not results_path.exists():
    print(f"No results file found at {results_path}")
    raise SystemExit(1)

df = pd.read_csv(results_path)
print('\nModel Comparison:')
print(df.to_string(index=False))

# Save comparison plots
fig, axes = plt.subplots(1, 3, figsize=(15,4))
metrics = ['MAE','RMSE','MAPE']
for i, metric in enumerate(metrics):
    axes[i].bar(df['Model'], df[metric])
    axes[i].set_title(f'{metric} Comparison')
    axes[i].set_ylabel(metric)
    axes[i].tick_params(axis='x', rotation=45)
plt.tight_layout()
plot_path = OUTPUT_DIR / 'model_comparison_metrics.png'
plt.savefig(plot_path, bbox_inches='tight')
plt.close()
print(f"Saved comparison plot to: {plot_path}")

