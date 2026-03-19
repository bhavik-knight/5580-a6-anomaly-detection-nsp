This repository now contains two clean Jupyter notebooks and companion .py files for readable editing.

Files created/updated:
- src/forecasting.ipynb  # Clean notebook with sections for processing, Prophet, LSTM, export
- src/forecasting.py     # Readable script mirroring notebook sections
- src/anomaly_detection.ipynb  # Clean notebook for anomaly detection (Z-score, IQR, IsolationForest)
- src/anomaly_detection.py      # Readable script mirroring notebook sections

Recommended run order (manual cell-by-cell execution):
1. Open `src/forecasting.ipynb` in Jupyter (jupyter lab or notebook).
   - Run the 'Imports and paths' cell to set paths.
   - Run the 'Feature Engineering and Processing' cell to build `data/processed_nsp.csv`.
     - If you prefer a sample during development, edit that cell to read `nrows=200000` or run `src/forecasting.py` directly.
2. Prepare Prophet regressors:
   - In `src/forecasting.ipynb`, run the Prophet preparation cell to create `prophet_df`.
   - Fit Prophet after installing `prophet` (see notebook comments). Add regressors `hdd`, `is_holiday`.
3. Multivariate LSTM:
   - Run the multivariate data-prep cells to scale and create sequences.
   - Add and run Keras training cells if TensorFlow is installed.
4. Export forecast results for Tableau:
   - After obtaining model outputs, run the export cell to create `output/forecast_results.csv`.
5. Open `src/anomaly_detection.ipynb`:
   - Run the first cell to load `data/processed_nsp.csv` (created above).
   - Run Z-score, IQR, and IsolationForest cells (each is self-contained and writes sample outputs in-memory).
   - Combine anomaly signals and save `output/anomaly_results.csv` in the final cell.

Notes:
- I retired standalone scripts in `tools/` and replaced them with placeholders to avoid duplication. The notebooks now hold the canonical, documented workflow.
- The .py companion files (in `src/`) provide readable, non-JSON versions you can open and edit. When ready, you can convert the .py to notebooks or run the cells in Jupyter.

Commands (bash) to open notebooks:
```bash
cd /home/bhavik/Dropbox/edu/smu/winter/data_mining/a6_anomaly_detection
jupyter lab src/forecasting.ipynb
# or
jupyter notebook src/forecasting.ipynb
```

If you'd like, I can also:
- Create minimal unit/check cells that validate the processed CSV contents (e.g., column presence, no NaNs in key columns).
- Add explicit saving steps in the notebooks to write preview CSVs in `output/` for quick inspection.

Tell me if you'd like validation cells or to keep notebooks as-is so you can run them step-by-step.
