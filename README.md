# Preprocessing_modeling

## Project Overview
This project covers the **preprocessing** and **analysis** of SCADA and weather data, followed by **modeling** for wind power forecasting.  

The workflow consists of the following modules:
- **Exploratory Data Analysis (EDA)** – `eda_tools.py`  
- **SCADA Data Cleaning** – `cleaning.py`  
- **SCADA Data Filtering** – `filtering.py`  
- **End-to-End Pipeline** – `run_steps.py`  

---

## End-to-End Pipeline Steps
1. **EDA**:  
   Initial data inspection (stats, plots, correlations).  

2. **Cleaning**:  
   - Select relevant features (wind speed, wind direction, power, pitch, availability).  
   - Drop invalid values (NaNs, duplicates, availability=0, out-of-range pitch/wind speed, etc.).  
   - Create derived features (`pitch_max`, wind vector components).  

3. **Filtering**:  
   - **Type-1 (Pitch-stop)**: detect θ* knee point on pitch–power curve.  
   - **Type-2 (Curtailment)**: detect setpoint curtailments (`sp < 0.99·Pr` & binding).  
   - Flag and remove affected samples.  

4. **Forecasting**:
   - Load raw SCADA measurements (after cleaning & filtering).
   - Load weather JSON, extract `{timestamp, wind_speed, wind_dir}`, apply unit conversion if needed.
   - Resample SCADA to hourly (mean wind speed, mean power, circular mean wind direction).
   - Merge SCADA × Weather on hourly timestamps.  
   - Apply power-law correction to scale weather wind speed to hub height.  
   - Tune exponent α with grid search (min RMSE).  
   - Optionally refine with Huber regression (robust calibration).
   - Compute wind vector components (`wind_x`, `wind_y`).  
   - Features: wind speed & wind vectors.  
   - Target: hourly power (`power_h1`). 
   - Train/test split (chronological).  
   - Model training with **XGBoost Regressor** and hyperparameter tuning (time-series cv).  
   - Evaluate performance (MAE, RMSE, R², NMAE, NRMSE).  
   - Generate diagnostic plots (scatter, time-series).  

---

