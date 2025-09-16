from res_forecasting.helpers.wf_data_handler import WFDataHandler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

from res_forecasting.data.preprocessing_modeling.cleaning import clean_scada_data
from res_forecasting.data.preprocessing_modeling.filtering import apply_prefilters

# Load raw SCADA data
with WFDataHandler("wf_data.duckdb") as wf_db_handler:
    scada_df = wf_db_handler.get_df(turbine_id=1)

# Explore Exploratoty Data Analysis (EDA) tools if you like
#from res_forecasting.data.preprocessing.eda_tools import explore_and_save
#results = explore_and_save(scada_df, output_dir="eda_outputs", correlation_threshold=0.75)

# Rename/keep columns
col_map = {
     "date_and_time": "timestamp",
     "wind_speed_ms": "wind_speed",
     "wind_direction": "wind_dir",
     "power_kw":      "power",
     "turbine_power_setpoint_kw":         "power_setpoint",
     "blade_angle_pitch_position_a":  "pitch_a",
     "blade_angle_pitch_position_b":  "pitch_b",
     "blade_angle_pitch_position_c":  "pitch_c",
   }

features_to_keep = [
    "timestamp","wind_speed","wind_dir","power",
    "power_setpoint","pitch_a","pitch_b","pitch_c", "data_availability", "potential_power_default_pc_kw", "potential_power_learned_pc_kw"
]

# Clean SCADA data 
cleaned_df, report = clean_scada_data(
     scada_df,
     col_map=col_map,
     features_to_keep=features_to_keep,
     dropna=True,                # keep your behavior
     basic_filter=True,
     min_wind_speed=None,
     max_wind_speed=None,
     make_wind_vectors=True,
     drop_wind_dir=False,
     verbose=True,               # prints your report
     pitch_bounds=(0.0, 95.0),
)

# Apply prefilters
Pr = 2050.0  # kW
pref_df, summary = apply_prefilters(
    cleaned_df,
    Pr=Pr,
    min_setpoint_frac=None,   # ignore very low setpoints
    obs_tol=0.15,             # default: power within 10% of setpoint → binding
    msf_kwargs={"near":(0.15,0.30), "min_count":50, "min_days":3},
    knee_kwargs={"q_hi":0.95, "bin_deg":1.0, "min_pts_per_bin":20, "drop_from_plateau":0.05, "return_debug": True}
    #keep: "kept" (default),
    #include_flags: False (default)
)

import json, os
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import HuberRegressor

"""
SCADA + Weather hourly merge and calibration pipeline:
1) Διαβάζει ωριαία weather JSON και εξάγει {timestamp, wind_speed, wind_dir} σε dataframe
2) Προαιρετκή μεταροπή μονάδων ταχύτητας (π.χ. mph → m/s)
3) Φιλτράρισμα σε [START, END] inclusive
4) Resample SCADA σε ωριαίο (mean wind_speed/power, circular mean wind_dir)
5) Inner-merge SCADA×WX στο ίδιο ωριαίο timestamp
6) Height correction (power-law) + calibration (Huber) σε train split
7) Υπολογισμός metrics (bias/MAE/RMSE/corr) για ταχύτητα/διεύθυνση
"""

WEATHER_JSON = os.path.expanduser(
    "~/projects/res_forecasting/res_forecasting/data/turbine1.json"
)
START = pd.Timestamp("2020-01-01")
END   = pd.Timestamp("2021-10-31")  # inclusive
ASSUME_WX_UNITS = "mph"  # "mph" ή "mps" (αν τα JSON είναι ήδη m/s)
MPH_TO_MPS = 0.44704

HUB_HEIGHT = 78.5  # m (ύψος πλήμνης)
Z_WX       = 10.0  # m (τυπικό ύψος μετεωρολογικών μετρήσεων)

ALPHAS = np.linspace(0.08, 0.30, 23)  # grid για power-law exponent
TRAIN_CUTOFF = pd.Timestamp("2021-01-01")  # train split (προαιρετικό)

# ------------------ Helpers (math) ------------------
def circular_mean_deg(deg_series: pd.Series) -> float:
    """Κυκλικός μέσος γωνιών (deg) στο [0, 360)."""
    vals = pd.to_numeric(deg_series, errors="coerce").dropna().values
    if vals.size == 0:
        return np.nan
    rad = np.deg2rad(vals)
    x = np.mean(np.cos(rad))
    y = np.mean(np.sin(rad))
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0

def circular_diff_deg(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray):
    """Signed διαφορά διεύθυνσης (deg) στο [-180, 180]."""
    return ((np.asarray(a) - np.asarray(b) + 180.0) % 360.0) - 180.0

def height_correct(v: np.ndarray, alpha: float, hub_height: float, z_ref: float):
    """Power-law correction από z_ref → hub_height."""
    return v * (hub_height / z_ref) ** float(alpha)

# ------------------ Weather I/O ---------------------
def weather_df_from_json(path: str | Path) -> pd.DataFrame:
    """
    Προσπαθεί να «περπατήσει» ποικίλες δομές JSON και να παράξει
    rows: {timestamp, wind_speed, wind_dir}.
    """
    with open(path, "r") as f:
        data = json.load(f)

    rows: list[dict] = []

    def add_row(ts, h: dict):
        if ts is None or pd.isna(ts):
            return
        # επιλογή πιθανών κλειδιών για ταχύτητα/διεύθυνση
        ws = (h.get("windspeed") or h.get("windSpeed") or
              h.get("wind_speed") or h.get("speed") or h.get("wind_spd"))
        wd = (h.get("winddir") or h.get("windDir") or
              h.get("wind_dir") or h.get("dir") or h.get("wind_dir_deg"))
        rows.append({"timestamp": ts, "wind_speed": ws, "wind_dir": wd})

    def parse_epoch(val):
        """δέχεται epoch σε sec ή ms και επιστρέφει Timestamp"""
        try:
            ival = int(val)
        except Exception:
            return pd.NaT
        # heuristic: αν έχει 13 ψηφία ~ ms
        if ival > 10_000_000_000:  # > 10^10
            return pd.to_datetime(ival, unit="ms", errors="coerce")
        return pd.to_datetime(ival, unit="s", errors="coerce")

    def walk(obj):
        if isinstance(obj, dict):
            # 1) Ημερήσιες εγγραφές με "hours": [...]
            if "hours" in obj and isinstance(obj["hours"], list):
                day_date = (obj.get("datetime") or obj.get("date") or
                            obj.get("day") or obj.get("dateStr"))
                for h in obj["hours"]:
                    # Ώρα ως string ή πλήρες timestamp
                    time_str = h.get("datetime") or h.get("time")
                    ts = None
                    if time_str:
                        if day_date and "-" in str(day_date):  # YYYY-MM-DD?
                            ts = pd.to_datetime(f"{day_date} {time_str}", errors="coerce")
                        else:
                            ts = pd.to_datetime(time_str, errors="coerce")
                    if ts is None or pd.isna(ts):
                        epoch = h.get("datetimeEpoch") or h.get("epoch") or h.get("ts")
                        ts = parse_epoch(epoch) if epoch is not None else pd.NaT
                    add_row(ts, h)

            # 2) Fallback: παράλληλοι arrays time[], wind_speed[], wind_dir[]
            if {"time", "wind_speed", "wind_dir"}.issubset(set(obj.keys())):
                t = obj.get("time") or []
                ws = obj.get("wind_speed") or []
                wd = obj.get("wind_dir") or []
                n = min(len(t), len(ws), len(wd))
                for i in range(n):
                    ts = pd.to_datetime(t[i], errors="coerce")
                    add_row(ts, {"wind_speed": ws[i], "wind_dir": wd[i]})

            # συνέχισε την αναδρομή
            for v in obj.values():
                walk(v)

        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(data)

    if not rows:
        raise ValueError("Δεν βρέθηκαν ωριαίες εγγραφές. Άγνωστη μορφή JSON.")

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df

# ------------------ Pipeline steps ------------------
def load_weather_hourly(path: str | Path,
                        start: pd.Timestamp,
                        end: pd.Timestamp,
                        assume_units: str = "mph") -> pd.DataFrame:
    """Διαβάζει weather JSON, φιλτράρει [start,end] inclusive, και επιστρέφει ωριαίες εγγραφές."""
    w = weather_df_from_json(path)
    # inclusive φίλτρο
    mask = (w["timestamp"] >= start) & (w["timestamp"] <= end)
    w = w.loc[mask, ["timestamp", "wind_speed", "wind_dir"]].copy()

    # μονάδες ταχύτητας
    if assume_units.lower() == "mph":
        w["wind_speed"] = w["wind_speed"] * MPH_TO_MPS  # mph → m/s
    elif assume_units.lower() == "mps":
        pass  # ήδη m/s
    else:
        raise ValueError("assume_units must be 'mph' or 'mps'.")

    # ευθυγράμμιση ώρας ακριβώς στη hh:00:00
    w["timestamp"] = pd.to_datetime(w["timestamp"], errors="coerce").dt.floor("H")
    w = w.dropna(subset=["timestamp"]).rename(
        columns={"wind_speed": "wind_speed_wx", "wind_dir": "wind_dir_wx"}
    )
    return w

def prepare_scada_hourly(pref_df: pd.DataFrame) -> pd.DataFrame:
    """Resample SCADA σε ωριαίο: mean wind_speed/power, circular mean wind_dir."""
    scada = pref_df[["timestamp", "wind_speed", "wind_dir", "power"]].copy()
    scada["timestamp"] = pd.to_datetime(scada["timestamp"], errors="coerce")
    scada = scada.dropna(subset=["timestamp"])
    # Ωριαίο resample με circular mean στη διεύθυνση
    scada_h1 = (scada
        .set_index("timestamp")
        .resample("1H")
        .agg({
            "wind_speed": "mean",
            "power": "mean",
            "wind_dir": circular_mean_deg
        })
        .reset_index()
        .rename(columns={
            "wind_speed": "wind_speed_scada_h1",
            "power": "power_h1",
            "wind_dir": "wind_dir_scada_h1"
        })
    )
    return scada_h1

def merge_hourly(scada_h1: pd.DataFrame, wx_h1: pd.DataFrame) -> pd.DataFrame:
    """Inner-merge στο ίδιο ωριαίο timestamp."""
    merged = (scada_h1
              .merge(wx_h1, on="timestamp", how="inner")
              .dropna(subset=["wind_speed_scada_h1", "wind_dir_scada_h1",
                              "wind_speed_wx", "wind_dir_wx", "power_h1"]))
    return merged

def fit_height_correction(merged_h1: pd.DataFrame,
                          alphas: np.ndarray,
                          hub_height: float,
                          z_wx: float,
                          train_cutoff: pd.Timestamp):
    """Grid search για alpha (power-law) + Huber fine-tune σε train split."""
    if merged_h1.empty:
        raise ValueError("Merged SCADA×WX είναι κενό — δεν μπορώ να εκπαιδεύσω.")
    train_mask = merged_h1["timestamp"] < train_cutoff
    y_tr = merged_h1.loc[train_mask, "wind_speed_scada_h1"].to_numpy()
    x_tr = merged_h1.loc[train_mask, "wind_speed_wx"].to_numpy()

    if y_tr.size == 0:
        # αν δεν υπάρχει train split, χρησιμοποίησε όλο το δείγμα
        y_tr = merged_h1["wind_speed_scada_h1"].to_numpy()
        x_tr = merged_h1["wind_speed_wx"].to_numpy()

    # grid για alpha
    best_alpha, best_rmse = None, np.inf
    for a in alphas:
        v_adj = height_correct(x_tr, a, hub_height, z_wx)
        rmse = np.sqrt(mean_squared_error(y_tr, v_adj))
        if rmse < best_rmse:
            best_alpha, best_rmse = float(a), float(rmse)

    # robust linear fine-tune (Huber) πάνω στο height-corrected
    Xtr2 = height_correct(x_tr.reshape(-1, 1), best_alpha, hub_height, z_wx)
    huber = HuberRegressor().fit(Xtr2, y_tr)

    return best_alpha, best_rmse, huber

def apply_height_models(merged_h1: pd.DataFrame,
                        alpha: float,
                        huber: HuberRegressor,
                        hub_height: float,
                        z_wx: float) -> pd.DataFrame:
    """Προσθέτει στήλες wind_speed_wx_hub (power-law μόνο) και wind_speed_wx_cal (power-law + Huber)."""
    v_wx = merged_h1["wind_speed_wx"].to_numpy()
    v_hub = height_correct(v_wx, alpha, hub_height, z_wx)
    v_cal = huber.intercept_ + huber.coef_[0] * v_hub
    df = merged_h1.copy()
    df["wind_speed_wx_hub"] = np.clip(v_hub, 0, None)
    df["wind_speed_wx_cal"] = np.clip(v_cal, 0, None)
    return df

def compute_metrics(merged_h1: pd.DataFrame):
    """Υπολογίζει bias/MAE/RMSE/corr για ταχύτητα, και bias/MAE/RMSE για διεύθυνση."""
    if merged_h1.empty:
        return {"n_matches": 0}

    out = {"n_matches": int(len(merged_h1))}

    # Διεύθυνση
    d_dir = circular_diff_deg(merged_h1["wind_dir_scada_h1"], merged_h1["wind_dir_wx"])
    out.update({
        "dir_bias_deg": float(np.nanmean(d_dir)),
        "dir_MAE_deg":  float(np.nanmean(np.abs(d_dir))),
        "dir_RMSE_deg": float(np.sqrt(np.nanmean(d_dir**2))),
    })

    # Ταχύτητα: raw WX, hub-corrected, calibrated
    for col in ["wind_speed_wx", "wind_speed_wx_hub", "wind_speed_wx_cal"]:
        if col not in merged_h1.columns:
            continue
        d_speed = merged_h1["wind_speed_scada_h1"] - merged_h1[col]
        out.update({
            f"{col}:speed_bias": float(np.nanmean(d_speed)),
            f"{col}:speed_MAE":  float(np.nanmean(np.abs(d_speed))),
            f"{col}:speed_RMSE": float(np.sqrt(np.nanmean(d_speed**2))),
            f"{col}:speed_corr": float(np.corrcoef(
                merged_h1["wind_speed_scada_h1"].fillna(np.nan),
                merged_h1[col].fillna(np.nan)
            )[0,1]),
        })
    return out

# ---------------------- Main run --------------------
# 1) Weather
wx = load_weather_hourly(WEATHER_JSON, START, END, assume_units=ASSUME_WX_UNITS)

# 2) SCADA (από το δικό σου pref_df που έχεις ήδη υπολογίσει πιο πριν)
#    Αν το pref_df δεν έχει δημιουργηθεί, βάλ’ το πριν από αυτό το script.
scada_h1 = prepare_scada_hourly(pref_df)

# 3) Merge
merged_h1 = merge_hourly(scada_h1, wx)
print(f"[INFO] hourly matches: {len(merged_h1)}")
if merged_h1.empty:
    raise SystemExit("[ERROR] No hourly matches between SCADA and Weather.")

# 4) Height correction (grid + Huber) και εφαρμογή στα δεδομένα
alpha_star, train_rmse, hub_model = fit_height_correction(
    merged_h1, ALPHAS, HUB_HEIGHT, Z_WX, TRAIN_CUTOFF
)
print(f"[height-corr] alpha* = {alpha_star:.3f} (train RMSE={train_rmse:.3f} m/s)")

merged_h1 = apply_height_models(merged_h1, alpha_star, hub_model, HUB_HEIGHT, Z_WX)

# 5) Metrics
metrics = compute_metrics(merged_h1)
print("[metrics]", metrics)

# 6) (προαιρετικά) δείξε εύρος/δείγμα
print(merged_h1["timestamp"].min(), "→", merged_h1["timestamp"].max())
print(merged_h1.head())

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import  mean_absolute_error, r2_score #mean_squared_error --> import already done

# 0) Start from your merged_h1
df = merged_h1.copy()

# 1) Features (speed & direction vectors) / Target 
FEATURES = ["wind_speed_scada_h1", "wind_x", "wind_y"] # ,"wind_speed", "wind_x", "wind_y"]
TARGET   = "power_h1"

# If wind_x/y are missing, compute them (uses wind_dir in degrees if present)
if any(c not in df.columns for c in ["wind_x","wind_y"]):
    if "wind_speed_scada_h1" in df.columns and "wind_dir_scada_h1" in df.columns:
        wd = np.deg2rad(pd.to_numeric(df["wind_dir_scada_h1"], errors="coerce"))
        ws = pd.to_numeric(df["wind_speed_scada_h1"], errors="coerce")
        df["wind_x"] = ws * np.cos(wd)
        df["wind_y"] = ws * np.sin(wd)
    else:
        raise KeyError("wind_x/wind_y not present and cannot be computed (need wind_speed & wind_dir).")

# Build X, y
X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(df[TARGET], errors="coerce")

# drop rows with any NaNs in features or target
valid = ~(X.isna().any(axis=1) | y.isna())
X = X.loc[valid].to_numpy()
y = y.loc[valid].to_numpy()
timestamps = pd.to_datetime(df.loc[valid, "timestamp"]) if "timestamp" in df.columns else pd.Index(np.arange(len(y)))


# 2) Chronological split (80/20) - no Shuffle
n = len(y)
n_train = int(0.80 * n)

X_train = X[:n_train]
y_train = y[:n_train]
t_train = timestamps[:n_train]

X_test  = X[n_train:] 
y_test  = y[n_train:] 
t_test  = timestamps[n_train:] 

print(f"x train: {X_train.shape} | y train: {y_train.shape}")
print(f"x test : {X_test.shape}  | y test : {y_test.shape}")

# 3) Metrics helper
def regression_metrics(y_true, y_pred, prefix=""):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    rng  = (np.max(y_true) - np.min(y_true)) or np.nan
    nmae  = mae  / rng if np.isfinite(rng) and rng > 0 else np.nan
    nrmse = rmse / rng if np.isfinite(rng) and rng > 0 else np.nan
    out = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "NMAE": nmae, "NRMSE": nrmse}
    print((prefix + " " if prefix else "") + f"metrics: {out}")
    return out

# 4) Base model and grid
base = xgb.XGBRegressor(
    objective='reg:squarederror',
    tree_method='hist',
    random_state=42
)

param_grid = {
    'learning_rate': [0.1, 0.05, 0.01],
    'n_estimators':  [300, 600, 1000],
    'subsample':     [0.4, 0.6, 0.8],
    'max_depth':     [4, 6, 8]
}

def model_validate(model, param_grid, x_train, y_train, 
                   model_name, k_folds=4, scoring='neg_mean_squared_error'): 
    # 1-D target
    y_train = np.asarray(y_train).ravel()
    
    # Time-series-safe CV
    cv = TimeSeriesSplit(n_splits=k_folds)

    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        refit=True,
        n_jobs=-1,
        verbose=0
    )
    gs.fit(x_train, y_train)

    print(f'[{model_name}] Best params: {gs.best_params_}')
    print(f'[{model_name}] CV MSE: {-gs.best_score_:.4f}')
    
    return gs

# 5) Train + Hyperparameter tuning with CV and model selection
xgb_tuned = model_validate(
    base, param_grid,
    X_train, y_train,
    model_name='XGBR', k_folds=5
)

xgb_model = xgb_tuned.best_estimator_

# Τελική αξιολόγηση στο TEST (εκτός GridSearch)
y_test_pred = xgb_model.predict(X_test)

regression_metrics(y_test, y_test_pred, prefix="TEST")

# 7) Plots
# (a) True vs Pred scatter (TEST)
plt.figure(figsize=(7,6))
plt.scatter(y_test, y_test_pred, s=10, alpha=0.6)
lim = [min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())]
plt.plot(lim, lim, "r--", linewidth=1)
plt.title("TEST: True vs Predicted")
plt.xlabel("True Power"); plt.ylabel("Predicted Power")
plt.tight_layout(); plt.show()

# (b) Time series (TEST)
plt.figure(figsize=(11,4))
plt.plot(t_test, y_test, label="True", linewidth=1)
plt.plot(t_test, y_test_pred, label="Pred", linewidth=1)
plt.title("TEST: Power time series")
plt.legend(); plt.tight_layout(); plt.show()
