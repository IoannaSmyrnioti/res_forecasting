from res_forecasting.helpers.wf_data_handler import WFDataHandler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

from res_forecasting.data.preprocessing.cleaning1 import clean_scada_data
from res_forecasting.data.preprocessing.filtering import apply_prefilters


with WFDataHandler("wf_data.duckdb") as wf_db_handler:
    scada_df = wf_db_handler.get_df()

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
    "power_setpoint","pitch_a","pitch_b","pitch_c"
]

cleaned_df, report = clean_scada_data(
     scada_df,
     col_map=col_map,
     features_to_keep=features_to_keep,
     dropna=True,                # keep your behavior
     basic_filter=True,
     drop_negative_power=True,
     min_wind_speed=None,
     make_wind_vectors=True,
     drop_wind_dir=False,
     verbose=True,               # prints your report
     pitch_bounds=(0.0, 95.0),
)

Pr = 2050.0  # kW
pref_df, summary = apply_prefilters(
    cleaned_df,
    Pr=Pr,
    persist_bins=2,
    min_setpoint_frac=0.10,   # ignore very low setpoints
    obs_tol=0.10,             # power within 10% of setpoint → binding
    # force_theta=30.0,       # optional: mimic paper for a sanity check
    theta_bounds=(22.0, 45.0) # keep θ* in a sensible band
)

print(pref_df.head())
