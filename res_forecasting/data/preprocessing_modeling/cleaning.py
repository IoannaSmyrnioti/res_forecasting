import pandas as pd
import numpy as np

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    # optional: normalize column labels to avoid hidden spaces / case mismatches
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
    )
    return df

def prepare_col_map(df_cols: list[str], col_map: dict | None) -> dict:
    """
    Ensure col_map is in the correct direction for pandas.rename:
    {existing_name -> new_standard_name}.
    If the user passed {standard -> existing}, flip it.
    """
    if not col_map:
        return {}

    df_set = set(df_cols)
    keys_in_df = [k in df_set for k in col_map.keys()]
    vals_in_df = [v in df_set for v in col_map.values()]

    if any(keys_in_df) and not any(vals_in_df):
        # Looks correct already: keys match df columns
        return col_map
    if any(vals_in_df) and not any(keys_in_df):
        # Looks reversed: values match df columns, flip it
        return {v: k for k, v in col_map.items()}

    # Ambiguous or neither matches; be explicit about the problem
    raise ValueError(
        "col_map does not match your DataFrame columns. "
        "Pass mapping as {existing_df_name: standard_name}. "
        f"DataFrame columns: {sorted(df_cols)} | col_map: {col_map}"
    )

def log_step(step_name: str, before: int, after: int, steps_list: list, initial_total: int):
    """Log step name, dropped count, and percentages."""
    dropped = before - after
    pct_prev = round(100 * dropped / before, 2) if before else 0.0
    pct_orig = round(100 * dropped / initial_total, 2) if initial_total else 0.0
    steps_list.append((step_name, dropped, pct_prev, pct_orig))

def clean_scada_data(
    df: pd.DataFrame,
    *,
    # schema awareness (prefer {existing_df_name: standard_name})
    col_map: dict | None = None,
    timestamp_key: str = "timestamp",
    wind_speed_key: str = "wind_speed",
    wind_dir_key: str = "wind_dir",
    power_key: str = "power",
    # behavior
    features_to_keep: list[str] | None = None,
    dropna: bool = True,
    basic_filter: bool = True,
    min_wind_speed: float | None = None,
    max_wind_speed: float | None = None,  
    make_wind_vectors: bool = True,
    drop_wind_dir: bool = False,
    verbose: bool = True,
    pitch_bounds: tuple[float, float] = (0.0 , 95.0)
):
    """
    Enhanced SCADA cleaning with per-step reporting and wind direction vectorization.
    Returns (clean_df, report_dict).
    """
    dfc = normalize_cols(df)
    n0 = len(dfc)
    steps = []

    # 0) Optional rename for schema consistency
    if col_map:
        rename_map = prepare_col_map(dfc.columns.tolist(), col_map)
        dfc = dfc.rename(columns=rename_map)
    
    # 0.5) Keep only requested features
    if features_to_keep:
        missing = [c for c in features_to_keep if c not in dfc.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")
        dfc = dfc[features_to_keep]

    # 1) Drop rows where data_availability is 0
    if "data_availability" in dfc.columns:
        before = len(dfc)
        dfc = dfc[dfc["data_availability"] != 0]
        log_step("drop_zero_data_availability", before, len(dfc), steps, n0)
    
    # 2) Clip obvious ranges
    if "power_setpoint" in dfc.columns:
        dfc["power_setpoint"] = dfc["power_setpoint"].clip(lower=0)

    for c in ("pitch_a", "pitch_b", "pitch_c"):
        if c in dfc.columns:
            dfc[c] = dfc[c].clip(lower=pitch_bounds[0], upper=pitch_bounds[1])

    # 3) Derive once; use later
    if {"pitch_a","pitch_b","pitch_c"}.issubset(dfc.columns):
        dfc["pitch_max"]    = dfc[["pitch_a","pitch_b","pitch_c"]].max(axis=1,  skipna=True)
        # Drop the original pitch columns, keep only pitch_max
        dfc = dfc.drop(columns=["pitch_a", "pitch_b", "pitch_c"])
        
    # 4) Drop NaNs early
    if dropna:
        before = len(dfc)
        dfc = dfc.dropna()
        log_step("dropna_initial", before, len(dfc), steps, n0)

    # 5) Drop duplicate timestamps
    if timestamp_key in dfc.columns:
        before = len(dfc)
        dfc = dfc.drop_duplicates(subset=[timestamp_key])
        log_step("duplicate_timestamps", before, len(dfc), steps, n0)

    # 6) Basic range filters
    if basic_filter:
        before = len(dfc)
        cond = pd.Series(True, index=dfc.index)
        if wind_speed_key in dfc.columns and not (min_wind_speed is not None and max_wind_speed is not None):
            cond &= dfc[wind_speed_key].between(0, 30)  # keep valid wind speed values (m/s)
        if power_key in dfc.columns:
            cond &= dfc[power_key] >= 0 # keep non-negative power values
        dfc = dfc[cond]
        log_step("basic_range_filters", before, len(dfc), steps, n0)

    # 7) Minimum and maximum wind speed filter
    if (min_wind_speed is not None or max_wind_speed is not None) and wind_speed_key in dfc.columns:
        before = len(dfc)
        cond = pd.Series(True, index=dfc.index)
        if min_wind_speed is not None:
            cond &= dfc[wind_speed_key] >= min_wind_speed
        if max_wind_speed is not None:
            cond &= dfc[wind_speed_key] <= max_wind_speed
        dfc = dfc[cond]
        log_step("wind_speed_cut_in_out_filter", before, len(dfc), steps, n0)

    # 8) Final NaN sweep
    before = len(dfc)
    dfc = dfc.dropna()
    log_step("dropna_final", before, len(dfc), steps, n0)

    # 9) Vectorize wind direction
    if make_wind_vectors and wind_dir_key in dfc.columns:
        wd_rad = np.deg2rad(dfc[wind_dir_key])
        dfc["wind_x"] = dfc[wind_speed_key] * np.cos(wd_rad)
        dfc["wind_y"] = dfc[wind_speed_key] * np.sin(wd_rad)
        if drop_wind_dir:
            dfc = dfc.drop(columns=[wind_dir_key])

    # Report
    report = {
        "initial_rows": n0,
        "final_rows": len(dfc),
        "removed_rows": n0 - len(dfc),
        "removed_pct": ((n0 - len(dfc)) / n0) * 100 if n0 else 0,
        "steps": steps,
        "params": {
            "features_kept": features_to_keep,
            "timestamp_key": timestamp_key,
            "wind_speed_key": wind_speed_key,
            "wind_dir_key": wind_dir_key,
            "power_key": power_key,
            "dropna": dropna,
            "basic_filter": basic_filter,
            "min_wind_speed": min_wind_speed,
            "max_wind_speed": max_wind_speed,
            "make_wind_vectors": make_wind_vectors,
            "drop_wind_dir": drop_wind_dir,
            "col_map": col_map,
            "pitch_bounds": pitch_bounds
        },
    }

    if verbose:
        print("\nClean report:")
        print(report)

    return dfc, report

