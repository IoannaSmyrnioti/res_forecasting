import numpy as np
import pandas as pd

def clean_scada_data(df, dropna=True, features_to_keep=None, verbose=True):
    """
    Basic SCADA cleaning function.

    Parameters:
        df (pd.DataFrame): Raw SCADA data
        dropna (bool): If True, drop rows with any NaNs
        features_to_keep (list): List of features to keep. If None, keeps all.
        verbose (bool): If True, print diagnostics

    Returns:
        pd.DataFrame: Cleaned dataframe
    """

    if features_to_keep:
        df = df[features_to_keep]

    original_shape = df.shape

    if verbose:
        print(f"Selected features: {df.columns.tolist()}")
        print(f"Original shape: {df.shape}")
        print("NaN counts:\n", df.isna().sum()[df.isna().sum() > 0])

    if dropna:
        df = df.dropna()
        rows_dropped = original_shape[0] - df.shape[0]
        percent_dropped = (rows_dropped / original_shape[0]) * 100
        
        if verbose:
            print(f"\n Dropped {rows_dropped} rows with NaNs ({percent_dropped:.2f}% of data)")
            print(f"After dropping NaNs: {df.shape}")

    return df

# Example usage:
# from res_forecasting.data.preprocessing.cleaning import clean_scada_data

# essential_features = [
#    "wind_speed_ms",
#    "wind_direction",
#    "power_kw"
# ]

# cleaned_df = clean_scada_data(scada_df, dropna=True, features_to_keep=essential_features, verbose=True)