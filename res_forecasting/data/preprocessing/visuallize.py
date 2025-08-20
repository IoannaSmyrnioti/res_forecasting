import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

def _pick_pitch_col(df):
    # prefer pitch_max (conservative), else pitch_mean
    if "pitch_max" in df.columns: return "pitch_max"
    if "pitch_mean" in df.columns: return "pitch_mean"
    raise KeyError("Need pitch_max or pitch_mean in the DataFrame.")

def plot_3d_wind_pitch_power_kept_removed(
    df,
    *,
    title="",
    wind_col="wind_speed",
    power_col="power",
    theta=None,               # optional: show pitch cutoff θ* as a plane
    sample_every=1,           # plot every Nth row to thin dense datasets
):
    pitch_col = _pick_pitch_col(df)
    d = df[[wind_col, pitch_col, power_col, "prefilter_remove"]].dropna().copy()
    if sample_every > 1:
        d = d.iloc[::sample_every, :]

    kept = d.loc[~d["prefilter_remove"]]
    rem  = d.loc[d["prefilter_remove"]]

    fig = plt.figure(figsize=(9, 6.5))
    ax  = fig.add_subplot(111, projection="3d")

    # Removed first (lighter alpha), then kept on top
    ax.scatter(rem[wind_col], rem[pitch_col], rem[power_col], s=6, alpha=0.35, label="Removed")
    ax.scatter(kept[wind_col], kept[pitch_col], kept[power_col], s=6, alpha=0.8,  label="Kept")

    ax.set_xlabel("Wind speed (m/s)")
    ax.set_ylabel("Pitch angle (°)")
    ax.set_zlabel("Power (kW)")

    ttl = title
    if theta is not None:
        # add translucent plane at pitch = θ*
        xs = np.linspace(d[wind_col].min(), d[wind_col].max(), 2)
        zs = np.linspace(d[power_col].min(), d[power_col].max(), 2)
        X, Z = np.meshgrid(xs, zs)
        Y = np.full_like(X, float(theta))
        ax.plot_surface(X, Y, Z, alpha=0.15)
        ttl = f"{title}   (θ* ≈ {theta:.1f}°)"
    ax.set_title(ttl)

    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

def plot_3d_wind_pitch_power(
    df,
    *,
    title="",
    wind_col="wind_speed",
    power_col="power",
    sample_every=1
):
    pitch_col = _pick_pitch_col(df)
    d = df[[wind_col, pitch_col, power_col]].dropna().copy()
    if sample_every > 1:
        d = d.iloc[::sample_every, :]

    fig = plt.figure(figsize=(9, 6.5))
    ax  = fig.add_subplot(111, projection="3d")
    ax.scatter(d[wind_col], d[pitch_col], d[power_col], s=6, alpha=0.8)

    ax.set_xlabel("Wind speed (m/s)")
    ax.set_ylabel("Pitch angle (°)")
    ax.set_zlabel("Power (kW)")
    ax.set_title(title)

    plt.tight_layout()
    plt.show()