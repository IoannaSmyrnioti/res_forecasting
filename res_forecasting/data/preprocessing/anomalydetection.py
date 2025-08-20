import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor


# ---------- utils ----------
def ensure_wind_vectors(df, wind_speed_col, wind_dir_col):
    d = df.copy()
    if "wind_x" in d.columns and "wind_y" in d.columns:
        return d, "wind_x", "wind_y"
    theta = np.deg2rad((d[wind_dir_col].astype(float)) % 360.0)
    d["wind_x"] = d[wind_speed_col].astype(float) * np.cos(theta)
    d["wind_y"] = d[wind_speed_col].astype(float) * np.sin(theta)
    return d, "wind_x", "wind_y"


def build_X(df, wind_speed_col, wind_dir_col, power_col, include_speed=True):
    d, xcol, ycol = ensure_wind_vectors(df, wind_speed_col, wind_dir_col)
    cols = [xcol, ycol, power_col]
    if include_speed:
        cols = [wind_speed_col] + cols
    X = d[cols].astype(float).to_numpy()
    return X, cols, d


# ---------- GMM ----------
def gmm_select_k_by_bic(X_scaled, k_min=1, k_max=8, random_state=42):
    bics = []
    models = []
    for k in range(k_min, k_max + 1):
        g = GaussianMixture(n_components=k, covariance_type="full", random_state=random_state)
        g.fit(X_scaled)
        bics.append(g.bic(X_scaled))
        models.append(g)
    k_star = np.argmin(bics) + k_min
    return k_star, bics, models[np.argmin(bics)]


def gmm_anomaly_flags(df, wind_speed_col="wind_speed", wind_dir_col="wind_dir",
                            power_col="power", include_speed=True, k_min=1, k_max=8,
                            random_state=42, threshold_rule="boxplot"):
    # features + scale
    X, feat_cols, d = build_X(df, wind_speed_col, wind_dir_col, power_col, include_speed)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # BIC selection
    k_star, bics, gmm = gmm_select_k_by_bic(Xs, k_min=k_min, k_max=k_max, random_state=random_state)

    # responsibilities (posterior probs per component); take max per point (paper)
    resp = gmm.predict_proba(Xs)                 # shape (n, k)
    max_resp = resp.max(axis=1)                  # affinity to closest Gaussian

    if threshold_rule == "boxplot":
        q1, q3 = np.percentile(max_resp, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        flags = max_resp < lower                 # low affinity => anomaly
        thr = lower
        score = max_resp                         # higher = more normal
        score_name = "gmm_max_resp"
    elif threshold_rule == "3sigma":
        # alternative: use negative log-likelihood and 3-sigma high tail
        nll = -gmm.score_samples(Xs)
        mu, sd = nll.mean(), nll.std(ddof=0)
        thr = mu + 3 * sd
        flags = nll > thr
        score = -nll                             # higher = more normal
        score_name = "gmm_minus_nll"
    else:
        raise ValueError("threshold_rule must be 'boxplot' or '3sigma'")

    out = df.copy()
    out["is_anomaly_gmm"] = flags
    out[score_name] = score
    return out, {"k_star": k_star, "bics": bics, "features": feat_cols, "threshold": float(thr)}


# ---------- LOF ----------
def lof_anomaly_flags(df, wind_speed_col="wind_speed", wind_dir_col="wind_dir",
                            power_col="power", include_speed=True, minpts=700, lof_threshold=1.5):
    X, feat_cols, d = build_X(df, wind_speed_col, wind_dir_col, power_col, include_speed)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # LOF returns negative_outlier_factor_ ~ -LOF
    lof = LocalOutlierFactor(n_neighbors=minpts, contamination="auto")
    preds = lof.fit_predict(Xs)                             # -1 outlier, +1 inlier (sklearn internal)
    neg_scores = lof.negative_outlier_factor_
    lof_vals = -neg_scores                                  # LOF value: ~1 normal, >1.5 outlier (paper)

    flags = lof_vals > lof_threshold

    out = df.copy()
    out["is_anomaly_lof"] = flags
    out["lof_value"] = lof_vals
    return out, {"features": feat_cols, "minpts": minpts, "lof_threshold": float(lof_threshold)}


# ---------- Compare & Plot ----------
def summarize_overlap(gmm_flags: np.ndarray, lof_flags: np.ndarray):
    gmm_n = int(gmm_flags.sum())
    lof_n  = int(lof_flags.sum())
    both   = int((gmm_flags & lof_flags).sum())
    either = int((gmm_flags | lof_flags).sum())
    return {
        "gmm_anomalies": gmm_n,
        "lof_anomalies": lof_n,
        "overlap": both,
        "either": either,
        "overlap_pct_of_either": round(100 * both / either, 2) if either else 0.0
    }


def plot_gmm_lof_panels(df, wind_speed_col="wind_speed", power_col="power",
                        gmm_flags=None, lof_flags=None, sample=20000, title="Anomaly comparison"):
    d = df[[wind_speed_col, power_col]].copy()
    n = len(d)
    idx = np.arange(n)
    if n > sample:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(idx, size=sample, replace=False))
    d = d.iloc[idx]
    gf = gmm_flags[idx] if gmm_flags is not None else None
    lf = lof_flags[idx]  if lof_flags  is not None else None

    plt.figure(figsize=(14, 6))

    ax1 = plt.subplot(1, 2, 1)
    ax1.scatter(d[wind_speed_col], d[power_col], s=6, alpha=0.25)
    if gf is not None:
        ax1.scatter(d.loc[gf, wind_speed_col], d.loc[gf, power_col], s=8, alpha=0.9)
    ax1.set_title("GMM anomalies")
    ax1.set_xlabel("Wind speed (m/s)")
    ax1.set_ylabel("Power (kW)")
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(1, 2, 2)
    ax2.scatter(d[wind_speed_col], d[power_col], s=6, alpha=0.25)
    if lf is not None:
        ax2.scatter(d.loc[lf, wind_speed_col], d.loc[lf, power_col], s=8, alpha=0.9)
    ax2.set_title("LOF anomalies")
    ax2.set_xlabel("Wind speed (m/s)")
    ax2.set_ylabel("Power (kW)")
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()