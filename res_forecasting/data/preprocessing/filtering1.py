import numpy as np, pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

def _persistent(flag: pd.Series, run_len:int=2) -> pd.Series:
    if run_len <= 1: return flag.astype(bool)
    x = flag.astype(bool).to_numpy(dtype=int)
    k = np.ones(run_len, dtype=int)
    y = np.convolve(x, k, mode="same") >= run_len
    return pd.Series(y, index=flag.index)

def _auto_theta_gmm(pitch_mean: pd.Series, min_n=200, max_components=3, seed=0):
    x = pd.to_numeric(pitch_mean, errors="coerce").dropna().to_numpy().reshape(-1,1)
    if len(x) < min_n: return None, {"reason":"too_few_samples"}
    best=None
    for k in range(1, max_components+1):
        g = GaussianMixture(n_components=k, covariance_type="full", n_init=5, random_state=seed).fit(x)
        bic = g.bic(x)
        best = (k,g,bic) if best is None or bic < best[2] else best
    k,gmm,_ = best
    if k==1: return None, {"reason":"unimodal_pitch"}
    order = np.argsort(gmm.means_.ravel())
    mu = gmm.means_.ravel()[order]
    var = np.array([gmm.covariances_[i].ravel()[0] for i in order])
    w   = gmm.weights_.ravel()[order]
    def _intersect(m1,s1,w1,m2,s2,w2):
        a = 1/(2*s1) - 1/(2*s2); b = m2/s2 - m1/s1
        c = (m1*m1)/(2*s1) - (m2*m2)/(2*s2) + np.log((w1*np.sqrt(s2))/(w2*np.sqrt(s1)))
        return np.roots([a,b,c]).real
    cands=[]
    for i in range(k-1):
        r = _intersect(mu[i],var[i],w[i],mu[i+1],var[i+1],w[i+1])
        mid=0.5*(mu[i]+mu[i+1]); in_between=[z for z in r if mu[i]<=z<=mu[i+1]]
        x_star = in_between[0] if in_between else r[np.argmin(np.abs(r-mid))]
        cands.append(x_star)
    if not cands or not np.isfinite(cands).all(): return None, {"reason":"no_intersection"}
    def _mix_pdf(z):
        s=np.sqrt(var); return np.sum(w*(1/(np.sqrt(2*np.pi)*s))*np.exp(-0.5*((z-mu)/s)**2))
    theta = float(cands[int(np.argmin([_mix_pdf(z) for z in cands]))])
    return theta, {"method":"gmm","theta":theta}

def _auto_theta_knee(pitch_mean: pd.Series, power_kw: pd.Series, Pr: float,
                     bin_deg=1.0, min_pts_per_bin=20, min_n=200, smooth_window=3):
    df = pd.DataFrame({"pitch": pd.to_numeric(pitch_mean, errors="coerce"),
                       "pfrac": pd.to_numeric(power_kw, errors="coerce")/float(Pr)}).dropna()
    if len(df) < min_n: return None, {"reason":"too_few_samples"}
    bins = np.arange(np.floor(df.pitch.min()), np.ceil(df.pitch.max())+bin_deg, bin_deg)
    idx  = np.digitize(df.pitch, bins)
    centers, med = [], []
    for b in np.unique(idx):
        vals = df.pfrac[idx==b]
        if len(vals) >= min_pts_per_bin:
            med.append(float(np.median(vals)))
            centers.append((bins[b-1] + (bins[b] if b < len(bins) else bins[-1]))/2.0)
    if not med: return None, {"reason":"insufficient_bins"}
    centers = np.asarray(centers); med = np.asarray(med)
    med_s = pd.Series(med).rolling(smooth_window, center=True, min_periods=1).median().to_numpy()
    med_env = np.maximum.accumulate(med_s[::-1])[::-1]
    x = (centers - centers.min())/(centers.ptp()+1e-12)
    y = (med_env - med_env.min())/(med_env.ptp()+1e-12)
    g = y - x
    theta = float(centers[int(np.argmax(g))])
    return theta, {"method":"knee","theta":theta,"bins_used":int(len(centers))}

def _type2_mask_observed(df, Pr: float,
                         min_setpoint_frac=0.10,   # ignore tiny setpoints
                         obs_tol=0.10,             # binding: power within 10% of setpoint
                         persist_bins=2):
    sp = pd.to_numeric(df["power_setpoint"], errors="coerce")
    pw = pd.to_numeric(df["power"], errors="coerce")
    t2_plain = (sp < 0.99*Pr) & (sp >= min_setpoint_frac*Pr)
    bind_obs = pw >= sp * (1 - obs_tol)
    return _persistent(t2_plain & bind_obs, run_len=persist_bins)

def apply_prefilters_knee(df, *, Pr: float,
                          persist_bins=2,
                          theta_bounds=(22.0,45.0),
                          min_setpoint_frac=0.10,
                          obs_tol=0.10):
    out = df.copy()
    theta, info = _auto_theta_knee(out["pitch_mean"], out["power"], Pr)
    if theta is not None:
        theta = float(np.clip(theta, *theta_bounds))
        t1 = _persistent(out["pitch_max"] > theta, run_len=persist_bins)
    else:
        t1 = pd.Series(False, index=out.index)
    t2 = _type2_mask_observed(out, Pr, min_setpoint_frac, obs_tol, persist_bins)
    out["Type1"], out["Type2"] = t1, t2
    out["prefilter_remove"] = t1 | t2
    n_before = len(out); n_after = int((~out["prefilter_remove"]).sum())
    return out, {
        "mode":"knee_only","theta":None if theta is None else float(theta),
        "theta_method": info.get("method") if theta is not None else None,
        "before":n_before,"after":n_after,
        "removed":n_before-n_after,
        "pct_removed":0 if n_before==0 else 100*(n_before-n_after)/n_before
    }

def apply_prefilters_gmm(df, *, Pr: float,
                         persist_bins=2,
                         theta_bounds=(22.0,45.0),
                         min_setpoint_frac=0.10,
                         obs_tol=0.10):
    out = df.copy()
    theta, info = _auto_theta_gmm(out["pitch_mean"])
    if theta is not None:
        theta = float(np.clip(theta, *theta_bounds))
        t1 = _persistent(out["pitch_max"] > theta, run_len=persist_bins)
    else:
        t1 = pd.Series(False, index=out.index)
    t2 = _type2_mask_observed(out, Pr, min_setpoint_frac, obs_tol, persist_bins)
    out["Type1"], out["Type2"] = t1, t2
    out["prefilter_remove"] = t1 | t2
    n_before = len(out); n_after = int((~out["prefilter_remove"]).sum())
    return out, {
        "mode":"gmm_only","theta":None if theta is None else float(theta),
        "theta_method": info.get("method") if theta is not None else None,
        "before":n_before,"after":n_after,
        "removed":n_before-n_after,
        "pct_removed":0 if n_before==0 else 100*(n_before-n_after)/n_before
    }

def compare_prefilter_methods(cleaned_df, Pr: float,
                              persist_bins=2, min_setpoint_frac=0.10, obs_tol=0.10,
                              theta_bounds=(22.0,45.0)):
    knee_df, ksum = apply_prefilters_knee(cleaned_df, Pr=Pr, persist_bins=persist_bins,
                                          theta_bounds=theta_bounds,
                                          min_setpoint_frac=min_setpoint_frac, obs_tol=obs_tol)
    gmm_df,  gsum = apply_prefilters_gmm(cleaned_df, Pr=Pr, persist_bins=persist_bins,
                                         theta_bounds=theta_bounds,
                                         min_setpoint_frac=min_setpoint_frac, obs_tol=obs_tol)
    comp = pd.DataFrame([
        {"method":"knee", **{k:v for k,v in ksum.items() if k in ["theta","theta_method","before","after","removed","pct_removed"]}},
        {"method":"gmm",  **{k:v for k,v in gsum.items() if k in ["theta","theta_method","before","after","removed","pct_removed"]}},
    ])
    return knee_df, gmm_df, comp

def plot_3d_kept_removed(df, title, theta=None,
                         wind_col="wind_speed", power_col="power"):
    pitch_col = "pitch_max" if "pitch_max" in df.columns else "pitch_mean"
    d = df[[wind_col, pitch_col, power_col, "prefilter_remove"]].dropna()
    fig = plt.figure(figsize=(9, 6.5)); ax = fig.add_subplot(111, projection="3d")
    rem  = d.loc[d["prefilter_remove"]]; kept = d.loc[~d["prefilter_remove"]]
    ax.scatter(rem[wind_col],  rem[pitch_col],  rem[power_col],  s=6, alpha=0.35, label="Removed")
    ax.scatter(kept[wind_col], kept[pitch_col], kept[power_col], s=6, alpha=0.85, label="Kept")
    if theta is not None:
        xs = np.linspace(d[wind_col].min(), d[wind_col].max(), 2)
        zs = np.linspace(d[power_col].min(), d[power_col].max(), 2)
        X, Z = np.meshgrid(xs, zs); Y = np.full_like(X, float(theta))
        ax.plot_surface(X, Y, Z, alpha=0.15); title = f"{title}  (θ ≈ {theta:.1f}°)"
    ax.set_xlabel("Wind speed (m/s)"); ax.set_ylabel("Pitch (°)"); ax.set_zlabel("Power (kW)")
    ax.set_title(title); ax.legend(loc="upper left"); plt.tight_layout(); plt.show()