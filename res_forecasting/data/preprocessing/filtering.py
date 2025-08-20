import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

def _persistent(flag: pd.Series, run_len:int=2) -> pd.Series:
    if run_len <= 1: return flag.astype(bool)
    x = flag.astype(bool).to_numpy(dtype=int)
    k = np.ones(run_len, dtype=int)
    y = np.convolve(x, k, mode="same") >= run_len
    return pd.Series(y, index=flag.index)

def _auto_theta_gmm(pitch_mean: pd.Series, min_n=200, max_components=3, seed=0):
    x = pd.to_numeric(pitch_mean, errors="coerce").dropna().to_numpy().reshape(-1,1)
    if len(x) < min_n: return None, {"reason":"too_few_samples"}
    best = None
    for k in range(1, max_components+1):
        g = GaussianMixture(n_components=k, covariance_type="full", n_init=5, random_state=seed).fit(x)
        bic = g.bic(x)
        if (best is None) or (bic < best[2]): best = (k,g,bic)
    k, gmm, _ = best
    if k == 1: return None, {"reason":"unimodal_pitch"}
    order = np.argsort(gmm.means_.ravel())
    mu = gmm.means_.ravel()[order]
    var = np.array([gmm.covariances_[i].ravel()[0] for i in order])
    w   = gmm.weights_.ravel()[order]

    def _intersect(m1,s1,w1,m2,s2,w2):
        a = 1/(2*s1) - 1/(2*s2); b = m2/s2 - m1/s1
        c = (m1*m1)/(2*s1) - (m2*m2)/(2*s2) + np.log((w1*np.sqrt(s2))/(w2*np.sqrt(s1)))
        return np.roots([a,b,c]).real

    cands = []
    for i in range(k-1):
        r = _intersect(mu[i],var[i],w[i],mu[i+1],var[i+1],w[i+1])
        mid = 0.5*(mu[i]+mu[i+1])
        in_between = [z for z in r if mu[i] <= z <= mu[i+1]]
        x_star = in_between[0] if in_between else r[np.argmin(np.abs(r-mid))]
        cands.append(x_star)
    if not cands or not np.isfinite(cands).all(): return None, {"reason":"no_intersection"}

    def _mix_pdf(z):
        s = np.sqrt(var)
        return np.sum(w*(1/(np.sqrt(2*np.pi)*s))*np.exp(-0.5*((z-mu)/s)**2))
    dens = [_mix_pdf(z) for z in cands]
    theta = float(cands[int(np.argmin(dens))])
    return theta, {"method":"gmm","theta":theta}

def _auto_theta_knee(pitch_mean: pd.Series, power_kw: pd.Series, Pr: float,
                     bin_deg=1.0, min_pts_per_bin=20, min_n=200):
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
    centers = np.asarray(centers)
    med = pd.Series(med).rolling(3, center=True, min_periods=1).median().to_numpy()
    med = np.maximum.accumulate(med[::-1])[::-1]  # monotone decreasing envelope
    x = (centers - centers.min())/(centers.ptp()+1e-12)
    y = (med - med.min())/(med.ptp()+1e-12)
    theta = float(centers[int(np.argmax(y - x))])
    return theta, {"method":"knee","theta":theta,"bins_used":int(len(centers))}

def apply_prefilters(
    df: pd.DataFrame,
    *,
    Pr: float,                      # kW
    persist_bins: int = 2,
    min_setpoint_frac: float = 0.10,# ignore very low setpoints
    obs_tol: float = 0.10,          # binding tolerance using observed power
    force_theta: float | None = None,
    theta_bounds: tuple[float,float] = (22.0, 45.0),  # guardrails
):
    out = df.copy()

    # --- θ* (pitch cutoff)
    if force_theta is not None:
        theta, tinfo = float(force_theta), {"method":"fixed"}
    else:
        theta, tinfo = _auto_theta_gmm(out["pitch_mean"])
        if (theta is None) and ("power" in out.columns):
            theta, tinfo = _auto_theta_knee(out["pitch_mean"], out["power"], Pr)
        # guardrails
        if theta is not None:
            lo, hi = theta_bounds
            theta = float(np.clip(theta, lo, hi))
            tinfo["theta_clamped_to"] = theta

    # --- Type-1 (pitch stop/transition)
    if theta is not None:
        t1_raw = out["pitch_max"] > theta
        out["Type1"] = _persistent(t1_raw, run_len=persist_bins)
    else:
        out["Type1"] = False

    # --- Type-2 (curtailment, observed binding only)
    if "power_setpoint" in out.columns:
        sp = pd.to_numeric(out["power_setpoint"], errors="coerce")
        pw = pd.to_numeric(out["power"], errors="coerce")
        t2_plain = (sp < 0.99*Pr) & (sp >= min_setpoint_frac*Pr)
        bind_obs = pw >= sp * (1 - obs_tol)  # close to setpoint → actually curtailed
        t2_raw   = t2_plain & bind_obs
        out["Type2"] = _persistent(t2_raw, run_len=persist_bins)
    else:
        out["Type2"] = False

    out["prefilter_remove"] = out["Type1"] | out["Type2"]

    # summary
    n_before = len(out)
    n_after  = int((~out["prefilter_remove"]).sum())
    summary = {
        "Pr_used": float(Pr),
        "theta": None if theta is None else float(theta),
        "theta_method": tinfo.get("method") if theta is not None else None,
        "before": n_before, "after": n_after,
        "removed": n_before - n_after,
        "pct_removed": 0 if n_before==0 else 100*(n_before - n_after)/n_before,
        "type1_frac": float(pd.Series(out["Type1"]).mean()),
        "type2_frac": float(pd.Series(out["Type2"]).mean()),
    }
    return out, summary

# 2) Prefilter (auto θ*, Type-1/2)
#Pr = 2050.0  # kW (put your turbine’s nameplate here)
#pref_df, summary = apply_prefilters(
#    cleaned_df,
#    Pr=Pr,
#    persist_bins=2,
#    min_setpoint_frac=0.10,   # ignore very low setpoints
#    obs_tol=0.10,             # power within 10% of setpoint → binding
#    # force_theta=30.0,       # optional: mimic paper for a sanity check
#    theta_bounds=(22.0, 45.0) # keep θ* in a sensible band
#)

#print(f"Before: {summary['before']:,}  After: {summary['after']:,}  "
#      f"Removed: {summary['removed']:,} ({summary['pct_removed']:.2f}%)")
#print("θ* =", summary["theta"], "method:", summary["theta_method"])

# 3) Keep data for Type-3 AD and modeling
#df_prefiltered = pref_df.loc[~pref_df["prefilter_remove"], :]