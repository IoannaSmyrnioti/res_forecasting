import numpy as np
import pandas as pd

def _auto_theta_knee(pitch_max: pd.Series,
                     power_kw: pd.Series,
                     Pr: float,
                     *,
                     q_hi=0.95,
                     bin_deg=1.0,
                     min_pts_per_bin=20,
                     min_n=200,
                     drop_from_plateau=0.05,  # 5% πτώση μετά το knee  
                     shift_bins: int | None = None, # εναλλακτικά: σταθερή μετατόπιση δεξιά)
                     return_debug: bool = True):
     """
        Αυτόματος υπολογισμός θ* (pitch cutoff) με βάση την κατανομή pitch_max vs normalized power (pfrac=power/Pr).
        Χρησιμοποιεί το kneedle algorithm για φθίνουσα καμπύλη (smoothed, monotonic q_hi quantile).
        1) Κανονικοποίηση power
        2) Binning στο pitch ανά bin_deg και υπολογισμός q_hi στο pfrac ανά bin (αγνοώ bins με < min_pts_per_bin)
        3) Εξομάλυνση (rolling median) και μονοτονία (φθίνουσα)
        4) Εφαρμογή kneedle για φθίνουσα καμπύλη και εύρεση knee μεγιστοποιώντας d=y-(1-x)
        5) Ορισμός plateau αριστερά του knee (μέσος όρος 3 bins) και threshold = plateau - drop_from_plateau
        6) Εύρεση πρώτου bin δεξιά του knee όπου q_hi <= threshold (αν υπάρχει)
           Αν δεν υπάρχει, επέλεξε το knee ή (αν δόθηκε) το knee + shift_bins 
        7) Επιστροφή θ* και info
        
     """
     df = pd.DataFrame({"pitch": pd.to_numeric(pitch_max, errors="coerce"),
                        "pfrac": pd.to_numeric(power_kw, errors="coerce")/float(Pr)}).dropna()
     if len(df) < min_n: return None, {"reason":"too_few_samples"}
     # binning στο pitch
     bins = np.arange(np.floor(df.pitch.min()), np.ceil(df.pitch.max())+bin_deg, bin_deg)
     idx  = np.digitize(df.pitch, bins)
     centers, qvals = [], []
     for b in np.unique(idx):
        vals = df.pfrac[idx==b]
        if len(vals) >= min_pts_per_bin:
            centers.append((bins[b-1] + (bins[b] if b < len(bins) else bins[-1]))/2.0)
            qvals.append(float(np.quantile(vals, q_hi)))
     if not qvals: return None, {"reason":"insufficient_bins"}
     centers = np.asarray(centers); qvals = np.asarray(qvals)
     # εξομάλυνση και μονοτονία (φθίνουσα)
     q_smooth = pd.Series(qvals).rolling(3, center=True, min_periods=1).median().to_numpy() 
     q_mono = np.maximum.accumulate(q_smooth[::-1])[::-1]  # monotone decreasing envelope 
     # kneedle για φθίνουσα
     x = (centers - centers.min())/(np.ptp(centers) + 1e-12) 
     y = (q_mono - q_mono.min()) / (np.ptp(q_mono) + 1e-12)
     d = y-(1-x)
     i_knee = int(np.argmax(d))
     # plateau αριστερά του knee
     left = max(0, i_knee - 10)
     right = max(i_knee, left + 3)
     plateau = float(np.median(q_mono[left:right])) if i_knee > 0 else float(q_mono[0]) 
     # κανόνας "right-of-knee": ψάξε το πρώτο bin δεξιά όπου q_hi (q_mono) <= thr
     thr = plateau - float(drop_from_plateau)
     cand = np.where(q_mono[i_knee:] <= thr)[0]
     if cand.size:
        theta = float(centers[i_knee + cand[0]])
        method = "knee_right_drop"
     elif isinstance(shift_bins, int) and (shift_bins > 0):
        i_shift = min(i_knee + shift_bins, len(centers)-1)
        theta = float(centers[i_shift])
        method = "knee_right_shift"
     else:
        theta = float(centers[i_knee])
        method = "knee"

     info = {
        "method": method,
        "theta": theta,
        "bins_used": int(len(centers)),
        "knee_index": i_knee,
        "plateau": plateau,
        "thr": thr,
        "q_hi": q_hi,
        "bin_deg": bin_deg,
        "min_pts_per_bin": min_pts_per_bin
     }
     if return_debug:
        info["centers"] = centers
        info["qvals"]   = qvals
        info["q_mono"]  = q_mono
        info["plateau_left"]  = int(left)
        info["plateau_right"] = int(right)
     return theta, info

def choose_msf(
    df: pd.DataFrame,
    Pr: float,
    sp_col: str = "power_setpoint",
    ts_col: str | None = "timestamp",
    *,
    # αρχικά & όρια για band
    start_band_kW: float | None = None,  # αν None -> adaptive από step
    max_band_kW: float = 150.0,
    band_step_kW: float = 10.0,
    # κριτήρια υποστήριξης
    min_count: int = 50,
    min_share: float = 0.002,    # 0.2% του δείγματος
    min_days: int | None = 3,    # προαιρετικά: εμφανίζεται σε ≥3 μέρες
    # “κοντά στο 0.20·Pr”
    near: tuple[float,float] = (0.18, 0.22),
    clip_range: tuple[float,float] = (0.10, 0.25),
    q_low: float = 0.05
):
    """
    Επιλογή minimun setpoint fraction (msf) κοντά σε ~0,20Pr για φιλτράρισμα περικοπών (curtailment).
    (Οι τουρμπίνες έχουν συνήθως setpoints γύρω στο 20% του Pr όταν δεν είναι σε λειτουργία. Στόχος είναι να βρεθεί
    αυτόματα μια τιμή κοντά σε αυτό το ποσοστό, αγνοώντας πολύ μικρές τιμές setpoint και λαμβάνοντας πραγματικές καταστάσεις curtailment.
    1) Εξαγωγή έγκυρων setpoints > 0
    2) Υπολογισμός min_sp = min(valid setpoints) και min_frac = min_sp/Pr
    3) Σάρωση band από start_band_kW έως max_band_kW με βήμα band_step_kW
       Για κάθε band, έλεγχος αν υπάρχουν αρκετά setpoints στο [min_sp, min_sp+band]
       που να πληρούν τα κριτήρια υποστήριξης (min_count, min_share, min_days)
       και αν το min_frac είναι εντός του εύρους near.
       Αν ναι, επέστρεψε το min_frac με info.
    4) Αν αποτύχει η υποστήριξη, βρες τη mode τιμή των setpoints εντός του εύρους near·Pr
       (με ομαδοποίηση σε βήματα των 50 kW για αποφυγή θορύβου).
       Αν βρεθεί, επέστρεψε τη mode/Pr με info.
    5) Αν αποτύχει και αυτό, επέστρεψε το q_low quantile των setpoints/Pr
       με clip στο clip_range (π.χ. [0.10, 0.25]) με info.  
    Αν αποτύχουν όλα, επέστρεψε 0.20 με info.
    """
    sp = pd.to_numeric(df[sp_col], errors="coerce")
    sp = sp[sp > 0]
    if sp.empty:
        return 0.20, {"method":"fallback_empty"}

    min_sp = float(sp.min())
    min_frac = min_sp/float(Pr)

    # εκτίμηση βηματισμού setpoints (quantization step) για adaptive αρχικό band
    if start_band_kW is None:
        uniq = np.sort(sp.unique())
        diffs = np.diff(uniq)
        if (diffs > 0).any():
            step_est = float(np.quantile(diffs[diffs > 0], 0.2))
        else:
            step_est = 50.0
        start_band_kW = float(np.clip(0.6*step_est, 0.01*Pr, 100.0))

    
    # helper για έλεγχο ημερών
    def days_support(mask_idx):
        if ts_col is None or ts_col not in df.columns:
            return None
        t = pd.to_datetime(df[ts_col], errors="coerce")
        t = t.loc[sp.index]  # ευθυγράμμιση σε valid SP
        return int(t[mask_idx].dt.date.nunique())

    # σαρώσω το band από start → max
    band = start_band_kW
    while band <= max_band_kW:
        # “γειτονιά” γύρω από min_sp (δε χρειάζεται κάτω όριο < min_sp)
        in_band_mask = (sp >= min_sp) & (sp <= min_sp + band)
        count = int(in_band_mask.sum())
        share = float(count / len(sp))
        days = days_support(in_band_mask.index[in_band_mask]) if min_days is not None else None

        in_near = (near[0] <= min_frac <= near[1])
        has_support = (count >= min_count) or (share >= min_share)
        days_ok = True if (min_days is None or days is None) else (days >= min_days)

        if in_near and has_support and days_ok:
            return min_frac, {
                "method": "min_with_support_adaptive",
                "min_frac": min_frac,
                "min_sp_kW": min_sp,
                "band_kW": band,
                "count": count,
                "share": share,
                "days": days,
                "start_band_kW": start_band_kW
            }

        band += band_step_kW

    # mode στη ζώνη near (αν απέτυχε η υποστήριξη στο min)
    sp_step = (sp/50).round().mul(50)
    lo2, hi2 = near[0]*Pr, near[1]*Pr
    band_vals = sp_step[(sp_step >= lo2) & (sp_step <= hi2)]
    if not band_vals.empty:
        mode_val = float(band_vals.value_counts().idxmax())
        msf_mode = mode_val/float(Pr)
        return msf_mode, {
            "method": "mode_band",
            "mode_kW": mode_val,
            "near": near
        }

    # quantile + clip (τελευταίο καταφύγιο)
    s_frac = sp/float(Pr)
    q = float(np.quantile(s_frac, q_low))
    msf = float(np.clip(q, *clip_range))
    return msf, {
        "method": "quantile_clip",
        "q_low": q_low, "q": q,
        "clip": clip_range
    }

def apply_prefilters(
    df: pd.DataFrame,
    *,
    # Type 2 (setpoint) options
    Pr: float,                      # kW
    min_setpoint_frac: float | None = None ,# ignore very low setpoints 
    obs_tol: float = 0.10,          # binding tolerance using observed power
    msf_kwargs: dict | None = None,  # kwargs για choose_msf()
    # θ* options (knee)
    force_theta: float | None = None, # δώσε σταθερό θ* για override
    theta_bounds: tuple[float,float] | None = None,  # π.χ. (22,45) αν θες guardrails
    knee_kwargs: dict | None = None,     # π.χ. {"q_hi":0.95,"bin_deg":1.0,"min_pts_per_bin":20,"drop_from_plateau":0.05}
    # flags για επιστροφή
    keep: str = "kept",  # "all", "kept", "removed"
    include_flags: bool = False  # αν True, κράτα τις στήλες Type1, Type2, prefilter_remove
):
    """
    Apply prefilters :
    - Type-1: pitch_stop -> pitch_max > θ*  (θ* αυτόματα με _auto_theta_knee ή σταθερό force_theta)
    - Type-2: curtailment -> setpoint<0.99·Pr & binding σε observed power

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least 'power' and (ideally)'pitch_max' and  'power_setpoint' columns.
    Pr : float
        Turbine nameplate power in kW.
    min_setpoint_frac : float, optional
        Minimum setpoint fraction of Pr to consider for Type-2 filtering. If None, it will be estimated (default is None).
    obs_tol : float, optional
        Tolerance for observed power to consider it binding (default is 0.10).
    msf_kwargs : dict, optional
        Additional keyword arguments for the choose_msf function (default is None).
    force_theta : float or None, optional
        If provided, uses this fixed value for θ* instead of estimating it (default is None).
    theta_bounds : tuple of float or None, optional
        Bounds to clamp the estimated θ* value (default is None).
    knee_kwargs : dict, optional
        Additional keyword arguments for the _auto_theta_knee function (default is None).
    keep : str, optional
        Determines which rows to return: 'all' (default), 'kept' (not removed), or 'removed'.
    include_flags : bool, optional
        If True, includes the boolean columns 'Type1', 'Type2', and 'prefilter_remove' in the output (default is False).
        
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame based on the specified criteria.
    dict
        Summary of the filtering process, including parameters used and counts of rows before and after filtering.
    """
    out = df.copy()
    knee_kwargs = knee_kwargs or {}
    msf_kwargs  = msf_kwargs or {}

    # Yπολογισμός min_setpoint_frac αν δεν δόθηκε
    msf_info = None
    if min_setpoint_frac is None:
        if "power_setpoint" in out.columns:
            try:
                msf, msf_info = choose_msf(
                    out, Pr, 
                    sp_col=msf_kwargs.get("sp_col","power_setpoint"),
                    ts_col=msf_kwargs.get("ts_col","timestamp"),
                    start_band_kW=msf_kwargs.get("start_band_kW",None),
                    max_band_kW=msf_kwargs.get("max_band_kW",150.0),
                    band_step_kW=msf_kwargs.get("band_step_kW",10.0),
                    min_count=msf_kwargs.get("min_count",50),
                    min_share=msf_kwargs.get("min_share",0.002),
                    min_days=msf_kwargs.get("min_days",3),
                    near=msf_kwargs.get("near",(0.18,0.22)),
                    clip_range=msf_kwargs.get("clip_range",(0.10,0.25)),
                    q_low=msf_kwargs.get("q_low",0.05)
                )
                valid_msf = float(msf) 
                if (not np.isfinite(valid_msf)) or (valid_msf <= 0) or (valid_msf >= 1):
                    raise ValueError("bad msf")
                min_setpoint_frac = valid_msf
            except Exception as e:
                min_setpoint_frac = 0.20
                msf_info = {"method":"fallback_fixed","min_setpoint_frac":min_setpoint_frac,"error":str(e)}
        else:
            min_setpoint_frac=0.0 #αν δεν υπαρχει η στηλη setpoint απενεργοποιησε το φιλτρο
            msf_info = {"method":"no_setpoint_col"}
    else:
        min_setpoint_frac = float(np.clip(min_setpoint_frac, 0.0, 0.99))
        msf_info = {"method":"provided","min_setpoint_frac":min_setpoint_frac}


    # θ* (pitch cutoff)
    tinfo = {}
    if force_theta is not None:
        theta, tinfo = float(force_theta), {"method":"fixed"}
    else:
        # SAFE PICK για pitch & power
        if "pitch_max" in out.columns:
            pser = out["pitch_max"]
        elif "pitch_mean" in out.columns:
            pser = out["pitch_mean"]
        else:
            pser = None

        if (pser is None) or ("power" not in out.columns):
            theta, tinfo = None, {"reason":"missing_cols"}
        else:
            theta, tinfo = _auto_theta_knee(pser, out["power"], Pr, **knee_kwargs)
            # (προαιρετικά) guardrails
            if (theta is not None) and (theta_bounds is not None):
                lo, hi = theta_bounds
                theta = float(np.clip(theta, lo, hi))
                tinfo["theta_clamped_to"] = theta

    # Type-1 (pitch-based) prefiltering
    if (theta is not None) and ("pitch_max" in out.columns):
        out["Type1"] = out["pitch_max"] > theta
    else:
        out["Type1"] = False

    # Type-2 (setpoint-based) prefiltering. Requires 'power' and 'power_setpoint' columns. 
    if "power_setpoint" in out.columns:
        sp = pd.to_numeric(out["power_setpoint"], errors="coerce")
        pw = pd.to_numeric(out["power"], errors="coerce")
        t2_plain = (sp < 0.99*Pr) & (sp >= min_setpoint_frac*Pr)
        bind_obs = pw >= sp * (1 - obs_tol)  # close to setpoint → actually curtailed
        out["Type2"] = (t2_plain & bind_obs).fillna(False)
    else:
        out["Type2"] = False

    # τελικό prefilter
    out["prefilter_remove"] = out["Type1"] | out["Type2"]

    # summary
    n_before = len(out)
    n_kept  = int((~out["prefilter_remove"]).sum())
    n_removed = n_before - n_kept
    
    # choose what to return in summary
    keep= keep.lower()
    if keep == "kept":
        result = out.loc[~out["prefilter_remove"]].copy()
    elif keep == "removed":
        result = out.loc[out["prefilter_remove"]].copy()
    elif keep == "all":
        result = out.copy()
    else:
        raise ValueError("Invalid value for 'keep'. Choose from 'all', 'kept', 'removed'.")
    
    if not include_flags:
        result = result.drop(columns=["Type1", "Type2", "prefilter_remove"], errors="ignore")
    
    summary = {
        "Pr_used": float(Pr),
        "theta": None if theta is None else float(theta),
        "theta_method": tinfo.get("method") if theta is not None else None,
        "theta_info": tinfo,
        "msf": float(min_setpoint_frac),
        "msf_info": msf_info,
        "before": n_before, 
        "kept": n_kept,
        "removed": n_removed,
        "pct_removed": 0 if n_before==0 else 100*n_removed/n_before,
        "type1_frac": float(pd.Series(out["Type1"]).mean()),
        "type2_frac": float(pd.Series(out["Type2"]).mean()),
        "returned": len(result),
        "return_mode": keep,
        "include_flags": bool(include_flags)  
    }
    return result, summary

