
import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_csv_auto(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=';')

def mmss_to_s(s):
    try:
        parts = str(s).split(':')
        if len(parts) == 2:
            m, ss = parts
            return int(m)*60 + int(ss)
        elif len(parts) == 3:
            h, m, ss = parts
            return int(h)*3600 + int(m)*60 + int(ss)
        else:
            return np.nan
    except Exception:
        return np.nan

def hrv_time_domain(rr_ms):
    rr_ms = np.asarray(rr_ms, dtype=float)
    rr_ms = rr_ms[np.isfinite(rr_ms)]
    if len(rr_ms) < 3:
        return dict(mean_hr=np.nan, sdnn=np.nan, rmssd=np.nan, pnn50=np.nan)
    mean_hr = 60000.0 / np.mean(rr_ms)
    sdnn = np.std(rr_ms, ddof=1)
    diff = np.diff(rr_ms)
    rmssd = np.sqrt(np.mean(diff**2))
    pnn50 = np.mean(np.abs(diff) > 50.0)
    return dict(mean_hr=mean_hr, sdnn=sdnn, rmssd=rmssd, pnn50=pnn50)

def aggregate_hr_per_window(hr_df, win_edges_s):
    out = []
    for (s, e) in win_edges_s:
        seg = hr_df[(hr_df['elapsed_s'] >= s) & (hr_df['elapsed_s'] < e)]
        if len(seg) == 0:
            out.append(dict(mean_hr=np.nan, sdnn=np.nan, rmssd=np.nan, pnn50=np.nan))
            continue
        rr_ms = 60000.0 / np.clip(seg['hr_bpm'].values, 25, 220)
        out.append(hrv_time_domain(rr_ms))
    return out

def aggregate_penalties_per_window(unity_df, win_edges_s):
    out = []
    for (s, e) in win_edges_s:
        seg = unity_df[(unity_df['Elapsed_s'] >= s) & (unity_df['Elapsed_s'] < e)]
        pen = pd.to_numeric(seg.get('ScenarioPenalty', pd.Series([])), errors='coerce').fillna(0.0).values
        rate = pen.sum() / max((e - s)/60.0, 1e-9)
        out.append(dict(pen_sum=float(np.sum(pen)), pen_cnt=int(np.sum(pen > 0)), pen_rate_per_min=float(rate)))
    return out

def make_windows_from_pred(pred_df, default_win=60.0):
    if 't_start' in pred_df.columns and 't_end' in pred_df.columns:
        return list(zip(pred_df['t_start'].values.astype(float), pred_df['t_end'].values.astype(float)))
    if 'elapsed_s' in pred_df.columns:
        starts = pred_df['elapsed_s'].values.astype(float)
        ends = starts + default_win
        return list(zip(starts, ends))
    n = len(pred_df)
    return [(i*default_win, (i+1)*default_win) for i in range(n)]

def smooth(series, n=2):
    return pd.Series(series).rolling(n, min_periods=1).mean().values

def hysteresis_labels(series, hi=0.65, lo=0.55, min_run=2):
    state = 0; run = 0; labels = []
    for v in series:
        target = state
        if state == 0 and v >= hi: target = 1
        elif state == 1 and v <= lo: target = 0
        if target != state:
            run += 1
            if run >= min_run:
                state = target; run = 0
        else:
            run = 0
        labels.append(state)
    return np.array(labels, dtype=int)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--hr", required=True)
    ap.add_argument("--unity", default=None)
    ap.add_argument("--win", type=float, default=60.0)
    ap.add_argument("--smooth-n", type=int, default=2)
    ap.add_argument("--hi", type=float, default=0.65)
    ap.add_argument("--lo", type=float, default=0.55)
    ap.add_argument("--out", default="adhd_observer_report.csv")
    ap.add_argument("--plots-dir", default="adhd_plots")
    args = ap.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)

    pred = read_csv_auto(args.pred)
    pred.columns = [c.strip().lower() for c in pred.columns]
    pcol = "p_stress" if "p_stress" in pred.columns else ("p_stress_raw" if "p_stress_raw" in pred.columns else None)
    if pcol is None:
        raise SystemExit("Pred CSV must contain p_stress or p_stress_raw.")
    wins = make_windows_from_pred(pred, default_win=args.win)
    p_vals = pred[pcol].values
    p_smooth = smooth(p_vals, n=args.smooth_n)
    p_high = hysteresis_labels(p_smooth, hi=args.hi, lo=args.lo, min_run=2)

    hr = read_csv_auto(args.hr)
    hr.columns = [c.strip() for c in hr.columns]
    if "elapsed_s" not in hr.columns or "hr_bpm" not in hr.columns:
        raise SystemExit("HR CSV must have columns: elapsed_s, hr_bpm")
    hr_metrics = aggregate_hr_per_window(hr, wins)

    unity_metrics = None
    if args.unity:
        unity = read_csv_auto(args.unity)
        unity.columns = [c.strip() for c in unity.columns]
        if "Elapsed" in unity.columns:
            # Convert to seconds from start
            def mmss_to_s(s):
                try:
                    parts = str(s).split(':')
                    if len(parts) == 2:
                        m, ss = parts
                        return int(m)*60 + int(ss)
                    elif len(parts) == 3:
                        h, m, ss = parts
                        return int(h)*3600 + int(m)*60 + int(ss)
                    else:
                        return np.nan
                except:
                    return np.nan
            unity["Elapsed_s"] = unity["Elapsed"].map(mmss_to_s)
            unity["Elapsed_s"] = unity["Elapsed_s"] - unity["Elapsed_s"].min()
        if "Elapsed_s" in unity.columns and "Event" in unity.columns:
            unity_samples = unity[unity["Event"].astype(str).str.lower().eq("sample")].copy()
            unity_metrics = aggregate_penalties_per_window(unity_samples, wins)

    rows = []
    for i, (s, e) in enumerate(wins):
        row = {
            "win_idx": i,
            "t_start": s,
            "t_end": e,
            "p_stress": float(p_vals[i]) if i < len(p_vals) else np.nan,
            "p_stress_smooth": float(p_smooth[i]) if i < len(p_smooth) else np.nan,
            "p_high_flag": int(p_high[i]) if i < len(p_high) else 0,
        }
        row.update(hr_metrics[i] if i < len(hr_metrics) else {})
        if unity_metrics is not None:
            row.update(unity_metrics[i] if i < len(unity_metrics) else {})
        rows.append(row)
    merged = pd.DataFrame(rows)
    merged.to_csv(args.out, index=False)

    # Session summaries (non-diagnostic)
    def pct(x, q):
        x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
        return float(np.percentile(x, q)) if len(x) else np.nan
    sess = {
        "p_stress_mean": float(np.nanmean(merged["p_stress_smooth"])),
        "p_stress_p90": pct(merged["p_stress_smooth"], 90),
        "high_state_pct": 100.0 * float(np.nanmean(merged["p_high_flag"])),
        "mean_hr": float(np.nanmean(merged["mean_hr"])),
        "rmssd_median": pct(merged["rmssd"], 50),
        "rmssd_p10": pct(merged["rmssd"], 10),
    }
    with open(os.path.join(args.plots_dir, "session_summary.txt"), "w", encoding="utf-8") as f:
        for k, v in sess.items():
            f.write(f"{k}: {v}\n")

    # Plots
    x = merged["t_start"].values / 60.0
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,3))
    plt.plot(x, merged["p_stress"].values)
    plt.plot(x, merged["p_stress_smooth"].values)
    plt.xlabel("Time (min)"); plt.ylabel("p_stress"); plt.title("p_stress (raw & smoothed)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.plots_dir, "plot_p_stress.png"), dpi=160); plt.close()

    plt.figure(figsize=(10,3))
    plt.plot(x, merged["mean_hr"].values)
    plt.xlabel("Time (min)"); plt.ylabel("HR (bpm)"); plt.title("Mean HR per window")
    plt.tight_layout()
    plt.savefig(os.path.join(args.plots_dir, "plot_hr_mean.png"), dpi=160); plt.close()

    plt.figure(figsize=(10,3))
    plt.plot(x, merged["rmssd"].values)
    plt.xlabel("Time (min)"); plt.ylabel("RMSSD (ms)"); plt.title("RMSSD per window")
    plt.tight_layout()
    plt.savefig(os.path.join(args.plots_dir, "plot_rmssd.png"), dpi=160); plt.close()

    if 'pen_rate_per_min' in merged.columns:
        plt.figure(figsize=(10,3))
        plt.plot(x, merged['pen_rate_per_min'].values)
        plt.xlabel("Time (min)"); plt.ylabel("penalty/min"); plt.title("Penalty rate per window")
        plt.tight_layout()
        plt.savefig(os.path.join(args.plots_dir, "plot_penalty_rate.png"), dpi=160); plt.close()

    print("Saved:", args.out, "and plots in", args.plots_dir)

if __name__ == "__main__":
    main()
