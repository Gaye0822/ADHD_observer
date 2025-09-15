# ADHD_observer

ADHD Observer — EDA + Heart Rate (+ optional Unity penalties)
=============================================================

WHAT THIS IS
------------
A small, clinician-friendly pipeline that turns:
- EDA-based stress probability (`p_stress`) per window (from your existing model),
- Heart-rate time series (HR in bpm),
- (Optional) Unity penalty logs,

into interpretable window summaries and simple plots. This is **not diagnostic**; it is meant to support observation and discussion.

FILES EXPECTED (same folder)
----------------------------
- adhd_observer.py          # the script that merges & reports
- requirements.txt          # Python deps
- arduino_features_from_upload_pred.csv   # EDA predictions (must contain p_stress)
- dummy_hr.csv (or your own HR CSV)       # columns: elapsed_s, hr_bpm
- Kayıt-4.csv (optional)                  # Unity penalties (Elapsed, Event, ScenarioPenalty)

INSTALL (Python 3.10+)
----------------------
# create & activate a virtualenv
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

HOW TO RUN
----------
(1) With Unity penalties:
python3 adhd_observer.py \
  --pred ./arduino_features_from_upload_pred.csv \
  --hr ./dummy_hr.csv \
  --unity "./Kayıt-4.csv" \
  --win 60 --smooth-n 2 --hi 0.65 --lo 0.55 \
  --out ./adhd_observer_report.csv \
  --plots-dir ./adhd_plots

(2) Without Unity penalties:
python3 adhd_observer.py \
  --pred ./arduino_features_from_upload_pred.csv \
  --hr ./dummy_hr.csv \
  --win 60 --smooth-n 2 --hi 0.65 --lo 0.55 \
  --out ./adhd_observer_report.csv \
  --plots-dir ./adhd_plots

Notes:
- If your training used 90 s windows, prefer: --win 90
- Use quotes around non-ASCII file names (e.g., "Kayıt-4.csv").

OUTPUTS
-------
- adhd_observer_report.csv   # one row per window with merged metrics
- adhd_plots/
    - plot_p_stress.png      # p_stress (raw & smoothed)
    - plot_hr_mean.png       # mean HR per window
    - plot_rmssd.png         # RMSSD per window
    - (optional) plot_penalty_rate.png  # penalties/min if Unity provided
    - session_summary.txt     # brief numeric summary (non-diagnostic)

WHAT THE SCRIPT COMPUTES (interpretable metrics)
------------------------------------------------
EDA (from your predictions CSV):
- p_stress: stress probability per window.
- p_stress_smooth: moving average (default 2 windows) to reduce jitter.
- p_high_flag: a "high state" flag using **hysteresis**:
  * Turns ON when p_stress_smooth ≥ 0.65,
  * Turns OFF when p_stress_smooth ≤ 0.55,
  * Requires two consecutive windows to switch state (prevents rapid toggling).

Heart Rate (from HR CSV):
- mean_hr: average HR in the window (bpm).
- SDNN: standard deviation of RR intervals (ms).
- RMSSD: root mean square of successive RR diffs (ms).
- pNN50: fraction of RR diffs > 50 ms.
  (RR is derived as RR(ms) = 60000 / HR(bpm). This is an approximation when only HR is available.)

Unity penalties (optional):
- pen_sum: total penalties in the window.
- pen_cnt: # of non-zero penalty events in the window.
- pen_rate_per_min: penalties per minute in the window.

SESSION-LEVEL NUMBERS (in session_summary.txt)
----------------------------------------------
- p_stress_mean (smoothed), p_stress_p90  → arousal/load tendency.
- high_state_pct (%)                      → fraction of session in the high state.
- mean_hr (bpm), RMSSD median & p10      → autonomic regulation tendency.

HOW TO READ THESE NUMBERS (non-diagnostic)
------------------------------------------
- Higher p_stress and larger high_state_pct suggest increased arousal / workload.
- Lower RMSSD suggests reduced vagal regulation / self-regulation capacity.
- If Unity penalties are provided, higher penalties/min together with high p_stress can indicate attentional difficulty during that period.

WHAT THESE CHOICES ARE BASED ON
-------------------------------
- EDA reflects sympathetic arousal and often increases with cognitive load; we smooth and use hysteresis to avoid false flips and emphasize sustained changes.
- HRV time-domain metrics (SDNN, RMSSD, pNN50) are widely used interpretable indices of autonomic regulation (RMSSD: short-term vagal activity).
- Fixed-length windows (60/90 s) are standard for stable HRV and EDA summaries. If EDA and Unity are recorded simultaneously, aligning by time window makes the signals comparable.
- This is an **observational** tool. Individual variability, anxiety, movement artifacts, caffeine, medication, and sleep can affect EDA/HRV. Results should be interpreted by a clinician and are not a diagnosis.

TIPS & LIMITATIONS
------------------
- If your EDA and Unity logs are simultaneous, consider adding a single start marker (keyboard/UDP) to align clocks. EDA phasic responses can lag by ~1–3 s; windowing absorbs most of this.
- If only HR is available (no RR), HRV from HR is an approximation. For accurate HRV, RR (beat-to-beat) intervals are preferred.
- Keep the window length consistent between model training and observation (60 or 90 s).
- Reduce movement artifacts: stable sensor placement and cables.

REPLACING THE DUMMY HR WITH YOUR OWN
------------------------------------
- Use a CSV with columns: elapsed_s (seconds from session start), hr_bpm.
- Then re-run the commands above with --hr set to your file.

LICENSE / DISCLAIMER
--------------------
This script is provided for research and educational purposes. It is not a medical device and does not provide a clinical diagnosis. Always consult qualified clinicians for diagnostic decisions.
