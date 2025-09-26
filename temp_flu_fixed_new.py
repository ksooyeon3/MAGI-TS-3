# === MAGI-X 2D (flu): rolling 12-week forecasts (Sep Y → Aug Y+1), retrain each step ===
# Key fixes:
#  - Each step trains on a LOCAL time axis (t=0 at that step's start)
#  - Pseudo-observations are stored in ABSOLUTE dates and converted to the local axis per step
#  - Removed the "offset" re-anchoring hack to prevent repeated-looking segments

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import time, os
import matplotlib.colors as mcolors

from scripts.magix.dynamic import nnSTModule, nnMTModule  # keep import paths
from scripts.magix.inference import FMAGI                 # keep import paths

# -------------------------- 0) Config --------------------------
DATA_CSV   = "/home/giung/MAGI-TS-3/data/flu_hosp_ili.csv"
SEED       = 188714368
GRID_SIZE  = 201
INTERP_ORD = 3
MAX_EPOCH  = 2500
LR         = 1e-3
COMPONENTS = ["hosp_US", "weighted_ili_US"]  # two dims
SAVE_FIGS  = True
FIG_DIR    = "./figs_fmagi_yearly_flu_0921_4week_use_true_paste"
INCLUDE_PSEUDO = False    # use previous-step forecasts as pseudo-obs
START_YEAR = 2025
STOP_YEAR  = None  # or e.g., 2017

# Scaling: choose one (per component)
SCALE = "minmax5"        # "minmax5" | "zscore" | None
#   minmax5: X_scaled = 5 * (X - min_train) / (max_train - min_train)

# Rolling forecast step
STEP_WEEKS = 4          # forecast horizon per step

np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------------- 1) Load ---------------------------
df = pd.read_csv(DATA_CSV)
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values("date")
else:
    df = df.reset_index().rename(columns={"index": "date"})
    df["date"] = pd.to_datetime(df["date"], unit="D", origin="unix")

# require both components
for col in ["date"] + COMPONENTS:
    if col not in df.columns:
        raise ValueError(f"Expected column '{col}' not found.")
df = df.dropna(subset=COMPONENTS).reset_index(drop=True)

# ------------------- 2) Yearly windows helper -----------------
def year_windows(dates: pd.Series):
    """
    Yield (year, idx_cut, idx_end) where:
      idx_cut = first row with date >= Sep 1 of 'year'
      idx_end = last row with date <= Aug 31 of 'year+1'
    Only yields if both bounds exist and idx_end > idx_cut.
    """
    years = sorted(dates.dt.year.unique())
    for y in years:
        sep1 = pd.Timestamp(year=y, month=9, day=1)
        aug31_next = pd.Timestamp(year=y+1, month=8, day=31, hour=23, minute=59, second=59)
        cut_idx_arr = np.where(dates.values >= sep1.to_datetime64())[0]
        end_idx_arr = np.where(dates.values <= aug31_next.to_datetime64())[0]
        if len(cut_idx_arr) == 0 or len(end_idx_arr) == 0:
            continue
        idx_cut = int(cut_idx_arr[0])
        idx_end = int(end_idx_arr[-1])
        if idx_end <= idx_cut:
            continue
        yield y, idx_cut, idx_end

# ---- helpers for scaling/unscaling a 1D vector (per component) ----
def fit_scaler_params(train_vals: np.ndarray, mode: str):
    train_vals = np.asarray(train_vals, dtype=float)
    if mode == "minmax5":
        lo = float(np.nanmin(train_vals)); hi = float(np.nanmax(train_vals))
        if not np.isfinite(hi - lo) or (hi - lo) == 0:
            hi = lo + 1.0
        denom = (hi - lo)
        def f(x):
            z = 5.0 * (x - lo) / denom
            return np.nan_to_num(z, nan=0.0)
        def finv(z):
            return lo + (z / 5.0) * (hi - lo)
        return f, finv, {"lo": lo, "hi": hi}
    elif mode == "zscore":
        mu = float(np.nanmean(train_vals)); sd = float(np.nanstd(train_vals, ddof=0))
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        def f(x):
            z = (x - mu) / sd
            return np.nan_to_num(z, nan=0.0)
        def finv(z):
            return mu + z * sd
        return f, finv, {"mu": mu, "sd": sd}
    else:
        f = lambda x: x
        finv = lambda x: x
        return f, finv, {}

def _align_pairs(t_arr: np.ndarray, y_arr: np.ndarray):
    """Return aligned (t, y) with equal length, sorted by time."""
    t = np.asarray(t_arr, dtype=float).ravel()
    y = np.asarray(y_arr, dtype=float).ravel()
    if t.size == 0 or y.size == 0:
        return np.empty((0,), dtype=float), np.empty((0,), dtype=float)
    k = min(t.size, y.size)
    t = t[:k]; y = y[:k]
    order = np.argsort(t, kind="stable")
    return t[order], y[order]

# ------------- 3.5) Helpers for stepwise index math ----------
def advance_by_weeks(dates: pd.Series, start_idx: int, weeks: int, idx_end: int) -> int:
    """Return the last index <= start_date + weeks, capped by idx_end; ensures progress."""
    start_date = dates.iloc[start_idx]
    target_date = start_date + pd.Timedelta(weeks=weeks)
    j = int(np.searchsorted(dates.values, target_date.to_datetime64(), side="right") - 1)
    j = max(j, start_idx + 1)          # ensure at least one-step progress
    j = min(j, idx_end)                # cap at window end
    return j

# -------- 4) One yearly rolling experiment (stitched forecast lines) --------
import matplotlib.dates as mdates

def run_one_year_rolling(df_slice, idx_cut, idx_end, year):
    if idx_cut < 3:
        print(f"[{year}] Not enough pre–Sep-1 training points (<3). Skipping this year.")
        return None

    df_full = df_slice.iloc[:idx_end+1].copy()
    dates = df_full["date"].reset_index(drop=True)
    X_full = df_full[COMPONENTS].to_numpy().astype(float)
    D = X_full.shape[1]

    # 결과 컨테이너 (절대 날짜축)
    yhat_full = np.full_like(X_full, fill_value=np.nan, dtype=float)

    # pseudo buffers (절대날짜, 언스케일 값)
    pseudo_dates = np.empty((0,), dtype='datetime64[ns]')
    pseudo_Y = [np.empty((0,), dtype=float) for _ in range(D)]

    cur_cut = int(idx_cut)
    step_id = 0

    print(f"\n=== [{year}] Rolling forecasts every {STEP_WEEKS} weeks (Sep→Aug) ===")
    print(f"Initial train end (exclusive): {dates.iloc[cur_cut].date()} (idx={cur_cut})")
    print(f"Forecast stop (inclusive):     {dates.iloc[idx_end].date()} (idx={idx_end})")

    while cur_cut <= idx_end - 1:
        blk_end = advance_by_weeks(dates, cur_cut, STEP_WEEKS, idx_end)
        if cur_cut < 3:
            print(f"[{year} | step {step_id}] Not enough training points (<3). Skipping.")
            break

        # 1) 현재 스텝의 로컬 학습축
        t_train_abs = dates.iloc[:cur_cut].to_numpy()
        t0 = t_train_abs[0]
        t_train_local = (t_train_abs - t0).astype('timedelta64[s]').astype(float) / (7*24*3600.0)
        X_train_true = X_full[:cur_cut, :]

        # 2) 스케일러
        scaler_fns, inv_scaler_fns = [], []
        for i in range(D):
            if INCLUDE_PSEUDO and pseudo_dates.size > 0 and pseudo_Y[i].size > 0:
                f, finv, _ = fit_scaler_params(
                    np.concatenate([X_train_true[:, i], pseudo_Y[i]]), SCALE
                )
            else:
                f, finv, _ = fit_scaler_params(X_train_true[:, i], SCALE)
            scaler_fns.append(f); inv_scaler_fns.append(finv)

        # 3) 관측치 (로컬축으로 정렬/스케일)
        obs = []
        for i in range(D):
            if INCLUDE_PSEUDO and pseudo_dates.size > 0 and pseudo_Y[i].size > 0:
                tPi_local = (pseudo_dates - t0).astype('timedelta64[s]').astype(float) / (7*24*3600.0)
                t_obs_local = np.concatenate([t_train_local, tPi_local])
                y_obs = np.concatenate([X_train_true[:, i], pseudo_Y[i]])
            else:
                t_obs_local = t_train_local.copy()
                y_obs = X_train_true[:, i].copy()
            t_obs_local, y_obs = _align_pairs(t_obs_local, y_obs)
            y_obs_scaled = scaler_fns[i](y_obs)
            obs.append(np.column_stack([t_obs_local, y_obs_scaled]))

        # 4) FMAGI 학습
        torch.manual_seed(SEED)
        fOde = nnMTModule(D, [512], dp=0.0)
        model = FMAGI(obs, fOde, grid_size=GRID_SIZE, interpolation_orders=INTERP_ORD)

        print(f"[{year} | step {step_id}] Train rows: [0:{cur_cut}) → Forecast rows: [{cur_cut}:{blk_end}] "
              f"({dates.iloc[cur_cut].date()} → {dates.iloc[blk_end].date()})")

        t_start = time.time()
        tinfer, xinfer = model.map(
            max_epoch=MAX_EPOCH,
            learning_rate=LR,
            decay_learning_rate=True,
            hyperparams_update=True,
            dynamic_standardization=True,
            verbose=True,
            returnX=True
        )
        print(f"[{year} | step {step_id}] map() {time.time() - t_start:.2f}s")

        # 5) 이번 스텝 예측 (로컬축 → 절대날짜 인덱스로 반영)
        tp_abs = dates.iloc[cur_cut:blk_end+1].to_numpy()
        tp_local = (tp_abs - t0).astype('timedelta64[s]').astype(float) / (7*24*3600.0)
        if tp_local.size == 0:
            print(f"[{year} | step {step_id}] Empty tp; breaking.")
            break

        tp_dense, xp_dense = model.predict(tp_local, tinfer, xinfer, random=False)
        tp_dense = np.asarray(tp_dense).reshape(-1)
        xp_dense = np.asarray(xp_dense)

        if tp_dense.size == tp_local.size and np.allclose(tp_dense, tp_local):
            y_pred_local = np.column_stack([inv_scaler_fns[i](xp_dense[:, i]) for i in range(D)])
        else:
            y_pred_local = np.empty((tp_local.shape[0], D), dtype=float)
            for i in range(D):
                y_dense_unscaled = inv_scaler_fns[i](xp_dense[:, i])
                y_pred_local[:, i] = np.interp(tp_local, tp_dense, y_dense_unscaled)

        yhat_full[cur_cut:blk_end+1, :] = y_pred_local

        # 6) pseudo carry
        if INCLUDE_PSEUDO:
            pseudo_dates = np.concatenate([pseudo_dates, tp_abs.astype('datetime64[ns]')])
            for i in range(D):
                pseudo_Y[i] = np.concatenate([pseudo_Y[i], y_pred_local[:, i]])

        cur_cut = blk_end + 1
        step_id += 1

    # === 여기서는 "그리진 않고" 결과와 메타만 리턴 ===
    return {
        "year": year,
        "dates": df_full["date"].to_numpy(),         # 절대 날짜
        "X_full": X_full,                            # 실제값
        "yhat_full": yhat_full,                      # 예측(스티치)
        "idx_cut": idx_cut                           # Sep 1 cut 위치 (절대 인덱스)
    }


# -------------------- 5) Run for target years and overlay plot --------------------
target_years = [2023, 2024]

# 기존 year_windows를 이용해 각 연도의 (idx_cut, idx_end) 찾기
runs = {y: None for y in target_years}
for y, idx_cut, idx_end in year_windows(df["date"]):
    if y in runs:
        runs[y] = (idx_cut, idx_end)

# 두 해 모두 있어야 그림을 그린다
available = {y: tup for y, tup in runs.items() if tup is not None}
if len(available) == 0:
    raise RuntimeError("No target years found in the data windowing (Sep→Aug).")

results = {}
for y, (idx_cut, idx_end) in available.items():
    df_slice = df.iloc[:idx_end+1].copy()
    res = run_one_year_rolling(df_slice, idx_cut=idx_cut, idx_end=idx_end, year=y)
    if res is not None:
        results[y] = res

if len(results) == 0:
    raise RuntimeError("No results to plot.")

# --------- Overlay plot: x-axis = absolute dates, 2 rows (components), curves per year ---------
# --------- Overlay plot: x-axis = absolute dates, 2 rows (components), unified forecast styling ---------
cols = list(mcolors.TABLEAU_COLORS.values())
fig, axes = plt.subplots(len(COMPONENTS), 1, figsize=(12, 7), sharex=False, facecolor='w')

if len(COMPONENTS) == 1:
    axes = [axes]

for i, comp in enumerate(COMPONENTS):
    ax = axes[i]
    ax.set_facecolor('#f5f5f5')
    ax.set_title(f"Flu rolling forecasts — {comp}")
    ax.set_ylabel(comp)

    # choose a stable color per component for FORECAST; use a lighter/different tone for TRUE points
    forecast_color = cols[i % len(cols)]
    true_color = cols[(i + 4) % len(cols)]  # just to separate visually

    # add each legend label only once per component
    added_true_label = False
    added_forecast_label = False

    for _, res in sorted(results.items()):
        dates_abs = res["dates"]
        true_vals = res["X_full"][:, i]
        preds     = res["yhat_full"][:, i]
        idx_cut   = res["idx_cut"]

        # TRUE points
        ax.scatter(
            dates_abs, true_vals, s=10, alpha=0.45, color=true_color,
            label=("True (weekly)" if not added_true_label else None)
        )
        added_true_label = True

        # FORECAST (both years same color & one legend entry)
        mask = ~np.isnan(preds)
        if mask.any():
            ax.plot(dates_abs[mask], preds[mask], lw=2.0, alpha=0.9, color=forecast_color,
                    label=("Forecast (stitched 4w)" if not added_forecast_label else None))
            ax.scatter(dates_abs[mask], preds[mask], s=14, alpha=0.9, color=forecast_color)
            added_forecast_label = True

        # optional: keep the Sep 1 cut markers
        ax.axvline(dates_abs[idx_cut], linestyle=':', color='k', lw=1.0)

    # --- Date ticks: every 6 months + tilted labels ---
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))     # 6-month spacing
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    for label in ax.get_xticklabels():
        label.set_rotation(30)                                      # tilt labels
        label.set_horizontalalignment('right')
    ax.grid(True, which='both', axis='x', alpha=0.25)

axes[-1].set_xlabel('Date')
axes[0].legend(loc='best', ncol=2)
plt.tight_layout()

if SAVE_FIGS:
    os.makedirs(FIG_DIR, exist_ok=True)
    out = f"{FIG_DIR}/fmagi_flu_rolling_overlay_2023_2024.png"
    plt.savefig(out, dpi=160)
    print(f"Saved: {out}")

plt.close(fig)