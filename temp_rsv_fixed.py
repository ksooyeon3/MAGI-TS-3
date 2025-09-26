# === MAGI-X (RSV 1D lifted to 2D): rolling 4-week forecasts (Sep Y → Aug Y+1), retrain each step ===
# Key fixes:
#  - Each step trains on a LOCAL time axis (t=0 at that step's start)
#  - Pseudo-observations are stored in ABSOLUTE dates and converted to the local axis per step
#  - Removed the "offset" re-anchoring hack to prevent repeated-looking segments
#  - FMAGI requires >=2 dims → when D=1 (RSV), we duplicate the single component internally (MODEL_D=2) and read back comp 0

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import time, os
import matplotlib.colors as mcolors

from scripts.magix.dynamic import nnSTModule, nnMTModule  # keep import paths
from scripts.magix.inference import FMAGI                 # keep import paths

# -------------------------- 0) Config --------------------------
DATA_CSV   = "/home/giung/MAGI-TS-3/data/rsv_hosp.csv"
SEED       = 188714368
GRID_SIZE  = 201
INTERP_ORD = 3
MAX_EPOCH  = 2500
LR         = 1e-3
COMPONENTS = ["hosp_US"]  # RSV file typically has just this column
SAVE_FIGS  = True
FIG_DIR    = "./figs_fmagi_yearly_rsv_hosp_3week_fixed"
INCLUDE_PSEUDO = True    # use previous-step forecasts as pseudo-obs
START_YEAR = 2025
STOP_YEAR  = None  # or e.g., 2017

# Scaling: choose one (per component)
SCALE = "minmax5"        # "minmax5" | "zscore" | None
#   minmax5: X_scaled = 5 * (X - min_train) / (max_train - min_train)

# Rolling forecast step
STEP_WEEKS = 3          # forecast horizon per step

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

# require listed components
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
def run_one_year_rolling(df_slice, idx_cut, idx_end, year):
    if idx_cut < 3:   # need at least a few points to train
        print(f"[{year}] Not enough pre–Sep-1 training points (<3). Skipping this year.")
        return

    df_full = df_slice.iloc[:idx_end+1].copy()
    dates = df_full["date"].reset_index(drop=True)
    X_full = df_full[COMPONENTS].to_numpy().astype(float)   # (N, D)
    D = X_full.shape[1]
    MODEL_D = max(D, 2)  # FMAGI requires >=2 dims; duplicate when D=1

    # absolute/local plotting axis (weeks since window start)
    t_full_weeks = (dates - dates.iloc[0]).dt.total_seconds().to_numpy() / (86400.0 * 7)

    # stitched forecast container: (N, D)
    yhat_full = np.full_like(X_full, fill_value=np.nan, dtype=float)

    # pseudo-observations buffers:
    # - store ABSOLUTE dates here (datetime64[ns])
    # - store values UNSCALED in the original units
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

        # 1) current true training slice (UNSCALED) — ABSOLUTE dates → LOCAL axis
        t_train_abs = dates.iloc[:cur_cut].to_numpy()                 # datetime64[ns]
        t0 = t_train_abs[0]
        t_train_local = (t_train_abs - t0).astype('timedelta64[s]').astype(float) / (7*24*3600.0)
        X_train_true = X_full[:cur_cut, :]   # (cur_cut, D)

        # 2) per-component scaler fit (optionally include pseudo) — fit in VALUE space
        scaler_fns, inv_scaler_fns = [], []
        for i in range(D):
            if INCLUDE_PSEUDO and pseudo_dates.size > 0 and pseudo_Y[i].size > 0:
                f, finv, _ = fit_scaler_params(
                    np.concatenate([X_train_true[:, i], pseudo_Y[i]]), SCALE
                )
            else:
                f, finv, _ = fit_scaler_params(X_train_true[:, i], SCALE)
            scaler_fns.append(f); inv_scaler_fns.append(finv)

        # 3) build obs = true [+ pseudo], scaled per component — pseudo mapped to THIS STEP'S local axis
        obs = []
        for i in range(MODEL_D):
            base_i = min(i, D - 1)  # if D=1 → always 0 (duplicate series)
            if INCLUDE_PSEUDO and pseudo_dates.size > 0 and pseudo_Y[base_i].size > 0:
                tPi_local = (pseudo_dates - t0).astype('timedelta64[s]').astype(float) / (7*24*3600.0)
                t_obs_local = np.concatenate([t_train_local, tPi_local])
                y_obs = np.concatenate([X_train_true[:, base_i], pseudo_Y[base_i]])
            else:
                t_obs_local = t_train_local.copy()
                y_obs = X_train_true[:, base_i].copy()

            # align & sort (safety)
            t_obs_local, y_obs = _align_pairs(t_obs_local, y_obs)
            y_obs_scaled = scaler_fns[base_i](y_obs)
            obs.append(np.column_stack([t_obs_local, y_obs_scaled]))

        # 4) fit FMAGI (MODEL_D dims)
        torch.manual_seed(SEED)
        fOde = nnMTModule(MODEL_D, [512], dp=0.0)
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
        # ---- force shapes: tinfer -> (T,), xinfer -> (T, MODEL_D) ----
        tinfer = np.asarray(tinfer).reshape(-1)     # (T,)
        xinfer = np.asarray(xinfer)                 # could be (T, D), (D, T), (T,), etc.

        if xinfer.ndim == 1:
            xinfer = xinfer.reshape(-1, MODEL_D)  # (T,) → (T, MODEL_D)
        elif xinfer.shape[0] == MODEL_D and xinfer.shape[1] != MODEL_D:
            xinfer = xinfer.T                     # (MODEL_D, T) → (T, MODEL_D)
        elif xinfer.shape[1] != MODEL_D:
            xinfer = xinfer.reshape(-1, MODEL_D)

        # final safety checks
        if xinfer.shape[0] != tinfer.shape[0]:
            raise ValueError(f"xinfer time length {xinfer.shape[0]} != tinfer length {tinfer.shape[0]}")
        if xinfer.shape[1] != MODEL_D:
            raise ValueError(f"xinfer state dim {xinfer.shape[1]} != MODEL_D {MODEL_D}")

        # 5) forecast ONLY [cur_cut:blk_end] — build THIS STEP'S local grid and predict on it
        tp_abs = dates.iloc[cur_cut:blk_end+1].to_numpy()  # absolute datetimes
        tp_local = (tp_abs - t0).astype('timedelta64[s]').astype(float) / (7*24*3600.0)

        if tp_local.size == 0:
            print(f"[{year} | step {step_id}] Empty tp; breaking.")
            break

        tp_dense, xp_dense = model.predict(tp_local, tinfer, xinfer, random=False)
        tp_dense = np.asarray(tp_dense).reshape(-1)
        xp_dense = np.asarray(xp_dense)  # (M_dense, MODEL_D)

        # align (interpolate if model returns a denser grid)
        if MODEL_D == D:
            # normal multi-D case (e.g., flu 2D)
            if tp_dense.size == tp_local.size and np.allclose(tp_dense, tp_local):
                y_pred_local = np.column_stack([inv_scaler_fns[i](xp_dense[:, i]) for i in range(D)])
            else:
                y_pred_local = np.empty((tp_local.shape[0], D), dtype=float)
                for i in range(D):
                    y_dense_unscaled = inv_scaler_fns[i](xp_dense[:, i])
                    y_pred_local[:, i] = np.interp(tp_local, tp_dense, y_dense_unscaled)
        else:
            # duplicated 1D → 2D model: use comp 0 only, then write to D=1
            if tp_dense.size == tp_local.size and np.allclose(tp_dense, tp_local):
                y0_unscaled = inv_scaler_fns[0](xp_dense[:, 0])
                y_pred_local = y0_unscaled.reshape(-1, 1)
            else:
                y0_unscaled = inv_scaler_fns[0](xp_dense[:, 0])
                y_pred_local = np.empty((tp_local.shape[0], 1), dtype=float)
                y_pred_local[:, 0] = np.interp(tp_local, tp_dense, y0_unscaled)

        # write predictions into the stitched year matrix (indices are absolute)
        yhat_full[cur_cut:blk_end+1, :] = y_pred_local

        # 6) carry pseudo to next round (ABSOLUTE dates; values UNSCALED)
        if INCLUDE_PSEUDO:
            pseudo_dates = np.concatenate([pseudo_dates, tp_abs.astype('datetime64[ns]')])
            for i in range(D):
                pseudo_Y[i] = np.concatenate([pseudo_Y[i], y_pred_local[:, i if D > 1 else 0]])

        # 7) advance
        cur_cut = blk_end + 1
        step_id += 1

    # -------------------- Plotting (stitched single line per component) --------------------
    cols = list(mcolors.TABLEAU_COLORS.values())
    nrows = len(COMPONENTS)
    fig, axes = plt.subplots(nrows, 1, figsize=(11, 3.5*nrows), facecolor='w', sharex=True)

    # axes can be a single Axes if nrows==1; normalize to a list
    if nrows == 1:
        axes = [axes]

    for ax in axes:
        ax.set_facecolor('#f5f5f5')

    for i, comp in enumerate(COMPONENTS):
        ax = axes[i]
        # ground truth (dots)
        ax.scatter(t_full_weeks, X_full[:, i],
                   s=10, marker='o', color=cols[i % len(cols)], alpha=0.55,
                   label=f"True (weekly) — {comp}")

        # stitched forecast (points + connected line)
        mask = ~np.isnan(yhat_full[:, i])
        if mask.any():
            ax.scatter(t_full_weeks[mask], yhat_full[mask, i],
                       s=14, marker='o', alpha=0.9, color=cols[(i+1) % len(cols)],
                       label="Forecast (points)")
            ax.plot(t_full_weeks[mask], yhat_full[mask, i],
                    lw=2.0, alpha=0.6, color=cols[(i+1) % len(cols)],
                    label=f"Forecast (stitched {STEP_WEEKS}w)")

        ax.axvline(x=t_full_weeks[idx_cut], linestyle=':', color='k', lw=1.0, label='Initial cut (Sep 1)')
        ax.set_title(f'[{year}] Rolling forecast (Sep→Aug) — {comp}')
        ax.set_ylabel(comp)
        ax.legend(loc='best')

    axes[-1].set_xlabel('Weeks since window start')
    plt.tight_layout()

    if SAVE_FIGS:
        os.makedirs(FIG_DIR, exist_ok=True)
        out = f"{FIG_DIR}/fmagi_rsv_rolling_{year}.png"
        plt.savefig(out, dpi=160)
        print(f"Saved: {out}")

    plt.close(fig)

# -------------------- 5) Run for all years (reverse order) --------------------
runs = list(year_windows(df["date"]))
runs = [r for r in runs if r[0] <= START_YEAR and (STOP_YEAR is None or r[0] >= STOP_YEAR)]

for year, idx_cut, idx_end in sorted(runs, key=lambda t: t[0], reverse=True):
    df_slice = df.iloc[:idx_end+1].copy()  # window start → Aug31 (Y+1)
    run_one_year_rolling(df_slice, idx_cut=idx_cut, idx_end=idx_end, year=year)
