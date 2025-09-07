
# 0. imports
from operator import truediv
import time, numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from scripts.magix.dynamic import nnSTModule, nnMTModule  # MAGI-X NN module
from scripts.magix.inference import FMAGI                 # MAGI-X inference

#Hes1 도 log transformed data로 바꿔야함!!! (아직 안바꿈)


# 1. UQ metric utilities

# ---------- UQ metric helpers ----------
def interp_truth_to(t_truth, X_truth, t_eval):
    D = X_truth.shape[1]
    X_eval = np.column_stack([np.interp(t_eval, t_truth, X_truth[:, d]) for d in range(D)])
    return X_eval

def percentile_band(samples, alpha=0.10):
    lo = np.percentile(samples, 100*(alpha/2), axis=0)
    hi = np.percentile(samples, 100*(1 - alpha/2), axis=0)
    return lo, hi

def coverage_and_width(samples, t_full, t_truth, X_truth, t_end_fit, alpha=0.10):
    """
    samples: (N, T, D)  stochastic trajectories (e.g., MC-dropout)
    t_full:  (T,)       time grid of samples (initial point included)
    t_truth: (T0,)      truth grid
    X_truth: (T0,D)     truth values
    """
    X_eval = interp_truth_to(t_truth, X_truth, t_full)  # (T,D)
    lo, hi = percentile_band(samples, alpha=alpha)      # (T,D)
    width  = hi - lo
    inside = (X_eval >= lo) & (X_eval <= hi)

    fit_mask = t_full <= t_end_fit
    fct_mask = ~fit_mask

    def summarize(mask):
        cov   = inside[mask].mean()
        w     = width[mask].mean()
        # normalized width: divide per-dim by dynamic range in the region
        Xm    = X_eval[mask]
        rng   = Xm.max(axis=0) - Xm.min(axis=0)
        rng[rng == 0] = 1.0
        w_norm = (width[mask].mean(axis=0) / rng).mean()
        return cov, w, w_norm

    cov_fit, w_fit, wN_fit = summarize(fit_mask)
    cov_fct, w_fct, wN_fct = summarize(fct_mask)

    # per-dim breakdown (handy for appendix tables)
    per_dim = []
    D = X_truth.shape[1]
    for d in range(D):
        pdict = dict(
            cov_fit = inside[fit_mask, d].mean(),
            cov_fct = inside[fct_mask, d].mean(),
            w_fit   = width[fit_mask, d].mean(),
            w_fct   = width[fct_mask, d].mean(),
        )
        per_dim.append(pdict)

    return dict(
        lo=lo, hi=hi, X_eval=X_eval,
        coverage_fit=cov_fit, coverage_fct=cov_fct,
        width_fit=w_fit, width_fct=w_fct,
        width_norm_fit=wN_fit, width_norm_fct=wN_fct,
        per_dim=per_dim
    )

# 2) MC-dropout sampler for MAGI-X

# ---------- MC-dropout sampler ----------
def mc_dropout_samples(model, fOde, trecon, xinfer, N=50):
    """
    Returns:
      t_full: (T,)
      samples: (N, T, D)
    """
    # keep dropout ON
    fOde.train()
    x0 = xinfer[0, :].squeeze()

    # one deterministic call to fix t_full and T
    tr1, xr1 = model.predict(trecon[1:], trecon[:1], x0, random=True)
    t_full = np.concatenate([trecon[:1], tr1])
    D = xr1.shape[1]
    T = t_full.size

    # collect stochastic draws (dropout is active)
    draws = np.empty((N, T, D), dtype=float)
    draws[0] = np.vstack([x0.reshape(1, -1), xr1])

    for i in range(1, N):
        tri, xri = model.predict(trecon[1:], trecon[:1], x0, random=True)
        draws[i] = np.vstack([x0.reshape(1, -1), xri])

    return t_full, draws


# 3) ODEs + truth simulators (FN / LV (log) / Hes1)

# ---------- Dynamics ----------
def FN(y, t, a, b, c):
    V, R = y
    dVdt = c * (V - V**3/3.0 + R)
    dRdt = -1.0/c * (V - a + b*R)
    return (dVdt, dRdt)

def LV_log(y, t, a, b, c, d):
    # state in log-space
    x1, x2 = np.exp(y)
    dx1dt = a*x1 - b*x1*x2
    dx2dt = c*x1*x2 - d*x2
    return [dx1dt/x1, dx2dt/x2]  # chain rule

def Hes1(y, t, a, b, c, d, e, f, g):
    P, M, H = y
    dPdt = -a*P*H + b*M - c*P
    dMdt = -d*M + e/(1 + P**2)
    dHdt = -a*P*H + f/(1 + P**2) - g*H
    return (dPdt, dMdt, dHdt)

def Hes1_log(y, t, a, b, c, d, e, f, g):
    # y = log([P, M, H])
    P, M, H = np.exp(y)

    dPdt = -a*P*H + b*M - c*P
    dMdt = -d*M + e/(1 + P**2)
    dHdt = -a*P*H + f/(1 + P**2) - g*H

    # dy/dt = (dx/dt) / x  (체인룰)
    return [dPdt / P, dMdt / M, dHdt / H]
# ---------- Truth simulators ----------
def simulate_FN():
    a, b, c = 0.2, 0.2, 3.0
    V0, R0 = -1.0, 1.0
    t = np.linspace(0, 40, 1281)
    X = odeint(FN, (V0, R0), t, args=(a, b, c))
    return t, X

def simulate_LV_log():
    a, b, c, d = 1.5, 1.0, 1.0, 3.0
    x1_0, x2_0 = 5.0, 0.2
    y0 = np.log([x1_0, x2_0])
    t = np.linspace(0, 12, 321)
    Y = odeint(LV_log, y0, t, args=(a, b, c, d))  # log-state
    # X = np.column_stack([Y[:,0], Y[:,1]])  
    X = Y
    return t, X

def simulate_Hes1():
    a, b, c, d, e, f, g = 0.022, 0.3, 0.031, 0.028, 0.5, 20.0, 0.3
    P0, M0, H0 = 1.438575, 2.037488, 17.90385
    t = np.linspace(0, 640, 1281)
    X = odeint(Hes1, (P0, M0, H0), t, args=(a, b, c, d, e, f, g))
    return t, X

def simulate_Hes1_log():
    a, b, c, d, e, f, g = 0.022, 0.3, 0.031, 0.028, 0.5, 20.0, 0.3
    P0, M0, H0 = 1.438575, 2.037488, 17.90385
    y0 = np.log([P0, M0, H0])           # 로그 초기값
    t = np.linspace(0, 640, 1281)
    Y = odeint(Hes1_log, y0, t, args=(a, b, c, d, e, f, g))  # 로그 상태로 적분
    return t, Y  # Y는 log-space (모델/관측과 일관)

# 4) Observation maker (same style you used)

def make_observations(t, X, no_train, noise, seed=0):
    """
    t: (T,), X: (T,D)
    noise: list-like of length D (std dev per component)
    returns: list of arrays [(n_i, 2) ...] with columns (t, y_obs)
    and the last observation time (fit boundary).
    """
    np.random.seed(seed)
    T, D = X.shape
    # observe first half of the horizon unless you change obs_idx
    obs_idx = np.linspace(0, (T-1)//2, no_train).astype(int)
    obs = []
    for d in range(D):
        tobs = t[obs_idx].copy()
        yobs = X[obs_idx, d].copy() + np.random.normal(0, noise[d], size=no_train)
        obs.append(np.c_[tobs, yobs])
    t_end_fit = max(o[:,0].max() for o in obs)
    return obs, t_end_fit

# 5) One-system pipeline (fit → samples → metrics)

def run_system(system_name,
               simulate_fn,
               nn_hidden=512,
               grid_size=161,
               max_epoch=2500,
               lr=1e-3,
               dropout_p=0.1,
               N_train=41,
               noise=None,
               seed=188714368,
               N_samples=50):
    """
    Returns a dict with metrics + a small table row for the paper.
    """
    # 1) truth & observations
    t_truth, X_truth = simulate_fn()
    D = X_truth.shape[1]
    if noise is None: noise = [0.1]*D
    obs, t_end_fit = make_observations(t_truth, X_truth, N_train, noise, seed=seed)

    # 2) fit MAGI-X (MT = multi-task)
    torch.manual_seed(seed)
    fOde = nnMTModule(D, [nn_hidden], dp=dropout_p)
    model = FMAGI(obs, fOde, grid_size=grid_size, interpolation_orders=3)

    # inference (MAP trajectory)
    trecon = t_truth[np.linspace(0, t_truth.size-1, 321).astype(int)]
    _, xinfer = model.map(max_epoch=max_epoch,
                          learning_rate=lr, decay_learning_rate=True,
                          hyperparams_update=False, dynamic_standardization=True,
                          verbose=False, returnX=True)

    # 3) MC-dropout samples
    t_full, samples = mc_dropout_samples(model, fOde, trecon, xinfer, N=N_samples)

    # 4) metrics
    m = coverage_and_width(samples, t_full, t_truth, X_truth, t_end_fit, alpha=0.10)

    # 5) compact rows for paper table
    row_fit = dict(system=system_name, region='fit',
                   coverage_90=m['coverage_fit'], width=m['width_fit'], width_norm=m['width_norm_fit'],
                   N_samples=N_samples, T=t_full.size, D=D, N_train=N_train)
    row_fct = dict(system=system_name, region='forecast',
                   coverage_90=m['coverage_fct'], width=m['width_fct'], width_norm=m['width_norm_fct'],
                   N_samples=N_samples, T=t_full.size, D=D, N_train=N_train)

    return dict(metrics=m, t_full=t_full, samples=samples, obs=obs,
                t_truth=t_truth, X_truth=X_truth,
                rows=[row_fit, row_fct])


# 6) Orchestrator: run FN / LV / Hes1 and print a paper-ready table
def run_all_systems(save_csv_path=None):
    jobs = [
        dict(name='FN',   sim=simulate_FN,     N_train=41, noise=[0.1, 0.1],
             lr=1e-3, max_epoch=2500, dropout_p=0.1),
        dict(name='LV',   sim=simulate_LV_log, N_train=41, noise=[0.1, 0.1],
             lr=1e-3, max_epoch=2500, dropout_p=0.1),
        dict(name='Hes1', sim=simulate_Hes1_log,   N_train=41, noise=[0.1, 0.1, 0.1],
             lr=5e-4, max_epoch=2500, dropout_p=0.1),
    ]

    # defaults
    DEF = dict(nn_hidden=512, grid_size=161, max_epoch=2500, lr=1e-3,
               dropout_p=0.1, N_samples=50)

    rows, per_system = [], {}
    for jb in jobs:
        # resolve params for this job
        params = dict(
            nn_hidden = jb.get('nn_hidden', DEF['nn_hidden']),
            grid_size = jb.get('grid_size', DEF['grid_size']),
            max_epoch = jb.get('max_epoch', DEF['max_epoch']),
            lr        = jb.get('lr',        DEF['lr']),
            dropout_p = jb.get('dropout_p', DEF['dropout_p']),
            N_train   = jb.get('N_train',   41),
            noise     = jb.get('noise',     None),
            N_samples = jb.get('N_samples', DEF['N_samples']),
        )

        out = run_system(
            system_name = jb['name'],
            simulate_fn = jb['sim'],
            **params
        )

        # add hyperparams into each row
        for row in out['rows']:
            row.update(dict(
                lr=params['lr'],
                max_epoch=params['max_epoch'],
                dropout_p=params['dropout_p'],
                nn_hidden=params['nn_hidden'],
                grid_size=params['grid_size'],
            ))
            rows.append(row)

        per_system[jb['name']] = out

    df = pd.DataFrame(rows)

    # nice formatting for paper display
    df_display = df.copy()
    df_display['coverage_90'] = (100*df_display['coverage_90']).map(lambda x: f"{x:5.1f}%")
    df_display['width']       = df_display['width'].map(lambda x: f"{x:.3f}")
    df_display['width_norm']  = df_display['width_norm'].map(lambda x: f"{x:.3f}")

    # put hyperparams in the printed table too if you want
    df_display = df_display[['system','region','coverage_90','width','width_norm',
                             'N_train','N_samples','T','D',
                             'lr','max_epoch','dropout_p','nn_hidden','grid_size']]

    print("\n=== Coverage@90% and Band Width (Fit vs Forecast) ===")
    print(df_display.to_string(index=False))

    if save_csv_path:
        df.to_csv(save_csv_path, index=False)
        print(f"\n[Saved raw metrics to {save_csv_path}]")

    return df, per_system




import os
import numpy as np
import matplotlib.pyplot as plt

# consistent colors for up to 3 dims
_COLS = ['b','r','g']

def _last_obs_time(obs):
    return max(o[:,0].max() for o in obs)

def plot_uq_single_axis(sys_out, system_name='System', save_path=None, alpha=0.10):
    """
    One axis: mean prediction + 90% band, truth (dotted), observations (points),
    and a vertical line marking the fit/forecast split.
    """
    t = sys_out['t_full']                  # (T,)
    samples = sys_out['samples']           # (N,T,D)
    obs = sys_out['obs']                   # list of (n_i,2)
    t_truth, X_truth = sys_out['t_truth'], sys_out['X_truth']

    # summarize
    mean_traj = samples.mean(axis=0)       # (T,D)
    lo = np.percentile(samples, 100*(alpha/2), axis=0)
    hi = np.percentile(samples, 100*(1 - alpha/2), axis=0)
    t_split = _last_obs_time(obs)

    # metrics for title footer
    m = sys_out['metrics']
    cov_fit  = 100*m['coverage_fit']
    cov_fct  = 100*m['coverage_fct']
    wd_fit   = m['width_norm_fit']
    wd_fct   = m['width_norm_fct']

    fig, ax = plt.subplots(figsize=(6.2, 4.6), facecolor='w')
    ax.set_facecolor('#dddddd')

    D = mean_traj.shape[1]
    for d in range(D):
        c = _COLS[d % len(_COLS)]
        ax.fill_between(t, lo[:, d], hi[:, d], color=c, alpha=0.18, lw=0, label=f'90% band (dim {d+1})' if d==0 else None)
        ax.plot(t, mean_traj[:, d], c, lw=2.0, alpha=0.95, label=f'pred (dim {d+1})')
        ax.plot(t_truth, X_truth[:, d], c + ':', lw=1.1, alpha=0.9, label=f'truth (dim {d+1})' if d==0 else None)
        ax.scatter(obs[d][:,0], obs[d][:,1], s=9, color=c, edgecolor='none', zorder=3,
                   label=f'obs (dim {d+1})' if d==0 else None)

    # fit/forecast separator + light shading on forecast
    ax.axvline(t_split, ls='--', color='k', alpha=0.45)
    ax.axvspan(t_split, t[-1], color='k', alpha=0.04)

    # limits & legend
    ymin = np.min(X_truth) - 0.12*np.ptp(X_truth)
    ymax = np.max(X_truth) + 0.12*np.ptp(X_truth)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(t[0], t[-1])
    ax.set_title(f'{system_name}: Inferred + Forecast with 90% Bands')
    ax.legend(loc='best', fontsize=9, framealpha=0.85)

    # small calibration footer
    ax.text(0.01, -0.18,
            f'Coverage@90 — FIT: {cov_fit:.1f}% (norm width {wd_fit:.3f}) | FORECAST: {cov_fct:.1f}% (norm width {wd_fct:.3f})',
            transform=ax.transAxes, fontsize=9)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=240, bbox_inches='tight')
    plt.show()


def plot_uq_grid(sys_out, system_name='System', save_path=None, alpha=0.10):
    """
    One subplot per dimension. Shows 90% band, mean, truth, obs.
    Handy when D=3 (e.g., Hes1).
    """
    t = sys_out['t_full']
    samples = sys_out['samples']
    obs = sys_out['obs']
    t_truth, X_truth = sys_out['t_truth'], sys_out['X_truth']
    mean_traj = samples.mean(axis=0)
    lo = np.percentile(samples, 100*(alpha/2), axis=0)
    hi = np.percentile(samples, 100*(1 - alpha/2), axis=0)
    t_split = _last_obs_time(obs)

    D = mean_traj.shape[1]
    fig, axes = plt.subplots(D, 1, figsize=(6.2, 1.9*D+1.0), sharex=True, facecolor='w')
    if D == 1: axes = [axes]
    for d, ax in enumerate(axes):
        ax.set_facecolor('#eeeeee')
        c = _COLS[d % len(_COLS)]
        ax.fill_between(t, lo[:, d], hi[:, d], color=c, alpha=0.20, lw=0)
        ax.plot(t, mean_traj[:, d], c, lw=2.0, alpha=0.95)
        ax.plot(t_truth, X_truth[:, d], c + ':', lw=1.1, alpha=0.9)
        ax.scatter(obs[d][:,0], obs[d][:,1], s=9, color=c, edgecolor='none', zorder=3)

        ax.axvline(t_split, ls='--', color='k', alpha=0.45)
        ax.axvspan(t_split, t[-1], color='k', alpha=0.04)
        ax.set_ylabel(f'dim {d+1}')
        ymin = np.min(X_truth[:, d]) - 0.12*np.ptp(X_truth[:, d])
        ymax = np.max(X_truth[:, d]) + 0.12*np.ptp(X_truth[:, d])
        ax.set_ylim(ymin, ymax)

    axes[-1].set_xlabel('time')
    fig.suptitle(f'{system_name}: 90% Uncertainty Bands (per dimension)', y=0.99)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=240, bbox_inches='tight')
    plt.show()


def plot_all_systems(per_system, out_dir='uq_figs'):
    """
    per_system: dict returned by run_all_systems()[1]
                keys should include 'FN', 'LV', 'Hes1' (or whatever you used)
    Saves two figures per system: single-axis and grid.
    """
    os.makedirs(out_dir, exist_ok=True)
    for name, sys_out in per_system.items():
        # single axis
        plot_uq_single_axis(
            sys_out, system_name=name,
            save_path=os.path.join(out_dir, f'{name}_uq_single.png')
        )
        # per-dimension grid
        plot_uq_grid(
            sys_out, system_name=name,
            save_path=os.path.join(out_dir, f'{name}_uq_grid.png')
        )
    print(f'[Saved figures under {out_dir}/]')


df_metrics, per_system = run_all_systems(save_csv_path="uq_metrics.csv")
plot_all_systems(per_system, out_dir='uq_figs')





