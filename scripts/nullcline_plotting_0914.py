# import system package
import os
import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import torch
from optparse import OptionParser

# Add TensorFlow imports for npode
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_probability as tfp
import types
tf.contrib = types.SimpleNamespace(
    distributions=types.SimpleNamespace(
        MultivariateNormalFullCovariance = tfp.distributions.MultivariateNormalFullCovariance,
        MultivariateNormalDiag           = tfp.distributions.MultivariateNormalDiag
    )
)

# Add torchdiffeq import for nrode
from torchdiffeq.adjoint import odeint_adjoint as odeint
from torchdiffeq import dynamic as nrode_dynamic

# import customize lib
import utils # experiment

# import magix
from magix.dynamic import nnMTModule
from magix.inference import FMAGI # inferred module

# import npode
from npode import npde_helper # inferred module

# Add matplotlib for dynamics analysis
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt

# DYNAMICS ANALYSIS FUNCTIONS
# ============================

def fitzhugh_nagumo_true(y, a=0.2, b=0.2, c=3.0):
    """FitzHugh-Nagumo true dynamics"""
    V, R = y
    dVdt = c * (V - V**3/3.0 + R)
    dRdt = -1.0/c * (V - a + b*R)
    return np.array([dVdt, dRdt])

def lotka_volterra_log_true(y, a=1.5, b=1.0, c=1.0, d=3.0):
    """Lotka-Volterra in log space true dynamics"""
    x1, x2 = np.exp(y)
    dx1dt = a*x1 - b*x1*x2
    dx2dt = c*x1*x2 - d*x2
    return np.array([dx1dt/x1, dx2dt/x2])

def hes1_true(y, a=0.022, b=0.3, c=0.031, d=0.028, e=0.5, f=20.0, g=0.3):
    """Hes1 true dynamics (only first 2 components for 2D visualization)"""
    P, M = y[:2]  # Only use first 2 components for 2D
    H = 17.90385  # Use equilibrium value for H in 2D projection
    dPdt = -a*P*H + b*M - c*P
    dMdt = -d*M + e/(1 + P**2)
    return np.array([dPdt, dMdt])


def draw_analytic_nullclines(ax, system, xlim, ylim):
    """
    Draw analytic nullclines for the true system on given axis.
    For FN:    V' = 0  => R = V^3/3 - V
               R' = 0  => R = (V - a)/b
    For LV(log-space): y1' = 0 => y2 = ln(a/b);  y2' = 0 => y1 = ln(d/c)
    For Hes1(2D proj): P' = 0 => M = ((aH + c)/b) P ;  M' = 0 => M = e/(d(1+P^2))
    """
    print(f"system is {system}")
    
    if system == "FN" or system =="fn":
        print("system FN inside draw_analytic_nullclines function ")
        # FitzHugh-Nagumo parameters
        a, b, c = 0.2, 0.2, 3.0
        
        # V-nullcline: V' = 0 => c * (V - V^3/3 + R) = 0 => R = V^3/3 - V
        x_vals = np.linspace(-3, 3, 1000)
        y_vals_v = x_vals**3/3 - x_vals
        # Only plot within y limits
        # valid_idx = (y_vals_v >= ylim[0]) & (y_vals_v <= ylim[1])
        # if np.any(valid_idx):
        ax.plot(x_vals, y_vals_v, 'r--', linewidth=2.0, 
                   label='True V-nullcline', alpha=0.8)
        
        # R-nullcline: R' = 0 => -(V - a + b*R)/c = 0 => R = (V - a)/b
        y_vals_r = (a - x_vals) / b
        # valid_idx = (y_vals_r >= ylim[0]) & (y_vals_r <= ylim[1])
        # if np.any(valid_idx):
        ax.plot(x_vals, y_vals_r, 'g--', linewidth=2.0, 
                   label='True R-nullcline', alpha=0.8)
                   
    elif system == "LV" or system == "lv":
        # Lotka-Volterra parameters  
        a, b, c, d = 1.5, 1.0, 1.0, 3.0
        
        # y1-nullcline: y1' = 0 => a - b*exp(y2) = 0 => y2 = ln(a/b)
        y2_null = np.log(a/b)
        if ylim[0] <= y2_null <= ylim[1]:
            ax.axhline(y=y2_null, color='r', linestyle='--', linewidth=2.0,
                      label='True y1-nullcline', alpha=0.8)
        
        # y2-nullcline: y2' = 0 => c*exp(y1) - d = 0 => y1 = ln(d/c)  
        y1_null = np.log(d/c)
        if xlim[0] <= y1_null <= xlim[1]:
            ax.axvline(x=y1_null, color='g', linestyle='--', linewidth=2.0,
                      label='True y2-nullcline', alpha=0.8)
                      
    elif system == "Hes1":
        # Hes1 parameters
        a, b, c, d, e, f, g = 0.022, 0.3, 0.031, 0.028, 0.5, 20.0, 0.3
        H = 17.90385  # equilibrium value for H
        
        # P-nullcline: P' = 0 => -a*P*H + b*M - c*P = 0 => M = (a*H + c)*P/b
        p_vals = np.linspace(xlim[0], xlim[1], 1000)
        m_vals_p = (a*H + c) * p_vals / b
        valid_idx = (m_vals_p >= ylim[0]) & (m_vals_p <= ylim[1]) & (p_vals >= xlim[0]) & (p_vals <= xlim[1])
        if np.any(valid_idx):
            ax.plot(p_vals[valid_idx], m_vals_p[valid_idx], 'r--', linewidth=2.0,
                   label='True P-nullcline', alpha=0.8)
        
        # M-nullcline: M' = 0 => -d*M + e/(1+P^2) = 0 => M = e/(d*(1+P^2))
        m_vals_m = e / (d * (1 + p_vals**2))
        valid_idx = (m_vals_m >= ylim[0]) & (m_vals_m <= ylim[1]) & (p_vals >= xlim[0]) & (p_vals <= xlim[1])
        if np.any(valid_idx):
            ax.plot(p_vals[valid_idx], m_vals_m[valid_idx], 'g--', linewidth=2.0,
                   label='True M-nullcline', alpha=0.8)
            

def plot_error_fields(magix_model=None, nrode_func=None, npode_model=None, npode_sess=None,
                     result_dir=".", run_id=0, system="FN", train_points=None):
    """
    Plot error fields showing the difference between learned and true derivatives.
    Uses consistent color scale across all models for proper comparison.
    """
    
    # Set plot limits and true function based on system
    if system == "FN" or system == "fn":
        xlim, ylim = (-3, 3), (-3, 3)
        true_fn = lambda y: fitzhugh_nagumo_true(y)
        system_name = "FitzHugh-Nagumo"
    elif system == "LV" or system == "lv": 
        xlim, ylim = (-1, 3), (-2, 2)
        true_fn = lambda y: lotka_volterra_log_true(y)
        system_name = "Lotka-Volterra"
    elif system == "Hes1":
        xlim, ylim = (0, 5), (0, 30)  
        true_fn = lambda y: hes1_true(y)
        system_name = "Hes1"
    else:
        xlim, ylim = (-3, 3), (-3, 3)
        true_fn = None
        system_name = "Unknown"
    
    if true_fn is None:
        print("No true dynamics function available for error field analysis")
        return
    
    # Create grid for error field analysis
    x_vals = np.linspace(xlim[0], xlim[1], 20)
    y_vals = np.linspace(ylim[0], ylim[1], 20)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Collect models to analyze
    models_to_analyze = []
    if magix_model is not None:
        models_to_analyze.append(("MAGIX", magix_model, "magix"))
    if nrode_func is not None:
        models_to_analyze.append(("Neural ODE", nrode_func, "nrode"))  
    if npode_model is not None:
        models_to_analyze.append(("NPODE", npode_model, "npode"))
    
    n_models = len(models_to_analyze)
    if n_models == 0:
        print("No models available for error field analysis")
        return
    
    # FIRST PASS: Calculate error magnitudes for all models to determine global scale
    all_error_magnitudes = []
    model_errors = []  # Store computed errors for reuse
    
    for name, model, model_type in models_to_analyze:
        print(f"Computing error field for {name}...")
        
        # Initialize arrays for this model
        U_error = np.zeros_like(X)
        V_error = np.zeros_like(Y)
        
        # Compute derivatives at each grid point
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                state = np.array([X[i, j], Y[i, j]])
                
                # Get true derivatives
                try:
                    true_derivs = true_fn(state)
                except Exception as e:
                    true_derivs = np.array([0.0, 0.0])
                
                # Get predicted derivatives
                try:
                    if model_type == "magix":
                        with torch.no_grad():
                            model.fOde.eval()
                            state_torch = torch.tensor(state, dtype=torch.float64).unsqueeze(0)
                            pred_derivs = model.fOde(state_torch).numpy().flatten()
                    elif model_type == "nrode":
                        with torch.no_grad():
                            state_torch = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                            t_dummy = torch.tensor([0.0])
                            pred_derivs = model(t_dummy, state_torch).numpy().flatten()
                    elif model_type == "npode":
                        try:
                            import tensorflow.compat.v1 as tf
                            state_tf = tf.constant(state.reshape(1, -1), dtype=tf.float64)
                            derivs_tf = model.f(state_tf, [0.0])
                            
                            if npode_sess is not None:
                                pred_derivs = npode_sess.run(derivs_tf).flatten()
                            else:
                                sess = tf.get_default_session()
                                if sess is not None:
                                    pred_derivs = sess.run(derivs_tf).flatten()
                                else:
                                    pred_derivs = np.array([0.0, 0.0])
                        except Exception as e:
                            pred_derivs = np.array([0.0, 0.0])
                    
                except Exception as e:
                    pred_derivs = np.array([0.0, 0.0])
                
                # Calculate error (predicted - true)
                U_error[i, j] = pred_derivs[0] - true_derivs[0]
                V_error[i, j] = pred_derivs[1] - true_derivs[1]
        
        # Calculate error magnitude for this model
        Error_magnitude = np.hypot(U_error, V_error)
        all_error_magnitudes.append(Error_magnitude)
        model_errors.append((name, model_type, U_error, V_error, Error_magnitude))
    
    # Calculate global min/max for consistent color scale across all models
    global_max_error = max(np.max(mag) for mag in all_error_magnitudes)
    global_min_error = 0.0  # Error magnitude is always non-negative
    
    print(f"Global error range for consistent color scale: [{global_min_error:.4f}, {global_max_error:.4f}]")
    
    # SECOND PASS: Create plots with consistent color scale
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    for idx, (name, model_type, U_error, V_error, Error_magnitude) in enumerate(model_errors):
        ax = axes[idx]
        
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=global_min_error, vmax=global_max_error)
        
        # Plot error field with CONSISTENT color scale across all models
        Q = ax.quiver(X, Y, U_error, V_error, Error_magnitude, 
                     cmap='Reds', scale=global_max_error, scale_units='xy', angles='xy',
                     width=0.003, alpha=0.8,
                     norm = norm)  # KEY FIX: consistent scale
        Q.set_clim(vmin=global_min_error, vmax=global_max_error)
        
        # Add colorbar with consistent scale
        cbar = plt.colorbar(Q, ax=ax)
        cbar.set_label("Error Magnitude", fontsize=10)
        
        # Add contour lines for error magnitude using GLOBAL scale
        try:
            contour_levels = np.linspace(global_min_error, global_max_error, 6)  # KEY FIX: global scale
            ax.contour(X, Y, Error_magnitude, levels=contour_levels, 
                      colors='black', linewidths=0.5, alpha=0.3)
        except Exception:
            pass
        
        # Scatter training observations if provided
        if train_points is not None:
            try:
                ax.scatter(train_points[:,0], train_points[:,1], 
                          s=15, c='blue', alpha=0.7, label='Training Points', 
                          zorder=5, edgecolors='white', linewidths=0.5)
            except Exception:
                pass
        
        # Set limits and labels
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('x₁', fontsize=12)
        ax.set_ylabel('x₂', fontsize=12)
        ax.set_title(f'{name} - Derivative Error Field', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add legend if training points are shown
        if train_points is not None:
            try:
                ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
            except Exception:
                pass
        
        # Print statistics
        max_error = np.max(Error_magnitude)
        mean_error = np.mean(Error_magnitude)
        print(f"{name} - Max error: {max_error:.4f}, Mean error: {mean_error:.4f}")
    
    # Set overall title
    plt.suptitle(f'{system_name} System - Derivative Error Fields (Run {run_id})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    save_path = f"{result_dir}/error_fields_run_{run_id}_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Error field analysis saved to: {save_path}")
    
    return save_path
def plot_heatmap(magix_model=None, nrode_func=None, npode_model=None, npode_sess=None,
                 result_dir=".", run_id=0, system="FN", train_points=None):
    """
    Plot heatmaps showing the absolute magnitude of errors between learned and true derivatives.
    Uses consistent color scale across all models for proper comparison.
    """
    
    # Set plot limits and true function based on system
    if system == "FN" or system == "fn":
        xlim, ylim = (-3, 3), (-3, 3)
        true_fn = lambda y: fitzhugh_nagumo_true(y)
        system_name = "FitzHugh-Nagumo"
    elif system == "LV" or system == "lv": 
        xlim, ylim = (-1, 3), (-2, 2)
        true_fn = lambda y: lotka_volterra_log_true(y)
        system_name = "Lotka-Volterra"
    elif system == "Hes1":
        xlim, ylim = (0, 5), (0, 30)  
        true_fn = lambda y: hes1_true(y)
        system_name = "Hes1"
    else:
        xlim, ylim = (-3, 3), (-3, 3)
        true_fn = None
        system_name = "Unknown"
    
    if true_fn is None:
        print("No true dynamics function available for heatmap analysis")
        return
    
    # Create grid for heatmap analysis
    x_vals = np.linspace(xlim[0], xlim[1], 40)  # Higher resolution for smoother heatmap
    y_vals = np.linspace(ylim[0], ylim[1], 40)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Collect models to analyze
    models_to_analyze = []
    if magix_model is not None:
        models_to_analyze.append(("MAGIX", magix_model, "magix"))
    if nrode_func is not None:
        models_to_analyze.append(("Neural ODE", nrode_func, "nrode"))  
    if npode_model is not None:
        models_to_analyze.append(("NPODE", npode_model, "npode"))
    
    n_models = len(models_to_analyze)
    if n_models == 0:
        print("No models available for heatmap analysis")
        return
    
    # FIRST PASS: Calculate error magnitudes for all models to determine global scale
    all_error_magnitudes = []
    model_errors = []  # Store computed errors for reuse
    
    for name, model, model_type in models_to_analyze:
        print(f"Computing error heatmap for {name}...")
        
        # Initialize arrays for this model
        U_error = np.zeros_like(X)
        V_error = np.zeros_like(Y)
        
        # Compute derivatives at each grid point
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                state = np.array([X[i, j], Y[i, j]])
                
                # Get true derivatives
                try:
                    true_derivs = true_fn(state)
                except Exception as e:
                    true_derivs = np.array([0.0, 0.0])
                
                # Get predicted derivatives
                try:
                    if model_type == "magix":
                        with torch.no_grad():
                            model.fOde.eval()
                            state_torch = torch.tensor(state, dtype=torch.float64).unsqueeze(0)
                            pred_derivs = model.fOde(state_torch).numpy().flatten()
                    elif model_type == "nrode":
                        with torch.no_grad():
                            state_torch = torch.tensor(state, dtype=torch.float64).unsqueeze(0)
                            t_dummy = torch.tensor([0.0], dtype=torch.float64)
                            pred_derivs = model(t_dummy, state_torch).numpy().flatten()
                    elif model_type == "npode":
                        try:
                            import tensorflow.compat.v1 as tf
                            state_tf = tf.constant(state.reshape(1, -1), dtype=tf.float64)
                            derivs_tf = model.f(state_tf, [0.0])
                            
                            if npode_sess is not None:
                                pred_derivs = npode_sess.run(derivs_tf).flatten()
                            else:
                                sess = tf.get_default_session()
                                if sess is not None:
                                    pred_derivs = sess.run(derivs_tf).flatten()
                                else:
                                    pred_derivs = np.array([0.0, 0.0])
                        except Exception as e:
                            pred_derivs = np.array([0.0, 0.0])
                    
                except Exception as e:
                    pred_derivs = np.array([0.0, 0.0])
                
                # Calculate error (predicted - true)
                U_error[i, j] = pred_derivs[0] - true_derivs[0]
                V_error[i, j] = pred_derivs[1] - true_derivs[1]
        
        # Calculate error magnitude for this model (absolute value for heatmap)
        Error_magnitude = np.hypot(U_error, V_error)
        all_error_magnitudes.append(Error_magnitude)
        model_errors.append((name, model_type, U_error, V_error, Error_magnitude))
    
    # Calculate global min/max for consistent color scale across all models
    global_max_error = max(np.max(mag) for mag in all_error_magnitudes)
    global_min_error = 0.0  # Error magnitude is always non-negative
    
    print(f"Global error range for consistent color scale: [{global_min_error:.4f}, {global_max_error:.4f}]")
    
    # SECOND PASS: Create heatmap plots with consistent color scale
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    for idx, (name, model_type, U_error, V_error, Error_magnitude) in enumerate(model_errors):
        ax = axes[idx]
        
        # Create heatmap of absolute error magnitudes
        im = ax.imshow(Error_magnitude, 
                      extent=[xlim[0], xlim[1], ylim[0], ylim[1]], 
                      origin='lower', 
                      cmap='Reds', 
                      vmin=global_min_error, 
                      vmax=global_max_error,
                      alpha=0.8,
                      interpolation='bilinear')  # Smooth interpolation for better appearance
        
        # Add colorbar with consistent scale
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Error Magnitude", fontsize=10)
        
        # Add contour lines for error magnitude using GLOBAL scale
        try:
            contour_levels = np.linspace(global_min_error, global_max_error, 6)
            cs = ax.contour(X, Y, Error_magnitude, levels=contour_levels, 
                           colors='black', linewidths=0.5, alpha=0.4)
            # Optionally add contour labels
            ax.clabel(cs, inline=True, fontsize=8, fmt='%.3f')
        except Exception:
            pass
        
        # Calculate quantitative error metrics
        max_error = np.max(Error_magnitude)
        mean_error = np.mean(Error_magnitude)
        rmse_error = np.sqrt(np.mean(Error_magnitude**2))  # RMSE of error magnitudes
        std_error = np.std(Error_magnitude)
        median_error = np.median(Error_magnitude)
        
        # Calculate component-wise RMSE for derivatives
        rmse_u = np.sqrt(np.mean(U_error**2))  # RMSE for first derivative component
        rmse_v = np.sqrt(np.mean(V_error**2))  # RMSE for second derivative component
        
        # Calculate percentage of points with high error (above 75th percentile of global max)
        high_error_threshold = 0.75 * global_max_error
        high_error_percentage = np.sum(Error_magnitude > high_error_threshold) / Error_magnitude.size * 100
        
        # Create text box with error statistics
        stats_text = (f'Error Statistics:\n'
                     f'RMSE (total): {rmse_error:.4f}\n'
                     f'RMSE (dx₁/dt): {rmse_u:.4f}\n'
                     f'RMSE (dx₂/dt): {rmse_v:.4f}'
                    #  f'Max: {max_error:.4f}\n'
                    #  f'Mean: {mean_error:.4f}\n'
                    #  f'Median: {median_error:.4f}\n'
                    #  f'Std: {std_error:.4f}\n'
                    #  f'High error: {high_error_percentage:.1f}%'
                    )       
        
        
        # Add text box to plot (positioned in lower left)
        from matplotlib.patches import Rectangle
        text_props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='bottom', bbox=text_props, family='monospace')
        
        # Scatter training observations if provided
        if train_points is not None:
            try:
                ax.scatter(train_points[:,0], train_points[:,1], 
                          s=20, c='blue', alpha=0.9, label='Training Points', 
                          zorder=5, edgecolors='white', linewidths=1.0)
            except Exception:
                pass
        
        # Set limits and labels
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('x₁', fontsize=12)
        ax.set_ylabel('x₂', fontsize=12)
        ax.set_title(f'{name} - Error Magnitude Heatmap', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add legend if training points are shown
        if train_points is not None:
            try:
                ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
            except Exception:
                pass
        
         # Print statistics to console
        print(f"{name} Error Analysis:")
        print(f"  RMSE (total): {rmse_error:.4f}")
        print(f"  RMSE (dx₁/dt): {rmse_u:.4f}")
        print(f"  RMSE (dx₂/dt): {rmse_v:.4f}")
        print(f"  Max error: {max_error:.4f}")
        print(f"  Mean error: {mean_error:.4f}")
        print(f"  Median error: {median_error:.4f}")
        print(f"  Std error: {std_error:.4f}")
        print(f"  High error regions: {high_error_percentage:.1f}% of field")
        print()
    
    
    # Set overall title
    plt.suptitle(f'{system_name} System - Derivative Error Magnitude Heatmaps (Run {run_id})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    save_path = f"{result_dir}/error_heatmaps_run_{run_id}_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Error heatmap analysis saved to: {save_path}")
    
    return save_path

def debug_nrode_model(nrode_func, test_states=None):
    """Debug NRODE model to ensure it's working correctly"""
    if test_states is None:
        test_states = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0], [0.5, -0.5]])
    
    print("=== NRODE Model Debug ===")
    print(f"Model type: {type(nrode_func)}")
    print(f"Model parameters dtype: {next(nrode_func.parameters()).dtype}")
    
    nrode_func.eval()
    with torch.no_grad():
        for i, state in enumerate(test_states):
            try:
                state_torch = torch.tensor(state, dtype=torch.float64).unsqueeze(0)
                t_dummy = torch.tensor([0.0], dtype=torch.float64)
                
                derivs = nrode_func(t_dummy, state_torch)
                derivs_np = derivs.squeeze(0).detach().numpy()
                
                print(f"State {state} -> Derivatives {derivs_np}")
                
                # Check if derivatives are all zeros (potential issue)
                if np.allclose(derivs_np, 0.0):
                    print(f"  WARNING: All derivatives are zero!")
                    
            except Exception as e:
                print(f"  ERROR at state {state}: {e}")
    
    print("========================")
    return True

def quick_vector_field_analysis(magix_model=None, nrode_func=None, npode_model=None, npode_sess=None,
                               result_dir=".", run_id=0, system="FN", train_points=None):
    """
    Quick analysis that generates vector field plots for trained models.
    """
    
    # Set plot limits based on system
    if system == "FN" or system == "fn":
        xlim, ylim = (-3, 3), (-3, 3)
        true_fn = lambda y: fitzhugh_nagumo_true(y)
    elif system == "LV" or system == "lv": 
        xlim, ylim = (-1, 3), (-2, 2)
        true_fn = lambda y: lotka_volterra_log_true(y)
    elif system == "Hes1":
        xlim, ylim = (0, 5), (0, 30)  
        true_fn = lambda y: hes1_true(y)
    else:
        xlim, ylim = (-3, 3), (-3, 3)
        true_fn = None
    
    # Grids for vector field
    x_c = np.linspace(xlim[0], xlim[1], 15)   # coarse grid for quiver
    y_c = np.linspace(ylim[0], ylim[1], 15)
    Xc, Yc = np.meshgrid(x_c, y_c)
    
    models_to_plot = []
    if magix_model is not None:
        models_to_plot.append(("MAGIX", magix_model, "magix"))
    if nrode_func is not None:
        models_to_plot.append(("Neural ODE", nrode_func, "nrode"))  
    if npode_model is not None:
        models_to_plot.append(("NPODE", npode_model, "npode"))
    if true_fn is not None:
        models_to_plot.append(("True System", true_fn, "true"))
    
    n_models = len(models_to_plot)
    if n_models == 0:
        return
        
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
    if n_models == 1:
        axes = [axes]
        
    # Define uniform color scale parameters
    
    if system == "FN" or system == "fn":
        vmin, vmax = 0, 5
    elif system == "LV" or system == "lv":
        vmin, vmax = 0, 8
    elif system == "Hes1":
        vmin, vmax = 0, 10  
    else:
        vmin, vmax = 0, 5  # Default fallback

    overflow_color = 'lightgray'
    
    # Create custom colormap that handles overflow
    import matplotlib.colors as mcolors
    from matplotlib.cm import viridis
    
    # Get the viridis colormap
    base_cmap = plt.cm.viridis
    
    for idx, (name, model, model_type) in enumerate(models_to_plot):
        ax = axes[idx]
        
        # Compute vector field
        U = np.zeros_like(Xc)
        V = np.zeros_like(Yc)    
        
        for i in range(Xc.shape[0]):
            for j in range(Xc.shape[1]):
                state = np.array([Xc[i, j], Yc[i, j]])
                
                try:
                    if model_type == "magix":
                        with torch.no_grad():
                            model.fOde.eval()
                            state_torch = torch.tensor(state, dtype=torch.float64).unsqueeze(0)
                            derivs = model.fOde(state_torch).numpy().flatten()
                    # elif model_type == "nrode":
                    #     with torch.no_grad():
                    #         state_torch = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    #         t_dummy = torch.tensor([0.0])
                    #         derivs = model(t_dummy, state_torch).numpy().flatten()
                    elif model_type == "nrode":
                        try:
                            with torch.no_grad():
                                # IMPORTANT: ODEFunc uses .double() so we need float64 tensors
                                state_torch = torch.tensor(state, dtype=torch.float64).unsqueeze(0)  # Shape: (1, 2)
                                t_dummy = torch.tensor([0.0], dtype=torch.float64)  # Time (not used by ODEFunc but required)
                                
                                # Set model to evaluation mode
                                model.eval()
                                
                                # Call the model: func(t, y) -> dy/dt
                                derivs_torch = model(t_dummy, state_torch)  # Returns (1, 2)
                                
                                # Convert to numpy and flatten
                                derivs = derivs_torch.squeeze(0).detach().numpy()  # Shape: (2,)
                                
                        except Exception as e:
                            print(f"NRODE evaluation error at state {state}: {e}")
                            derivs = np.array([0.0, 0.0])
                    elif model_type == "npode":
                        try:
                            import tensorflow.compat.v1 as tf
                            state_tf = tf.constant(state.reshape(1, -1), dtype=tf.float64)
                            derivs_tf = model.f(state_tf, [0.0])
                            
                            # Use the training session to evaluate
                            if npode_sess is not None:
                                derivs = npode_sess.run(derivs_tf).flatten()
                            else:
                                print("!!!No TensorFlow session available for NPODE evaluation!!!")
                                # Fallback: try to get default session
                                sess = tf.get_default_session()
                                if sess is not None:
                                    derivs = sess.run(derivs_tf).flatten()
                                else:
                                    print("No TensorFlow session available for NPODE evaluation")
                                    derivs = np.array([0.0, 0.0])
                                    
                        except Exception as e:
                            print(f"NPODE evaluation error: {e}")
                            derivs = np.array([0.0, 0.0])
                            
                    elif model_type == "true":
                        derivs = model(state)
                    
                    U[i, j] = derivs[0] 
                    V[i, j] = derivs[1]
                    
                except Exception as e:
                    U[i, j] = 0
                    V[i, j] = 0
        
        # Compute magnitude for coloring
        M = np.hypot(U, V)
        
        # Create masked array where values > 5 are masked out
        M_masked = np.ma.masked_where(M > vmax, M)
        
        # Create arrays for overflow points (magnitude > 5)
        overflow_mask = M > vmax
        
        # Subsample quiver to reduce clutter
        QUIVER_EVERY = 1
        s = (slice(None, None, QUIVER_EVERY), slice(None, None, QUIVER_EVERY))
        Q = ax.quiver(Xc[s], Yc[s], U[s], V[s], M_masked[s], cmap='viridis', scale=25, width=0.003, pivot='mid')
        Q.set_clim(vmin=vmin, vmax=vmax)
        
        # Plot overflow points in light gray
        if np.any(overflow_mask):
            Q_overflow = ax.quiver(Xc[overflow_mask], Yc[overflow_mask], 
                                 U[overflow_mask], V[overflow_mask], 
                                 color=overflow_color, scale=25, width=0.003, pivot='mid',
                                 alpha=0.7, label=f'Magnitude > {vmax}')
        
        
        # Add colorbar with uniform scale (only for the first subplot to avoid repetition)
        if idx == n_models -1 :
            cbar = plt.colorbar(Q, ax=ax)
            cbar.set_label("Magnitude", fontsize=10)
            cbar.set_ticks(np.arange(vmin, vmax+1))
        
        # cbar = plt.colorbar(Q, ax=ax)
        # cbar.set_label("Magnitude")
        
        # Overlay estimated nullclines from the current model (using contours)
        # try:
        #     ax.contour(Xc, Yc, U, levels=[0], colors='red', linewidths=2.0, alpha=0.6)
        # except Exception:
        #     pass
        # try:
        #     ax.contour(Xc, Yc, V, levels=[0], colors='green', linewidths=2.0, alpha=0.6)
        # except Exception:
        #     pass
        
        # Draw analytic true nullclines
        # try:
        #     draw_analytic_nullclines(ax, system, xlim, ylim)
        #     print(f"Model {name}: Successfully plotted analytic nullclines")
        # except Exception as e:
        #     print(f"Model {name}: Failed to plot analytic nullclines: {e}")
        
        # Scatter training observations if provided
        if train_points is not None:
            try:
                ax.scatter(train_points[:,0], train_points[:,1], s=10, c='k', alpha=0.6, label='Observations', zorder=5)
            except Exception:
                pass
                
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        # Legend
        try:
            ax.legend(loc='upper left', framealpha=0.9, fontsize=8)
        except Exception:
            pass
            
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title(f'{name}')
        ax.grid(True, alpha=0.3)
        
        # Add text showing the range of magnitudes in this plot
        mag_min, mag_max = np.min(M), np.max(M)
        # ax.text(0.02, 0.98, f'Mag range: [{mag_min:.2f}, {mag_max:.2f}]', 
        #        transform=ax.transAxes, verticalalignment='top',
        #        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
        #        fontsize=8)
        ax.text(0.98, 0.02, f'Mag range: [{mag_min:.2f}, {mag_max:.2f}]', 
        transform=ax.transAxes, verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
        fontsize=8)
    
    
    plt.suptitle(f'Vector Fields Comparison - Run {run_id}', fontsize=14)
    plt.tight_layout()
    
    # Add timestamp import at the top of the function
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save plot
    save_path = f"{result_dir}/vector_fields_run_{run_id}_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Vector field analysis saved to: {save_path}")


def analyze_final_models_with_error_fields(magix_model=None, npode_model=None, npode_sess=None, nrode_func=None,
                        result_dir="results", run_id=0, system_name="FN", train_points=None,
                        include_error_fields=False, include_heatmap=True):
    """
    Analyze dynamics for trained models
    """
    
    print(f"Analyzing learned dynamics for run {run_id}...")
    
    # Only analyze 2D systems for now
    if system_name.startswith("Hes1") and system_name != "Hes1":
        print("Skipping dynamics analysis for non-2D Hes1 system")
        return
    if nrode_func is not None:
        print("Debugging NRODE model before analysis...")
        debug_nrode_model(nrode_func)
    try:
        # Generate vector field comparison
        quick_vector_field_analysis(magix_model=magix_model, nrode_func=nrode_func, npode_model=npode_model, npode_sess=npode_sess, train_points=train_points,
            result_dir=result_dir,
            run_id=run_id,
            system=system_name
        )
        # Generate error field analysis if requested
        if include_error_fields:
            plot_error_fields(
                magix_model=magix_model,
                nrode_func=nrode_func,
                npode_model=npode_model,
                npode_sess=npode_sess,
                train_points=train_points,
                result_dir=result_dir,
                run_id=run_id,
                system=system_name
            )
        if include_heatmap:
            plot_heatmap(
                magix_model=magix_model,
                nrode_func=nrode_func,
                npode_model=npode_model,
                npode_sess=npode_sess,
                train_points=train_points,
                result_dir=result_dir,
                run_id=run_id,
                system=system_name
            )
            
        print("Dynamics analysis completed successfully!")
        
    except Exception as e:
        print(f"Error in dynamics analysis: {e}")
        import traceback
        traceback.print_exc()

def main():
    # read in option
    usage = "usage:%prog [options] arg"
    parser = OptionParser(usage)
    parser.add_option("-p", "--parameter", dest = "parameter_file_path",
                      type = "string", help = "path to the parameter file")
    parser.add_option("-r", "--result_dir", dest = "result_dir",
                      type = "string", help = "path to the result directory")
    parser.add_option("-s", dest = "random_seed", default = None,
                      type = "string", help = "random seed")
    parser.add_option("--error_fields", dest = "error_fields", action="store_true",
                        default=False, help = "generate error field analysis plots")
    parser.add_option("--heatmaps", dest="heatmaps", action="store_true",
                  default=True, help="generate error heatmap analysis plots")
    
    (options, args) = parser.parse_args()

    # process parser information
    result_dir = options.result_dir
    if (not os.path.exists(result_dir)):
        os.makedirs(result_dir, exist_ok=True)
    seed = options.random_seed
    if (seed is None):
        seed = (np.datetime64('now').astype('int')*104729) % 1e9
        seed = str(int(seed))
        
    # Get error field flag
    include_error_fields = options.error_fields
    if include_error_fields:
        print("Error field analysis will be generated")


    # read in parameters
    parameters = utils.params()
    parameters.read(options.parameter_file_path)
    if (parameters.get('experiment','seed') is None):
        parameters.add('experiment','seed',seed)

    # read in data
    example = parameters.get('data','example')
    if (example is None):
        raise ValueError('parameter data: example must be provided!')
    data = np.loadtxt('data/321/%s.txt' %(example))
    tdata = data[:,0] # time
    xdata = data[:,1:] # component values
    no_comp = xdata.shape[1] # number of components
    # read in number of trainning points
    no_train = parameters.get('data','no_train')
    if (no_train is None):
        no_train = int((tdata.size-1)/2) + 1
        parameters.add('data','no_train',no_train)
    no_train = int(no_train)
    FIT_END = int((tdata.size-1)/2) + 1  # 161 when tdata.size == 321
    obs_idx = np.linspace(0,int((tdata.size-1)/2),no_train).astype(int)
    # obtain noise parameters
    noise = parameters.get('data','noise')
    noise = [float(x) for x in noise.split(',')]
    if (len(noise) != no_comp):
        if (len(noise) == 1):
            noise = [noise[0] for i in range(no_comp)]
        else:
            raise ValueError('noise parameters must have %d components!' %(no_comp))

    # read in experiment set up
    no_run = int(parameters.get('experiment','no_run'))
    # initialize/read in random seed for the data noise
    seed = int(parameters.get('experiment','seed'))
    exp_seed = utils.seed()
    exp_seed_file = parameters.get('experiment','seed_file')
    if (exp_seed_file is None):
        exp_seed.random(no_run, seed)
    else:
        exp_seed.load(exp_seed_file)
    exp_seed_file = os.path.join(result_dir, 'exp_seed.txt')
    exp_seed.save(exp_seed_file)
    parameters.add('experiment','seed_file',exp_seed_file)

    # read in model flag and model set-up
    # magix
    magix_run = parameters.get('magix','run')
    if (magix_run == 'yes'):
        magix_run = True
        # magix number of iterations
        magix_no_iter = int(parameters.get('magix','no_iteration'))
        # magix robust parameter
        magix_robust_eps = float(parameters.get('magix','robust_eps'))
        # magix parameters
        magix_node = parameters.get('magix','hidden_node')
        magix_node = [no_comp] + [int(x) for x in magix_node.split(',')] + [no_comp]
        # set up heading for the output file
        magix_output_path = os.path.join(result_dir, 'magix.txt')
        magix_output = open(magix_output_path, 'w')
        magix_output.write('run,time')
        # heading for the output files
        for i in range(no_comp):
            for ptype in ['imputation','forecast','overall']:
                magix_output.write(',rmse_c%d_%s' %(i,ptype))
        magix_output.write('\n')
        magix_output.close()
    else:
        magix_run = False

    # npode
    npode_run = parameters.get('npode','run')
    if (npode_run == 'yes'):
        npode_run = True
        # npode number of iterations
        npode_no_iter = int(parameters.get('npode','no_iteration'))
        # set up heading for the output file
        npode_output_path = os.path.join(result_dir, 'npode.txt')
        npode_output = open(npode_output_path, 'w')
        npode_output.write('run,time')
        # heading for the output files
        for i in range(no_comp):
            for ptype in ['imputation','forecast','overall']:
                npode_output.write(',rmse_c%d_%s' %(i,ptype))
        npode_output.write('\n')
        npode_output.close()
    else:
        npode_run = False

    # neural ode
    nrode_run = parameters.get('nrode','run')
    if (nrode_run == 'yes'):
        nrode_run = True
        # neural ode number of iterations
        nrode_no_iter = int(parameters.get('nrode','no_iteration'))
        # neural ode parameters
        nrode_node = parameters.get('nrode','hidden_node')
        nrode_node = [int(x) for x in nrode_node.split(',')]
        nrode_node = [no_comp] + nrode_node + [no_comp]
        # set up heading for the output file
        nrode_output_path = os.path.join(result_dir, 'nrode.txt')
        nrode_output = open(nrode_output_path, 'w')
        nrode_output.write('run,time')
        # heading for the mse
        for i in range(no_comp):
            for ptype in ['imputation','forecast','overall']:
                nrode_output.write(',rmse_c%d_%s' %(i,ptype)) # reconstructed only
        nrode_output.write('\n')
        nrode_output.close()
    else:
        nrode_run = False

    # save parameters file
    parameters.save(result_dir)

    # Initialize variables to store final models for dynamics analysis
    final_magix_model = None
    final_npode_model = None
    final_nrode_func = None

    # run experiment
    for k in range(no_run):
        print(f"Starting run {k+1}/{no_run}")
        
        # data preprocessing
        obs = []
        np.random.seed(exp_seed.get(k)) # set random seed for noise
        for i in range(no_comp):
            tobs = tdata[obs_idx].copy()
            yobs = xdata[obs_idx,i].copy() + np.random.normal(0,noise[i],no_train)
            obs.append(np.hstack((tobs.reshape(-1,1),yobs.reshape(-1,1))))

        # Prepare data for npode/nrode (they need full observation vectors)
        tobs_full = tdata[obs_idx].copy()
        yobs_full = xdata[obs_idx,:].copy()
        for i in range(no_comp):
            yobs_full[:,i] = yobs_full[:,i] + np.random.normal(0,noise[i],no_train)

        # run models
        # magix
        if (magix_run):
            print('running magix...')
            # set random seed
            torch.manual_seed(exp_seed.get(k))
            # inference/learning
            start_time = time.time()
            fOde = nnMTModule(no_comp, 512) # define nn dynamic
            magix_model = FMAGI(obs,fOde,grid_size=161,interpolation_orders=3)
            tinfer, xinfer = magix_model.map(max_epoch=magix_no_iter,
                    learning_rate=1e-3, decay_learning_rate=True,
                    hyperparams_update=False, dynamic_standardization=True,
                    verbose=True, returnX=True)
            end_time = time.time()
            
            # --- Compute inferred + forecast errors (no full reconstruction) ---
            # In-sample (imputation) error: compare inferred xinfer (161 x D) to ground truth first 161 points
            # Assumes tinfer aligns with tdata[:FIT_END] (default in FMAGI with grid_size=161)
            # Forecast: integrate forward from the last inferred state
            # Use t0 as last fit time, x0 as last inferred state
            t0_fore = tdata[FIT_END-1:FIT_END]       # shape (1,)
            x0_fore = xinfer[FIT_END-1:FIT_END, :]    # shape (1, D)
            tp_fore = tdata[FIT_END:]                 # shape (160,)

            # Predict only the forecasting segment
            t_pred, x_fore = magix_model.predict(
                tp=tp_fore,  # forecast times
                t0=t0_fore,  # initial time (last fit time)
                x0=x0_fore,  # initial state (last inferred state)
                random=False
            )
        
            # Align forecast arrays to ground-truth length (160)
            if x_fore.shape[0] == tp_fore.shape[0] + 1:
                x_fore = x_fore[1:, :]
                t_pred = t_pred[1:]

            run_time = end_time - start_time

            # Write RMSEs: imputation = inferred-fit; forecast = forecast segment; overall = concat
            magix_output = open(magix_output_path, 'a')
            magix_output.write('%d,%s' %(k,run_time))
            for i in range(no_comp):
                # Inferred (fit) error over first 161 points
                xi_err_fit = xinfer[:,i] - xdata[:FIT_END,i]
                rmse_fit = np.sqrt(np.mean(np.square(xi_err_fit)))
                magix_output.write(',%s' %(rmse_fit))

                # Forecast error over remaining 160 points
                xi_err_fore = x_fore[:,i] - xdata[FIT_END:,i]
                rmse_fore = np.sqrt(np.mean(np.square(xi_err_fore)))
                magix_output.write(',%s' %(rmse_fore))

                # Overall error across fit+forecast
                xi_err_all = np.concatenate([xi_err_fit, xi_err_fore], axis=0)
                rmse_all = np.sqrt(np.mean(np.square(xi_err_all)))
                magix_output.write(',%s' %(rmse_all))
            magix_output.write('\n')
            magix_output.close()

            # Store final model for dynamics analysis
            if k == no_run - 1:
                final_magix_model = magix_model
            else:
                # release memory for non-final runs
                del fOde
                del magix_model

        # npode
        if (npode_run):
            print('running npode...')
            # npode cannot handle (partial) missing data
            # remove missing data by taking the observation of first index
            npode_tobs = [tobs_full]
            npode_yobs = [yobs_full]
            tf.reset_default_graph()
            sess = tf.InteractiveSession()
            tf.set_random_seed(exp_seed.get(k))
            # inference/learning
            start_time = time.time()
            npode = npde_helper.build_model(sess, npode_tobs, npode_yobs, model='ode', sf0=1.0, ell0=np.ones(no_comp), W=6, ktype="id")
            x0, npode = npde_helper.fit_model(sess, npode, npode_tobs, npode_yobs, num_iter=npode_no_iter, print_every=100,eta=0.02, plot_=False)
            end_time = time.time()
            xrecon = npode.predict(x0,tdata).eval() # reconstruction
            
            # performance evaluation
            run_time = end_time - start_time
            npode_output = open(npode_output_path, 'a')
            npode_output.write('%d,%s' %(k,run_time))
            for i in range(no_comp):
                xi_error = xrecon[:,i] - xdata[:,i]
                xi_rmse_imputation = np.sqrt(np.mean(np.square(xi_error[:FIT_END])))
                npode_output.write(',%s' %(xi_rmse_imputation))
                xi_rmse_forecast = np.sqrt(np.mean(np.square(xi_error[FIT_END:])))
                npode_output.write(',%s' %(xi_rmse_forecast))
                xi_rmse_overall = np.sqrt(np.mean(np.square(xi_error)))
                npode_output.write(',%s' %(xi_rmse_overall))
            npode_output.write('\n')
            npode_output.close()
            
            # Store final model for dynamics analysis
            if k == no_run - 1:
                final_npode_model = npode
                # Keep session open for dynamics analysis
                final_npode_sess = sess
            else:
                sess.close()
                # release memory
                tf.get_default_graph().finalize()
                tf.reset_default_graph()
                del npode
                del sess

        # neural ode
        if (nrode_run):
            print('running nrode...')
            # neural ode cannot handle (partial) missing data
            # remove missing data by taking the observation of first index
            nrode_tobs = torch.tensor(tobs_full)
            nrode_yobs = torch.tensor(yobs_full).unsqueeze(1)
            # set random seed
            torch.manual_seed(exp_seed.get(k))
            # inference/learning
            start_time = time.time()
            func = nrode_dynamic.ODEFunc(nrode_node) # define neural network dynamic
            x0 = nrode_yobs[0]
            x0.requires_grad_(True)
            optimizer = torch.optim.RMSprop(list(func.parameters())+[x0], lr=1e-3)
            for itr in range(nrode_no_iter):
                optimizer.zero_grad()
                ypred = odeint(func, x0, nrode_tobs)
                loss = torch.mean(torch.abs(ypred - nrode_yobs))
                loss.backward()
                optimizer.step()
            end_time = time.time()
            with torch.no_grad():
                xrecon = odeint(func, x0, torch.tensor(tdata))
            xrecon = xrecon.detach().squeeze().numpy()
            run_time = end_time - start_time
            nrode_output = open(nrode_output_path, 'a')
            nrode_output.write('%d,%s' %(k,run_time))
            for i in range(no_comp):
                xi_error = xrecon[:,i] - xdata[:,i]
                xi_rmse_imputation = np.sqrt(np.mean(np.square(xi_error[:FIT_END])))
                nrode_output.write(',%s' %(xi_rmse_imputation))
                xi_rmse_forecast = np.sqrt(np.mean(np.square(xi_error[FIT_END:])))
                nrode_output.write(',%s' %(xi_rmse_forecast))
                xi_rmse_overall = np.sqrt(np.mean(np.square(xi_error)))
                nrode_output.write(',%s' %(xi_rmse_overall))
            nrode_output.write('\n')
            nrode_output.close()
            
            # Store final model for dynamics analysis
            if k == no_run - 1:
                final_nrode_func = func
            else:
                # release memory for non-final runs
                del func
                del x0

    # DYNAMICS ANALYSIS FOR FINAL RUN
    # ===============================
    if no_comp == 2:  # Only analyze 2D systems
        
        print("Running dynamics analysis for 2D system...")
        # Pass the session if NPODE was used
        npode_sess = final_npode_sess if npode_run and 'final_npode_sess' in locals() else None
        
        analyze_final_models_with_error_fields(
            magix_model=final_magix_model, 
            npode_model=final_npode_model, 
            npode_sess=npode_sess,  # Add this parameter
            nrode_func=final_nrode_func, 
            result_dir=result_dir, 
            run_id=no_run-1, 
            system_name=example, 
            train_points=yobs_full,
            include_error_fields=include_error_fields
        )

        # print("Running dynamics analysis for 2D system...")
        # analyze_final_models(magix_model=final_magix_model, npode_model=final_npode_model, nrode_func=final_nrode_func, result_dir=result_dir, run_id=no_run-1, system_name=example, train_points=yobs_full)
    else:
        print(f"Skipping dynamics analysis for {no_comp}D system (only 2D supported)")

    # Clean up final npode session if it exists
    if npode_run and 'final_npode_sess' in locals():
        final_npode_sess.close()
        tf.get_default_graph().finalize()
        tf.reset_default_graph()

if __name__ == "__main__":
    main()
