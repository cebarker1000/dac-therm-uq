"""
This module provides a set of utility functions for analyzing and visualizing
the results of the heat flow simulations. It includes functions for:

- Loading and processing simulation and experimental data.
- Plotting temperature curves and comparing them with experimental data.
- Calculating performance metrics, such as RMSE and residual variance.
- Plotting residuals to visualize the error between the simulation and experiment.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import yaml


def _compute_baseline(times: np.ndarray, temps: np.ndarray, cfg: dict):
    """Compute baseline according to cfg['baseline'] settings."""
    baseline_cfg = cfg.get('baseline', {})
    if not baseline_cfg.get('use_average', False):
        return float(temps[0])

    t_window = float(baseline_cfg.get('time_window', 0.0))
    mask = times <= t_window
    if mask.any():
        return float(np.mean(temps[mask]))
    return float(temps[0])


def load_and_process_data(cfg, output_folder):
    """
    Load simulation and experimental data, and normalize it.
    """
    # Load simulation watcher data
    watcher_csv_path = os.path.join(output_folder, 'watcher_points.csv')
    if not os.path.exists(watcher_csv_path):
        raise FileNotFoundError(f"Watcher data file not found at {watcher_csv_path}")
    
    df_sim = pd.read_csv(watcher_csv_path)
    
    # Load experimental data
    processed_data_path = os.path.join(output_folder, 'processed_experimental_data.csv')
    if os.path.exists(processed_data_path):
        df_exp = pd.read_csv(processed_data_path)
    else:
        exp_file = cfg['heating']['file']
        if not os.path.exists(exp_file):
            raise FileNotFoundError(f"Experimental data file not found at {exp_file}")
        df_exp = pd.read_csv(exp_file)

    # Get watcher point names from config
    watcher_points = cfg['output']['watcher_points']['points']
    sim_columns = list(watcher_points.keys())
    
    if len(sim_columns) < 2:
        raise ValueError("Need at least 2 watcher points for comparison plot")
        
    pside_col, oside_col = sim_columns[0], sim_columns[1]

    # Normalize simulation data
    times_sim = df_sim['time'].values
    pside_baseline_sim = _compute_baseline(times_sim, df_sim[pside_col].values, cfg)
    oside_baseline_sim = _compute_baseline(times_sim, df_sim[oside_col].values, cfg)
    pside_excursion = (df_sim[pside_col] - pside_baseline_sim).max() - (df_sim[pside_col] - pside_baseline_sim).min()
    
    if pside_excursion == 0:
        raise ValueError("P-side excursion is zero; normalization cannot be performed.")

    sim_pside_normed = (df_sim[pside_col] - pside_baseline_sim) / pside_excursion
    sim_oside_normed = (df_sim[oside_col] - oside_baseline_sim) / pside_excursion

    # Normalize experimental data
    times_exp = df_exp['time'].values
    smoothing_enabled = cfg.get('heating', {}).get('smoothing', {}).get('enabled', False)

    if smoothing_enabled and 'temp_raw' in df_exp.columns:
        pside_baseline_exp = _compute_baseline(times_exp, df_exp['temp'].values, cfg)
        exp_excursion = (df_exp['temp'] - pside_baseline_exp).max() - (df_exp['temp'] - pside_baseline_exp).min()
        exp_pside_normed = (df_exp['temp'] - pside_baseline_exp) / exp_excursion
        exp_pside_raw_normed = (df_exp['temp_raw'] - pside_baseline_exp) / exp_excursion
    else:
        pside_baseline_exp = _compute_baseline(times_exp, df_exp['temp'].values, cfg)
        exp_excursion = (df_exp['temp'] - pside_baseline_exp).max() - (df_exp['temp'] - pside_baseline_exp).min()
        exp_pside_normed = (df_exp['temp'] - pside_baseline_exp) / exp_excursion
        exp_pside_raw_normed = None

    if 'oside' in df_exp.columns:
        oside_baseline_exp = _compute_baseline(times_exp, df_exp['oside'].values, cfg)
        exp_oside_normed = (df_exp['oside'] - oside_baseline_exp) / exp_excursion
    else:
        exp_oside_normed = exp_pside_normed.copy()

    return {
        "sim_time": df_sim['time'],
        "sim_pside_normed": sim_pside_normed,
        "sim_oside_normed": sim_oside_normed,
        "exp_time": df_exp['time'],
        "exp_pside_normed": exp_pside_normed,
        "exp_oside_normed": exp_oside_normed,
        "exp_pside_raw_normed": exp_pside_raw_normed,
    }


def plot_temperature_curves(sim_time, sim_pside, sim_oside, exp_pside, exp_oside, 
                          exp_time=None, save_path=None, show_plot=True, exp_pside_raw=None):
    """
    Plot temperature curves comparing simulation and experimental data.
    """
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot simulation curves
    plt.plot(sim_time, sim_pside, 'b-', linewidth=2, label='Sim P-side')
    plt.plot(sim_time, sim_oside, 'r-', linewidth=2, label='Sim O-side')
    
    # Plot experimental points
    if exp_pside_raw is not None:
        plt.scatter(exp_time, exp_pside, color='blue', marker='o', s=40, label='Exp P-side (smoothed)')
        plt.scatter(exp_time, exp_pside_raw, color='lightblue', marker='x', s=30, alpha=0.7, label='Exp P-side (raw)')
    else:
        plt.scatter(exp_time, exp_pside, color='blue', marker='o', s=40, label='Exp P-side')
        
    plt.scatter(exp_time, exp_oside, color='red', marker='o', s=40, label='Exp O-side')
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Normalized Temperature', fontsize=12)
    plt.title('Temperature: Simulation vs Experiment', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Temperature curves plot saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


def calculate_rmse(exp_time, exp_data, sim_time, sim_data):
    """
    Calculate RMSE between experimental and simulation data at experimental time points.
    
    Parameters:
    -----------
    exp_time : pd.Series or array-like
        Experimental time points
    exp_data : pd.Series or array-like
        Experimental data values
    sim_time : pd.Series or array-like
        Simulation time points
    sim_data : pd.Series or array-like
        Simulation data values
    
    Returns:
    --------
    float
        RMSE value between the two datasets
    """
    
    # Interpolate simulation data at experimental time points
    sim_data_at_exp_times = np.interp(exp_time, sim_time, sim_data)
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((sim_data_at_exp_times - exp_data)**2))
    
    return rmse


def calculate_residual_variance(exp_time, exp_data, sim_time, sim_data):
    """
    Calculate the variance of residuals between experimental and simulation data.
    This can be interpreted as sensor noise variance.
    
    Parameters:
    -----------
    exp_time : pd.Series or array-like
        Experimental time points
    exp_data : pd.Series or array-like
        Experimental data values
    sim_time : pd.Series or array-like
        Simulation time points
    sim_data : pd.Series or array-like
        Simulation data values
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'variance': float, variance of residuals
        - 'std': float, standard deviation of residuals (sqrt of variance)
        - 'residuals': np.ndarray, the actual residual values
        - 'sim_interpolated': np.ndarray, simulation data interpolated to exp time points
    """
    
    # Interpolate simulation data at experimental time points
    sim_data_at_exp_times = np.interp(exp_time, sim_time, sim_data)
    
    # Calculate residuals
    residuals = exp_data - sim_data_at_exp_times
    
    # Calculate variance and standard deviation
    variance = np.var(residuals)
    std = np.sqrt(variance)
    
    return {
        'variance': variance,
        'std': std,
        'residuals': residuals,
        'sim_interpolated': sim_data_at_exp_times
    }


def plot_residuals(exp_time, exp_data, sim_time, sim_data, save_path=None, show_plot=True):
    """
    Create a residual plot comparing experimental and simulation data.
    This helps visualize sensor noise patterns.
    
    Parameters:
    -----------
    exp_time : pd.Series or array-like
        Experimental time points
    exp_data : pd.Series or array-like
        Experimental data values
    sim_time : pd.Series or array-like
        Simulation time points
    sim_data : pd.Series or array-like
        Simulation data values
    save_path : str, optional
        Path to save the plot. If None, plot is not saved
    show_plot : bool, default True
        Whether to display the plot
    """
    
    # Calculate residual statistics
    residual_stats = calculate_residual_variance(exp_time, exp_data, sim_time, sim_data)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Data comparison
    ax1.plot(exp_time, exp_data, 'bo-', label='Experimental', markersize=4, alpha=0.7)
    ax1.plot(exp_time, residual_stats['sim_interpolated'], 'r-', label='Simulation (interpolated)', linewidth=2)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Normalized Temperature')
    ax1.set_title('Experimental vs Simulation Data')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    ax2.plot(exp_time, residual_stats['residuals'], 'ko-', markersize=4, alpha=0.7)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero line')
    ax2.axhline(y=residual_stats['std'], color='g', linestyle=':', alpha=0.7, label=f'+1σ ({residual_stats["std"]:.4f})')
    ax2.axhline(y=-residual_stats['std'], color='g', linestyle=':', alpha=0.7, label=f'-1σ ({residual_stats["std"]:.4f})')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Residuals (Exp - Sim)')
    ax2.set_title(f'Residuals (Variance: {residual_stats["variance"]:.6f}, Std: {residual_stats["std"]:.4f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Residual plot saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return residual_stats
