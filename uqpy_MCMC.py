import yaml
import os
import sys
import argparse
from analysis.config_utils import create_uqpy_distributions, get_param_defs_from_config
from UQpy.distributions.collection.JointIndependent import JointIndependent
from UQpy.sampling.mcmc.Stretch import Stretch
from UQpy.inference.inference_models.LogLikelihoodModel import LogLikelihoodModel
from UQpy.inference.BayesParameterEstimation import BayesParameterEstimation
from scipy.special import logsumexp
from scipy.optimize import minimize
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ------------------------------------------------------------
# Baseline utility
# ------------------------------------------------------------


def _compute_baseline(times: np.ndarray, temps: np.ndarray, *, cfg_path: str | None = None):
    """Return baseline temperature according to *baseline* section in *cfg_path*.

    If *cfg_path* is None or the *baseline* section is missing, fall back to the
    first data-point (legacy behaviour).
    """
    if cfg_path is None or not os.path.exists(cfg_path):
        return float(temps[0])

    with open(cfg_path, "r") as f:
        sim_cfg = yaml.safe_load(f)

    baseline_cfg = sim_cfg.get("baseline", {})
    if not baseline_cfg.get("use_average", False):
        return float(temps[0])

    t_window = float(baseline_cfg.get("time_window", 0.0))
    mask = times <= t_window
    if mask.any():
        return float(np.mean(temps[mask]))
    return float(temps[0])

# grab surrogate model
from train_surrogate_models import FullSurrogateModel

logging.getLogger("UQpy").setLevel(logging.DEBUG)

def load_experimental_data(data_path, cfg_path, use_raw_for_normalization=False):
    """Load experimental data and normalise using the same baseline rules as the
    simulation.

    Normalisation:
        (oside - baseline_oside) / excursion_pside
    where *pside* is the column ``temp`` (or ``temp_raw`` if use_raw_for_normalization=True).

    Parameters:
    -----------
    data_path : str
        Path to experimental data CSV file
    cfg_path : str
        Path to simulation config file (for baseline settings)
    use_raw_for_normalization : bool, optional
        If True, use temp_raw column for normalization (if available). Otherwise use temp.

    Returns:
    --------
    y_obs : np.ndarray
        Normalized oside data
    times : np.ndarray
        Time values
    """
    data = pd.read_csv(data_path)
    oside_data = data["oside"].values
    times = data["time"].values

    # Determine which pside data to use for normalization
    if use_raw_for_normalization and "temp_raw" in data.columns:
        temp_data = data["temp_raw"].values  # Use raw (unsmoothed) pside data
        print("Using temp_raw column for normalization")
    else:
        temp_data = data["temp"].values  # Use regular temp column
        if use_raw_for_normalization:
            print("Warning: temp_raw column not found, using temp column instead")

    baseline_pside = _compute_baseline(times, temp_data, cfg_path=cfg_path)
    baseline_oside = _compute_baseline(times, oside_data, cfg_path=cfg_path)

    excursion_pside = (temp_data - baseline_pside).max() - (temp_data - baseline_pside).min()
    if excursion_pside <= 0.0:
        raise ValueError("Temp excursion is zero – check experimental data")

    y_obs = (oside_data - baseline_oside) / excursion_pside

    # Diagnostics
    print("Experimental data normalisation (new baseline logic):")
    print(f"  Baseline p-side (temp): {baseline_pside:.3f} K")
    print(f"  Baseline o-side:        {baseline_oside:.3f} K")
    print(f"  Excursion p-side:       {excursion_pside:.3f} K")
    print(f"  y_obs range:            {y_obs.min():.4f} – {y_obs.max():.4f}")

    return y_obs, times

def interpolate_surrogate_to_exp_grid(surrogate_pred, surrogate_time_grid, exp_time):
    """Interpolate surrogate predictions onto experimental time grid.
    
    Parameters:
    -----------
    surrogate_pred : np.ndarray, shape (n, T_sim)
        Surrogate predictions on simulation time grid
    surrogate_time_grid : np.ndarray, shape (T_sim,)
        Simulation time grid
    exp_time : np.ndarray, shape (T_exp,)
        Experimental time grid
        
    Returns:
    --------
    pred_on_exp_grid : np.ndarray, shape (n, T_overlap)
        Predictions interpolated onto experimental grid, restricted to overlapping region
    exp_time_overlap : np.ndarray, shape (T_overlap,)
        Experimental time points in overlapping region
    overlap_mask : np.ndarray, shape (T_exp,)
        Boolean mask indicating which experimental points are in overlapping region
    """
    from scipy.interpolate import interp1d
    
    # Ensure inputs are numpy arrays
    surrogate_pred = np.asarray(surrogate_pred)
    surrogate_time_grid = np.asarray(surrogate_time_grid).flatten()
    exp_time = np.asarray(exp_time).flatten()
    
    # Validate input shapes
    if surrogate_pred.ndim == 1:
        surrogate_pred = surrogate_pred.reshape(1, -1)
    if surrogate_pred.shape[1] != len(surrogate_time_grid):
        raise ValueError(
            f"Shape mismatch: surrogate_pred has {surrogate_pred.shape[1]} time points, "
            f"surrogate_time_grid has {len(surrogate_time_grid)} points"
        )
    
    # Find overlapping time region
    t_min = max(surrogate_time_grid.min(), exp_time.min())
    t_max = min(surrogate_time_grid.max(), exp_time.max())
    
    # Create mask for experimental points in overlapping region
    # Use inclusive boundaries: >= and <= to include boundary points
    overlap_mask = (exp_time >= t_min) & (exp_time <= t_max)
    exp_time_overlap = exp_time[overlap_mask]
    
    if len(exp_time_overlap) == 0:
        raise ValueError(
            f"No overlapping time region between surrogate and experimental grids. "
            f"Surrogate range: [{surrogate_time_grid.min():.6e}, {surrogate_time_grid.max():.6e}], "
            f"Experimental range: [{exp_time.min():.6e}, {exp_time.max():.6e}]"
        )
    
    # Interpolate surrogate predictions onto experimental grid (overlapping region only)
    surrogate_pred = np.atleast_2d(surrogate_pred)  # (n, T_sim)
    n_samples = surrogate_pred.shape[0]
    
    pred_on_exp_grid = np.zeros((n_samples, len(exp_time_overlap)))
    
    for i in range(n_samples):
        interp_func = interp1d(surrogate_time_grid, surrogate_pred[i], kind='linear',
                               bounds_error=False, fill_value=np.nan)
        pred_on_exp_grid[i] = interp_func(exp_time_overlap)
    
    return pred_on_exp_grid, exp_time_overlap, overlap_mask

SENSOR_VARIANCE = 1.0e-3  # You can adjust this value based on your sensor characteristics
INCLUDE_SURROGATE_UNCERT = True  # Set to False to use only sensor variance

def estimate_variance_from_raw_data(data_path, cfg_path):
    """Estimate sensor variance from normalized raw (unsmoothed) experimental data.
    
    This computes the variance of the normalized oside data in the baseline region only,
    where the signal should be constant, so any variation is pure sensor noise.
    
    Parameters:
    -----------
    data_path : str
        Path to experimental data CSV file
    cfg_path : str
        Path to simulation config file (for baseline settings)
    
    Returns:
    --------
    float
        Estimated variance in normalized units
    """
    # Load normalized raw data
    y_obs_raw, times = load_experimental_data(data_path, cfg_path, use_raw_for_normalization=True)
    
    # Get baseline time window from config
    with open(cfg_path, "r") as f:
        sim_cfg = yaml.safe_load(f)
    
    baseline_cfg = sim_cfg.get("baseline", {})
    use_avg = bool(baseline_cfg.get("use_average", False))
    
    if use_avg:
        t_window = float(baseline_cfg.get("time_window", 0.0))
        # Filter to baseline region only
        baseline_mask = times <= t_window
        if baseline_mask.any():
            y_obs_baseline = y_obs_raw[baseline_mask]
            variance = np.var(y_obs_baseline, ddof=1)  # Use ddof=1 for sample variance
            sigma = np.sqrt(variance)
            mean_baseline = np.mean(y_obs_baseline)
            
            # Diagnostic: check how many baseline points fall within ±1σ
            within_1sigma = np.sum(np.abs(y_obs_baseline - mean_baseline) <= sigma)
            within_2sigma = np.sum(np.abs(y_obs_baseline - mean_baseline) <= 2*sigma)
            pct_1sigma = 100 * within_1sigma / len(y_obs_baseline)
            pct_2sigma = 100 * within_2sigma / len(y_obs_baseline)
            
            print(f"\nVariance estimation from raw experimental data (baseline region only):")
            print(f"  Baseline time window: {t_window:.6e} s")
            print(f"  Number of baseline points: {len(y_obs_baseline)}")
            print(f"  Baseline mean: {mean_baseline:.6e}")
            print(f"  Baseline range: [{y_obs_baseline.min():.6e}, {y_obs_baseline.max():.6e}]")
            print(f"  Baseline span: {y_obs_baseline.max() - y_obs_baseline.min():.6e}")
            print(f"  Estimated variance (normalized units): {variance:.6e}")
            print(f"  Standard deviation (normalized units): {sigma:.6e}")
            print(f"  Points within ±1σ: {within_1sigma}/{len(y_obs_baseline)} ({pct_1sigma:.1f}%)")
            print(f"  Points within ±2σ: {within_2sigma}/{len(y_obs_baseline)} ({pct_2sigma:.1f}%)")
            print(f"  Expected for normal distribution: ~68% within ±1σ, ~95% within ±2σ")
            
            # Warn if coverage is too low
            if pct_1sigma < 50:
                print(f"  ⚠️  WARNING: Only {pct_1sigma:.1f}% of baseline points within ±1σ")
                print(f"     This suggests variance may be underestimated or distribution is non-normal")
            
            return variance
        else:
            print("Warning: No points in baseline time window, using first point only")
            variance = 0.0  # Can't estimate from single point
    else:
        # If not using average, baseline is just first point - can't estimate variance
        print("Warning: Baseline uses first point only, cannot estimate variance from baseline")
        print("  Using variance of entire time series (may overestimate)")
        variance = np.var(y_obs_raw, ddof=1)
    
    print(f"\nVariance estimation from raw experimental data:")
    print(f"  Estimated variance (normalized units): {variance:.6e}")
    print(f"  Standard deviation (normalized units): {np.sqrt(variance):.6e}")
    
    return variance

def _gaussian_loglike(y_pred: np.ndarray, y_obs: np.ndarray, *, sigma2) -> np.ndarray:
    """Vectorised Gaussian log-likelihood."""
    resid = y_pred - y_obs                          # (m, T)
    sigma2 = np.asarray(sigma2)                     # ensure array for broadcasting
    return -0.5 * np.sum(resid ** 2 / sigma2 + np.log(2 * np.pi * sigma2), axis=1)

CALLS = 0

def log_likelihood_full(params=None, data=None, surrogate=None, param_names=None, sensor_variance=None, exp_time=None, surrogate_time_grid=None):
    """Log likelihood function for full parameter MCMC with conditional error inflation.
    
    Now interpolates surrogate predictions onto experimental time grid and restricts
    to overlapping region.
    """
    global CALLS
    params = np.atleast_2d(params)      # (n, n_params)
    n, _ = params.shape
    log_L = np.empty(n)
    
    # Use provided sensor_variance or fall back to global SENSOR_VARIANCE
    if sensor_variance is None:
        sensor_variance = SENSOR_VARIANCE
    
    # Check if error_inflation parameter is present
    has_error_inflation = param_names is not None and 'error_inflation' in param_names
    error_inf_idx = param_names.index('error_inflation') if has_error_inflation else None
    
    # Extract experimental data and time grid
    y_obs = data['y_obs']  # Normalized experimental data at original time points
    exp_time_vals = data['exp_time']  # Experimental time grid
    
    # Validate that y_obs and exp_time_vals have the same length
    if len(y_obs) != len(exp_time_vals):
        raise ValueError(f"Length mismatch: y_obs has {len(y_obs)} points, exp_time has {len(exp_time_vals)} points")
    
    # Ensure arrays are 1D
    y_obs = np.asarray(y_obs).flatten()
    exp_time_vals = np.asarray(exp_time_vals).flatten()
    
    for i in range(n):
        # Extract parameters for surrogate (exclude error_inflation if present)
        if has_error_inflation:
            # Remove error_inflation parameter for surrogate call
            surrogate_params = np.delete(params[i:i+1], error_inf_idx, axis=1)
        else:
            surrogate_params = params[i:i+1]
        
        # Generate predictions and predictive uncertainty using the surrogate model
        y_pred_sim, _, _, curve_uncert_sim = surrogate.predict_temperature_curves(surrogate_params)  # Shapes (1, T_sim)

        # Interpolate predictions onto experimental time grid (restricted to overlapping region)
        y_pred_exp, exp_time_overlap, overlap_mask = interpolate_surrogate_to_exp_grid(
            y_pred_sim, surrogate_time_grid, exp_time_vals
        )
        y_pred_exp = y_pred_exp[0]  # Extract single prediction (n=1)
        
        # Interpolate uncertainty onto experimental grid if needed
        if INCLUDE_SURROGATE_UNCERT:
            curve_uncert_exp, _, _ = interpolate_surrogate_to_exp_grid(
                curve_uncert_sim, surrogate_time_grid, exp_time_vals
            )
            curve_uncert_exp = curve_uncert_exp[0]
        else:
            curve_uncert_exp = np.zeros_like(y_pred_exp)
        
        # Extract experimental data in overlapping region
        y_obs_overlap = y_obs[overlap_mask]
        
        # Validate that arrays have matching lengths after overlap filtering
        if len(y_pred_exp) != len(y_obs_overlap):
            raise ValueError(
                f"Length mismatch after overlap filtering: "
                f"y_pred_exp has {len(y_pred_exp)} points, y_obs_overlap has {len(y_obs_overlap)} points. "
                f"Overlap mask selected {overlap_mask.sum()} points from {len(exp_time_vals)} total."
            )
        if len(y_pred_exp) != len(exp_time_overlap):
            raise ValueError(
                f"Length mismatch: y_pred_exp has {len(y_pred_exp)} points, "
                f"exp_time_overlap has {len(exp_time_overlap)} points"
            )
        
        # Check for NaN values (could indicate interpolation issues)
        if np.any(np.isnan(y_pred_exp)):
            n_nan = np.isnan(y_pred_exp).sum()
            raise ValueError(
                f"Found {n_nan} NaN values in interpolated predictions. "
                f"This may indicate interpolation outside valid range or numerical issues."
            )
        if np.any(np.isnan(y_obs_overlap)):
            n_nan = np.isnan(y_obs_overlap).sum()
            raise ValueError(
                f"Found {n_nan} NaN values in experimental data. "
                f"Check experimental data for missing values."
            )

        # Get error inflation factor if parameter is present
        error_inflation = params[i, error_inf_idx] if has_error_inflation else 1.0
        
        if INCLUDE_SURROGATE_UNCERT:
            sigma2 = error_inflation * (sensor_variance + curve_uncert_exp**2)  # (T_overlap,)
        else:
            sigma2 = error_inflation * sensor_variance  # scalar

        ll = _gaussian_loglike(y_pred_exp.reshape(1, -1), y_obs_overlap.reshape(1, -1), sigma2=sigma2)
        log_L[i] = ll[0]  # Extract scalar value
            
    CALLS += params.shape[0]
    if CALLS % 10000 == 0:  # More frequent progress reporting
        print(f"{CALLS:,} proposals evaluated")
    return log_L

def get_parameter_bounds(param_defs):
    """Extract parameter bounds from parameter definitions for optimization.
    
    Parameters
    ----------
    param_defs : list of dict
        Parameter definitions from config file
        
    Returns
    -------
    bounds : list of tuples
        List of (low, high) bounds for each parameter
    """
    bounds = []
    for param_def in param_defs:
        if param_def['type'] == 'uniform':
            bounds.append((param_def['low'], param_def['high']))
        elif param_def['type'] == 'lognormal':
            # Use wide bounds around center (±5 sigma_log)
            center = param_def['center']
            sigma_log = param_def['sigma_log']
            low = center * np.exp(-5 * sigma_log)
            high = center * np.exp(5 * sigma_log)
            bounds.append((low, high))
        elif param_def['type'] == 'normal':
            # Use ±5 sigma bounds
            center = param_def['center']
            sigma = param_def['sigma']
            bounds.append((center - 5 * sigma, center + 5 * sigma))
        else:
            raise ValueError(f"Unknown parameter type: {param_def['type']}")
    return bounds

def get_parameter_initial_guess(param_defs):
    """Get initial guess for parameters (prior mean/center).
    
    Parameters
    ----------
    param_defs : list of dict
        Parameter definitions from config file
        
    Returns
    -------
    x0 : np.ndarray
        Initial guess vector
    """
    x0 = []
    for param_def in param_defs:
        if 'center' in param_def:
            x0.append(param_def['center'])
        elif param_def['type'] == 'uniform':
            # Use midpoint
            x0.append((param_def['low'] + param_def['high']) / 2.0)
        else:
            raise ValueError(f"Cannot determine initial guess for parameter type: {param_def['type']}")
    return np.array(x0)

def find_least_squares_start(surrogate, y_obs, exp_time, param_defs, param_names,
                              sensor_variance, surrogate_time_grid, n_starts=10, 
                              use_ls_init=True):
    """Find best-fit parameters using least squares optimization.
    
    This function minimizes the negative log-likelihood to find a good starting
    point for MCMC chains. It tries multiple random starting points to avoid
    local minima.
    
    Parameters
    ----------
    surrogate : FullSurrogateModel
        Trained surrogate model
    y_obs : np.ndarray
        Normalized experimental data
    exp_time : np.ndarray
        Experimental time grid
    param_defs : list of dict
        Parameter definitions
    param_names : list of str
        Parameter names
    sensor_variance : float
        Sensor variance for likelihood calculation
    surrogate_time_grid : np.ndarray
        Surrogate time grid
    n_starts : int
        Number of random starting points to try
    use_ls_init : bool
        If False, return None (use prior initialization)
        
    Returns
    -------
    best_params : np.ndarray or None
        Best-fit parameter vector, or None if use_ls_init=False
    """
    if not use_ls_init:
        return None
        
    print("\n" + "=" * 60)
    print("FINDING LEAST-SQUARES STARTING POINT")
    print("=" * 60)
    
    # Get bounds and initial guess
    bounds = get_parameter_bounds(param_defs)
    x0_base = get_parameter_initial_guess(param_defs)
    
    # Objective function (negative log-likelihood for minimization)
    def objective(params):
        params_2d = params.reshape(1, -1)
        try:
            ll = log_likelihood_full(
                params=params_2d,
                data={'y_obs': y_obs, 'exp_time': exp_time},
                surrogate=surrogate,
                param_names=param_names,
                sensor_variance=sensor_variance,
                surrogate_time_grid=surrogate_time_grid
            )
            return -ll[0]  # Negate for minimization
        except (ValueError, RuntimeError) as e:
            # Return large value for invalid parameters
            return 1e10
    
    # Try multiple starting points
    best_result = None
    best_value = np.inf
    successful_starts = 0
    
    rng = np.random.default_rng(42)  # Reproducible randomness
    
    print(f"Trying {n_starts} random starting points...")
    for i in range(n_starts):
        # Start from prior mean/center with small random perturbation
        if i == 0:
            # First try: use exact prior center for non-uniform parameters,
            # draw randomly from range for uniform parameters
            x0 = x0_base.copy()
            for j, param_def in enumerate(param_defs):
                if param_def['type'] == 'uniform':
                    # Draw randomly from the uniform parameter range
                    x0[j] = rng.uniform(param_def['low'], param_def['high'])
        else:
            # Subsequent tries: add random perturbation
            # For lognormal parameters, perturb in log space
            x0 = x0_base.copy()
            for j, param_def in enumerate(param_defs):
                if param_def['type'] == 'lognormal':
                    # Perturb in log space
                    log_x0 = np.log(x0[j])
                    sigma_log = param_def['sigma_log']
                    log_x0 += rng.normal(0, 0.5 * sigma_log)
                    x0[j] = np.exp(log_x0)
                elif param_def['type'] == 'normal':
                    sigma = param_def['sigma']
                    x0[j] += rng.normal(0, 0.5 * sigma)
                elif param_def['type'] == 'uniform':
                    # Draw randomly from the uniform parameter range
                    x0[j] = rng.uniform(param_def['low'], param_def['high'])
        
        # Ensure x0 is within bounds
        for j, (low, high) in enumerate(bounds):
            x0[j] = np.clip(x0[j], low, high)
        
        # Optimize
        try:
            res = minimize(objective, x0=x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 1000, 'ftol': 1e-6})
            if res.success and res.fun < best_value:
                best_value = res.fun
                best_result = res.x.copy()
                successful_starts += 1
        except Exception as e:
            # Skip failed optimizations
            continue
    
    if best_result is None:
        print("⚠️  WARNING: All optimization attempts failed!")
        print("   Falling back to prior initialization.")
        return None
    
    print(f"\n✓ Found best-fit parameters ({successful_starts}/{n_starts} successful optimizations)")
    print(f"  Best negative log-likelihood: {best_value:.2f}")
    print(f"\n  Best-fit parameters:")
    print(f"  {'Parameter':<25s} {'Value':<20s} {'Units/Notes'}")
    print(f"  {'-'*25} {'-'*20} {'-'*30}")
    
    # Get parameter info for better display
    for i, (name, param_def) in enumerate(zip(param_names, param_defs)):
        value = best_result[i]
        units = param_def.get('units', '')
        if not units:
            units = param_def.get('description', '')
        print(f"  {name:<25s} {value:<20.6e} {units}")
    
    # Also compute and display the actual log-likelihood (positive)
    params_2d = best_result.reshape(1, -1)
    actual_ll = log_likelihood_full(
        params=params_2d,
        data={'y_obs': y_obs, 'exp_time': exp_time},
        surrogate=surrogate,
        param_names=param_names,
        sensor_variance=sensor_variance,
        surrogate_time_grid=surrogate_time_grid
    )[0]
    print(f"\n  Log-likelihood at best fit: {actual_ll:.2f}")
    
    return best_result

def main():
    parser = argparse.ArgumentParser(description="Run MCMC analysis for heat flow model.")
    parser.add_argument('--config_path', type=str, required=True, help="Path to the distributions YAML file.")
    parser.add_argument('--surrogate_path', type=str, required=True, help="Path to the trained surrogate model file.")
    parser.add_argument('--exp_data_path', type=str, required=True, help="Path to the experimental data CSV file.")
    parser.add_argument('--sim_cfg', type=str, required=True, help="Path to the simulation YAML config (for baseline settings).")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the MCMC results NPZ file.")
    parser.add_argument('--plot_path_prefix', type=str, required=True, help="Prefix for output plot filenames.")
    parser.add_argument('--n_walkers', type=int, default=60, help="Number of MCMC walkers.")
    parser.add_argument('--n_samples', type=int, default=1000000, help="Number of MCMC samples to generate.")
    parser.add_argument('--burn_length', type=int, default=20000, help="Number of burn-in samples.")
    parser.add_argument('--estimate-variance-from-raw', action='store_true',
                        help="Estimate sensor variance from normalized raw (unsmoothed) experimental data.")
    parser.add_argument('--no-ls-init', action='store_true',
                        help="Disable least-squares initialization (use prior instead).")
    parser.add_argument('--ls-n-starts', type=int, default=10,
                        help="Number of random starting points for least-squares optimization (default: 10).")
    parser.add_argument('--ls-init-spread', type=float, default=0.05,
                        help="Spread around LS fit for walker initialization (default: 0.05 = 5%%).")
    
    args = parser.parse_args()
    
    # Set use_ls_init based on flag (default: True)
    args.use_ls_init = not args.no_ls_init

    import time
    start_time = time.time()
    
    # Load components based on command-line arguments
    param_defs = get_param_defs_from_config(config_path=args.config_path)
    uqpy_dists = create_uqpy_distributions(param_defs)
    full_prior = JointIndependent(marginals=uqpy_dists)
    
    # Calculate parameters info
    n_params = len(param_defs)
    param_names = [param_def['name'] for param_def in param_defs]
    
    surrogate = FullSurrogateModel.load_model(args.surrogate_path)
    y_obs, exp_time = load_experimental_data(args.exp_data_path, args.sim_cfg)
    
    # Get surrogate time grid
    surrogate_time_grid = surrogate.time_grid
    
    # Estimate variance from raw data if requested
    sensor_variance = SENSOR_VARIANCE
    if args.estimate_variance_from_raw:
        sensor_variance = estimate_variance_from_raw_data(args.exp_data_path, args.sim_cfg)
    
    # Check overlapping region
    t_min = max(surrogate_time_grid.min(), exp_time.min())
    t_max = min(surrogate_time_grid.max(), exp_time.max())
    overlap_mask = (exp_time >= t_min) & (exp_time <= t_max)
    n_overlap = overlap_mask.sum()
    
    print("\n" + "=" * 60)
    print("MCMC SIMULATION SETUP")
    print("=" * 60)
    print(f"Config: {args.config_path}")
    print(f"Surrogate: {args.surrogate_path}")
    print(f"Experimental Data: {args.exp_data_path}")
    print(f"Output: {args.output_path}")
    print(f"Including surrogate uncertainty: {INCLUDE_SURROGATE_UNCERT}")
    print(f"Sensor variance: {sensor_variance:.6e} {'(estimated from raw data)' if args.estimate_variance_from_raw else '(fixed)'}")
    print(f"Total parameters: {n_params}")
    print(f"Parameter names: {param_names}")
    print(f"\nTime grid alignment:")
    print(f"  Surrogate time range: [{surrogate_time_grid.min():.6e}, {surrogate_time_grid.max():.6e}] s ({len(surrogate_time_grid)} points)")
    print(f"  Experimental time range: [{exp_time.min():.6e}, {exp_time.max():.6e}] s ({len(exp_time)} points)")
    print(f"  Overlapping region: [{t_min:.6e}, {t_max:.6e}] s ({n_overlap} points)")
    
    # Check if error inflation is included
    has_error_inflation = 'error_inflation' in param_names
    if has_error_inflation:
        print(f"✓ Error inflation parameter included")
        print(f"  Surrogate expects {len(param_names) - 1} parameters (excluding error_inflation)")
        print(f"  MCMC will sample {len(param_names)} parameters (including error_inflation)")
    else:
        print(f"✗ Error inflation parameter not included (using fixed error model)")
        print(f"  Surrogate expects {len(param_names)} parameters")
        print(f"  MCMC will sample {len(param_names)} parameters")
    
    print("\n" + "=" * 60)
    print("STARTING MCMC SIMULATION")
    print("=" * 60)
    
    # UQpy expects data to be a list or numpy array, not a dict
    # We'll pass y_obs as the data and make exp_time available via closure
    log_likelihood_with_surrogate = lambda params, data: log_likelihood_full(
        params=params, 
        data={'y_obs': data, 'exp_time': exp_time},  # data is y_obs from UQpy, we add exp_time
        surrogate=surrogate, 
        param_names=param_names, 
        sensor_variance=sensor_variance,
        surrogate_time_grid=surrogate_time_grid
    )
    ll_model = LogLikelihoodModel(n_parameters=n_params, log_likelihood=log_likelihood_with_surrogate)
    ll_model.prior = full_prior
    
    # Set up sampler
    # Find least-squares starting point if requested
    ls_start = find_least_squares_start(
        surrogate=surrogate,
        y_obs=y_obs,
        exp_time=exp_time,
        param_defs=param_defs,
        param_names=param_names,
        sensor_variance=sensor_variance,
        surrogate_time_grid=surrogate_time_grid,
        n_starts=args.ls_n_starts,
        use_ls_init=args.use_ls_init
    )
    
    if ls_start is not None:
        # Ask user for confirmation before proceeding
        print("\n" + "=" * 60)
        print("LEAST-SQUARES INITIALIZATION CONFIRMATION")
        print("=" * 60)
        print("The least-squares fit has been found. Review the results above.")
        print("\nOptions:")
        print("  [y] Yes - Proceed with MCMC using LS initialization")
        print("  [n] No  - Use prior initialization instead")
        print("  [q] Quit - Exit without running MCMC")
        
        while True:
            try:
                response = input("\nProceed with MCMC using LS initialization? [y/n/q]: ").strip().lower()
                if response in ['y', 'yes']:
                    use_ls = True
                    break
                elif response in ['n', 'no']:
                    use_ls = False
                    print("\nSwitching to prior initialization...")
                    break
                elif response in ['q', 'quit', 'exit']:
                    print("\nExiting without running MCMC.")
                    sys.exit(0)
                else:
                    print("Please enter 'y', 'n', or 'q'")
            except (EOFError, KeyboardInterrupt):
                print("\n\nInterrupted. Exiting without running MCMC.")
                sys.exit(1)
        
        if not use_ls:
            ls_start = None
    
    if ls_start is not None:
        # Initialize walkers around least-squares fit
        print(f"\nInitializing {args.n_walkers} walkers around least-squares fit...")
        print(f"  Spread: {args.ls_init_spread*100:.1f}% around best-fit parameters")
        
        rng = np.random.default_rng(42)
        initial_positions = np.zeros((args.n_walkers, n_params))
        
        for i in range(args.n_walkers):
            # Add random perturbation around LS fit
            # For lognormal parameters, perturb in log space
            perturbed = ls_start.copy()
            for j, param_def in enumerate(param_defs):
                if param_def['type'] == 'lognormal':
                    # Perturb in log space
                    log_val = np.log(perturbed[j])
                    sigma_log = param_def['sigma_log']
                    log_val += rng.normal(0, args.ls_init_spread * sigma_log)
                    perturbed[j] = np.exp(log_val)
                elif param_def['type'] == 'normal':
                    sigma = param_def['sigma']
                    perturbed[j] += rng.normal(0, args.ls_init_spread * sigma)
                elif param_def['type'] == 'uniform':
                    # Perturb by fraction of range
                    range_val = param_def['high'] - param_def['low']
                    perturbed[j] += rng.normal(0, args.ls_init_spread * range_val)
                else:
                    # Generic perturbation (5% of value)
                    perturbed[j] += rng.normal(0, args.ls_init_spread * np.abs(perturbed[j]))
            
            # Ensure within bounds
            bounds = get_parameter_bounds(param_defs)
            for j, (low, high) in enumerate(bounds):
                perturbed[j] = np.clip(perturbed[j], low, high)
            
            initial_positions[i] = perturbed
        
        # Ensure at least one walker starts exactly at the LS fit
        initial_positions[0] = ls_start.copy()
    else:
        # Fall back to prior initialization
        print(f"\nInitializing {args.n_walkers} walkers from prior distribution...")
        initial_positions = full_prior.rvs(nsamples=args.n_walkers)
    
    stretch_sampler = Stretch(
        burn_length=args.burn_length,
        jump=1,
        dimension=n_params,
        seed=initial_positions.tolist(),
        save_log_pdf=True,
        scale=2.4,
        n_chains=args.n_walkers,
        concatenate_chains=False
    )
    
    # Reset CALLS counter right before MCMC sampling starts
    # (after LS optimization is complete, so progress reporting only counts MCMC calls)
    global CALLS
    CALLS = 0
    print(f"\nResetting CALLS counter to {CALLS} (MCMC sampling starting now)")
    print(f"Progress updates will appear every 10,000 likelihood evaluations")
    
    bpe = BayesParameterEstimation(
        inference_model=ll_model,
        data=y_obs,  # Pass y_obs as array (UQpy requirement)
        sampling_class=stretch_sampler,
        nsamples=args.n_samples
    )
    
    samples_full = bpe.sampler.samples
    accepted_mask = np.isfinite(bpe.sampler.log_pdf_values)
    elapsed_time = time.time() - start_time
    
    np.savez(args.output_path, 
             samples_full=samples_full,
             accepted_mask=accepted_mask,
             log_pdf_values=bpe.sampler.log_pdf_values,
             param_names=param_names)
    
    print(f"\nMCMC complete! Total samples: {samples_full.size}")
    print(f"Acceptance rate: {bpe.sampler.acceptance_rate}")
    
    # Handle sample format (always 3D from Stretch with concatenate_chains=False)
    # Flatten to (n_samples * n_chains, n_dimensions) for statistics
    samples_flat = samples_full.reshape(-1, samples_full.shape[2])
    print(f"Sample format: 3D with shape {samples_full.shape}, flattened to {samples_flat.shape}")
    
    # Print parameter statistics
    print(f"\nParameter Statistics (mean ± σ):")
    print(f"{'Parameter':<15} {'Posterior Mean':<15} {'Posterior Std':<15}")
    print("-" * 50)
    
    for i, name in enumerate(param_names):
        post_mean = samples_flat[:, i].mean()
        post_std = samples_flat[:, i].std()
        print(f"{name:<15} {post_mean:<15.3e} {post_std:<15.3e}")
    
    # Conditional error inflation analysis
    if 'error_inflation' in param_names:
        error_inf_idx = param_names.index('error_inflation')
        error_inf_mean = samples_flat[:, error_inf_idx].mean()
        error_inf_std = samples_flat[:, error_inf_idx].std()
        print(f"\nError Inflation Analysis:")
        print(f"  Posterior mean: {error_inf_mean:.3f} ± {error_inf_std:.3f}")
        print(f"  This means the effective error variance is {error_inf_mean:.3f}x the base variance")
        if error_inf_mean > 1.0:
            print(f"  → Errors are inflated (wider posterior)")
        elif error_inf_mean < 1.0:
            print(f"  → Errors are deflated (narrower posterior)")
        else:
            print(f"  → Errors are at baseline level")
    
    # Print convergence diagnostics
    print(f"\nConvergence Diagnostics:")
    print(f"  Total samples: {samples_full.shape[0] * samples_full.shape[1]}")
    print(f"  Samples per walker: {samples_full.shape[0]}")
    print(f"  Number of walkers: {args.n_walkers}")
    print(f"  Total time: {elapsed_time:.1f} seconds")
    print(f"  Time per sample: {elapsed_time/(samples_full.shape[0] * samples_full.shape[1]):.3f} seconds")
    if hasattr(bpe.sampler, 'scale'):
        print(f"  Scale parameter: {bpe.sampler.scale}")
    
    print(f"MCMC results saved to {args.output_path}")
    
    # Create corner plot for all parameters
    try:
        import corner
        corner_plot_path = f"{args.plot_path_prefix}_corner.png"
        fig_corner = corner.corner(
            samples_flat,
            labels=param_names,
            show_titles=True,
            title_fmt=".2e",
            title_kwargs={"fontsize": 10}
        )
        fig_corner.savefig(corner_plot_path, dpi=300, bbox_inches="tight")
        print(f"Corner plot saved to {corner_plot_path}")
        plt.close(fig_corner)
    except ImportError:
        print("Could not import 'corner' library, skipping corner plot.")

    # Create trace plots for all parameters
    trace_plot_path = f"{args.plot_path_prefix}_trace.png"
    fig_trace, axes = plt.subplots(n_params, 1, figsize=(12, 2 * n_params), sharex=True)
    if n_params == 1:
        axes = [axes]
    
    for i in range(n_params):
        for j in range(args.n_walkers):
            axes[i].plot(samples_full[:, j, i], alpha=0.5)
        axes[i].set_ylabel(param_names[i])
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Sample Index')
    fig_trace.suptitle("MCMC Trace Plots", fontsize=16)
    fig_trace.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(trace_plot_path, dpi=300, bbox_inches='tight')
    print(f"Trace plots saved to {trace_plot_path}")
    plt.close(fig_trace)

if __name__ == "__main__":
    main()


