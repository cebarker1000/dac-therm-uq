#!/usr/bin/env python3
"""
Least squares fitting using UQpy for thermal conductivity estimation.
Compares experimental time series directly to surrogate-predicted curves.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import yaml
from scipy.optimize import minimize
from train_surrogate_models import FullSurrogateModel
from run_and_compare_simulation import SimulationComparer
from analysis.config_utils import get_param_defs_from_config

# Load parameter definitions from config file
param_defs = get_param_defs_from_config()
param_names = [p["name"] for p in param_defs]

def extract_params_from_config(cfg):
    """Extract parameters from YAML config file (from compare_single_parameter_set.py)"""
    params = {}
    mats = cfg.get("mats", {})
    # sample
    if "sample" in mats:
        samp = mats["sample"]
        params["d_sample"] = samp.get("z", 1.84e-6)
        params["rho_cv_sample"] = samp.get("rho_cv", 2.764828e6)
        params["k_sample"] = samp.get("k", 3.8)
    # coupler
    if "p_coupler" in mats:
        cou = mats["p_coupler"]
        params["d_coupler"] = cou.get("z", 6.2e-8)
        params["rho_cv_coupler"] = cou.get("rho_cv", 3.44552e6)
        params["k_coupler"] = cou.get("k", 350.0)
    # insulators
    if "p_ins" in mats:
        ins = mats["p_ins"]
        params["d_ins_pside"] = ins.get("z", 6.3e-6)
        params["rho_cv_ins"] = ins.get("rho_cv", 2.764828e6)
        params["k_ins"] = ins.get("k", 10.0)
    if "o_ins" in mats:
        oins = mats["o_ins"]
        params["d_ins_oside"] = oins.get("z", 3.2e-6)
        # fallback rho_cv_ins / k_ins already set above
    # heating
    if "heating" in cfg:
        params["fwhm"] = cfg["heating"].get("fwhm", 12e-6)
    return params

def least_squares_objective(params_free, exp_timeseries, config_file):
    """
    Compute negative sum of squared residuals for least squares fitting.
    This is what we minimize.
    """
    # Load surrogate model
    surrogate = FullSurrogateModel.load_model("outputs/full_surrogate_model.pkl")
    
    # Handle both 1D and 2D parameter arrays from UQpy
    params_free = np.asarray(params_free)
    if params_free.ndim == 2:
        # Multiple parameter sets - process each one
        results = []
        for params in params_free:
            # Build full parameter vector in correct order
            params_full = build_full_parameter_vector(params, config_file)
            
            # Get predicted curve from surrogate
            predicted_curve, _, _, _ = surrogate.predict_temperature_curves(params_full.reshape(1, -1))
            predicted_curve = predicted_curve[0]
            
            # Compute sum of squared residuals
            residuals = predicted_curve - exp_timeseries
            ssr = np.sum(residuals**2)
            results.append(-ssr)  # Negative for maximization
        return np.array(results)
    else:
        # Single parameter set
        # Build full parameter vector in correct order
        params_full = build_full_parameter_vector(params_free, config_file)
        
        # Get predicted curve from surrogate
        predicted_curve, _, _, _ = surrogate.predict_temperature_curves(params_full.reshape(1, -1))
        predicted_curve = predicted_curve[0]
        
        # Compute sum of squared residuals
        residuals = predicted_curve - exp_timeseries
        ssr = np.sum(residuals**2)
        return -ssr  # Negative for maximization

def build_full_parameter_vector(params_free, config_file):
    """
    Build the full 11-parameter vector in the correct order expected by the surrogate.
    Now reads fixed parameters from config file instead of hardcoding them.
    """
    # Load surrogate to get parameter names
    from train_surrogate_models import FullSurrogateModel
    surrogate = FullSurrogateModel.load_model("outputs/full_surrogate_model.pkl")
    
    # Load config file and extract parameters
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)
    params_dict = extract_params_from_config(cfg)
    
    # Update the free parameters (k_sample, k_ins, k_coupler) with the optimization values
    params_dict['k_sample'] = params_free[0]
    params_dict['k_ins'] = params_free[1]
    params_dict['k_coupler'] = params_free[2]
    
    # Build vector in the order expected by surrogate
    vector = np.array([params_dict[name] for name in surrogate.parameter_names])
    return vector

def main():
    parser = argparse.ArgumentParser(description="Least squares fitting using UQpy")
    parser.add_argument("config_file", help="YAML config file defining fixed parameter values")
    parser.add_argument("--mc", type=int, default=0, help="If >0, run Monte Carlo with this many draws of the 8 fixed parameters and report statistics on fitted k values")
    parser.add_argument("--n_starts", type=int, default=10, help="Number of random initial guesses")
    parser.add_argument("--noise_std", type=float, default=0.05, help="Std dev of Gaussian noise added to central initial guess")
    parser.add_argument("--central", nargs=3, type=float, metavar=("k_sample", "k_ins", "k_coupler"), help="Central initial guess for the three conductivities")
    args = parser.parse_args()
    
    print("Setting up least squares fitting...")
    
    # Diagnostic: Show what values are being used
    print("\n" + "="*50)
    print("PARAMETER VALUE COMPARISON")
    print("="*50)
    
    # Show Monte Carlo distribution centers
    print("Monte Carlo distribution centers (param_defs):")
    for d in param_defs[:8]:
        print(f"  {d['name']:15s}: {d['center']:.2e} (sigma_log={d['sigma_log']:.3f})")
    
    # Show config file values
    with open(args.config_file, "r") as f:
        cfg_diag = yaml.safe_load(f)
    config_params = extract_params_from_config(cfg_diag)
    print("\nConfig file values (extract_params_from_config):")
    for name in ["d_sample", "rho_cv_sample", "rho_cv_coupler", "rho_cv_ins", "d_coupler", "d_ins_oside", "d_ins_pside", "fwhm"]:
        if name in config_params:
            print(f"  {name:15s}: {config_params[name]:.2e}")
        else:
            print(f"  {name:15s}: NOT FOUND")
    
    # Check for mismatches
    print("\nMismatches:")
    for d in param_defs[:8]:
        mc_center = d['center']
        config_val = config_params.get(d['name'], 'NOT_FOUND')
        if config_val != 'NOT_FOUND' and abs(mc_center - config_val) / mc_center > 0.01:
            print(f"  {d['name']:15s}: MC={mc_center:.2e}, Config={config_val:.2e}, Diff={100*(config_val-mc_center)/mc_center:+.1f}%")
    
    print("="*50)
    
    # Load experimental data and normalise identically to SimulationComparer
    comparer = SimulationComparer(config_file=args.config_file)
    exp_time, exp_temp, exp_oside = comparer.load_experimental_data()
    aligned_oside = comparer.align_experimental_data(exp_time, exp_oside)
    pside_exc_exp = np.max(exp_temp) - np.min(exp_temp)
    exp_timeseries = (aligned_oside - aligned_oside[0]) / pside_exc_exp  # shape (50,) aligns with surrogate output
    
    print(f"Experimental time series shape: {exp_timeseries.shape}")
    print(f"Time series range: [{exp_timeseries.min():.4f}, {exp_timeseries.max():.4f}]")

    # ------------------------------------------------------------------
    # Utility: draw the 8 nuisance parameters from their metrology priors
    # ------------------------------------------------------------------
    def draw_fixed_params(rng: np.random.Generator):
        p = {}
        for d in param_defs[:8]:  # first 8 entries are the fixed / nuisance params
            name = d["name"]
            if d["type"] == "lognormal":
                mu = np.log(d["center"])
                sigma = d["sigma_log"]
                p[name] = float(rng.lognormal(mu, sigma))
            else:
                # first 8 are lognormal, no other distribution types expected
                pass
        return p

    # Helper: build full 11-parameter vector from a fixed-param dict and free ks
    surrogate = FullSurrogateModel.load_model("outputs/full_surrogate_model.pkl")
    def vector_from_dict(k_vec, fixed_dict):
        vec = []
        for name in surrogate.parameter_names:
            if name == "k_sample":
                vec.append(k_vec[0])
            elif name == "k_ins":
                vec.append(k_vec[1])
            elif name == "k_coupler":
                vec.append(k_vec[2])
            else:
                vec.append(fixed_dict[name])
        return np.asarray(vec, dtype=float)

    # ------------------------------------------------------------------
    # If --mc > 0 ➜ run Monte-Carlo propagation instead of single fit
    # ------------------------------------------------------------------
    if args.mc > 0:
        Nmc = args.mc
        rng = np.random.default_rng(42)

        # Start from YAML central values for initial guess, unless overridden
        if args.central is not None:
            central_k = np.array(args.central, dtype=float)
        else:
            with open(args.config_file, "r") as f:
                cfg_tmp = yaml.safe_load(f)
            p_tmp = extract_params_from_config(cfg_tmp)
            central_k = np.array([p_tmp["k_sample"], p_tmp["k_ins"], p_tmp["k_coupler"]], dtype=float)

        bounds = [(2.8, 4.8), (7.0, 13.0), (300, 400)]

        k_samples = []
        fixed_params_samples = []  # Store the fixed parameters used for each draw
        pc_scores_samples = []  # Store the PC scores for each draw

        iterator = trange(Nmc, desc="MC draws") if trange is not range else range(Nmc)
        for i in iterator:
            # Print progress every 10%
            if (i + 1) % max(1, Nmc // 10) == 0:
                print(f"MC progress: {i + 1}/{Nmc} ({100 * (i + 1) / Nmc:.0f}%)")
            
            fixed_draw = draw_fixed_params(rng)

            # Objective for this draw (consistent with one-shot approach)
            def obj(k):
                curve, _, _, _ = surrogate.predict_temperature_curves(vector_from_dict(k, fixed_draw).reshape(1, -1))
                resid = curve[0] - exp_timeseries
                return np.sum(resid ** 2)  # Minimize sum of squared residuals directly

            res = minimize(obj, x0=central_k, method="L-BFGS-B", bounds=bounds)
            if res.success:
                k_samples.append(res.x)
                fixed_params_samples.append(fixed_draw)  # Store the fixed parameters
                
                # Calculate PC scores for this draw
                full_vector = vector_from_dict(res.x, fixed_draw)
                curve, fpca_coeffs, _, _ = surrogate.predict_temperature_curves(full_vector.reshape(1, -1))
                pc_scores_samples.append(fpca_coeffs[0])  # Store PC scores (flatten batch dimension)
            else:
                print(f"[warn] optimisation failed on draw {i}: {res.message}")

        k_samples = np.array(k_samples)
        if k_samples.size == 0:
            raise RuntimeError("All Monte Carlo optimisations failed")

        means = k_samples.mean(axis=0)
        stds = k_samples.std(axis=0)
        print("\nMonte-Carlo propagation results (N =", len(k_samples), ")")
        for name, m, s in zip(["k_sample", "k_ins", "k_coupler"], means, stds):
            print(f"  {name:9s}: {m:.4f} ± {s:.4f}  (1σ)")
        
        # Test the MC mean parameters with the original fixed parameters from config
        print("\nTesting MC mean parameters with config fixed parameters...")
        with open(args.config_file, "r") as f:
            cfg_test = yaml.safe_load(f)
        fixed_params_config = extract_params_from_config(cfg_test)
        
        # Build parameter vector using MC means and config fixed params
        test_vector = vector_from_dict(means, fixed_params_config)
        test_curve, _, _, _ = surrogate.predict_temperature_curves(test_vector.reshape(1, -1))
        test_curve = test_curve[0]
        test_rmse = np.sqrt(np.mean((test_curve - exp_timeseries) ** 2))
        print(f"RMSE with MC means + config fixed params: {test_rmse:.6f}")
        
        # Test a few individual MC draws to see their quality
        print("\nTesting individual MC draws...")
        for i in range(min(5, len(k_samples))):
            # Reconstruct the full parameter vector for this draw using the actual fixed params
            test_vector = vector_from_dict(k_samples[i], fixed_params_samples[i])
            test_curve, _, _, _ = surrogate.predict_temperature_curves(test_vector.reshape(1, -1))
            test_curve = test_curve[0]
            test_rmse = np.sqrt(np.mean((test_curve - exp_timeseries) ** 2))
            print(f"  Draw {i}: k={k_samples[i]}, RMSE={test_rmse:.6f}")
            
        # Find the best individual draw
        best_draw_idx = np.argmin([np.sqrt(np.mean((surrogate.predict_temperature_curves(
            vector_from_dict(k, fixed_params_samples[i]).reshape(1, -1))[0][0] - exp_timeseries) ** 2)) 
            for i, k in enumerate(k_samples)])
        best_k = k_samples[best_draw_idx]
        best_fixed = fixed_params_samples[best_draw_idx]
        print(f"\nBest individual draw (index {best_draw_idx}): k={best_k}")
        
        # Test the best draw
        best_vector = vector_from_dict(best_k, best_fixed)
        best_curve, _, _, _ = surrogate.predict_temperature_curves(best_vector.reshape(1, -1))
        best_curve = best_curve[0]
        best_rmse = np.sqrt(np.mean((best_curve - exp_timeseries) ** 2))
        print(f"Best draw RMSE: {best_rmse:.6f}")
        
        # Convert PC scores to numpy array
        pc_scores_samples = np.array(pc_scores_samples)
        
        # Optionally save
        np.savez("outputs/propagated_k_values.npz", 
                 k_samples=k_samples, 
                 mean=means, 
                 std=stds,
                 fixed_params_samples=fixed_params_samples,
                 pc_scores_samples=pc_scores_samples,
                 best_draw_idx=best_draw_idx)
        print("Saved raw samples to outputs/propagated_k_values.npz")
        
        # Compare with one-shot result using average fixed parameters
        print("\n" + "="*50)
        print("COMPARISON: One-shot vs Monte Carlo")
        print("="*50)
        print("The difference between one-shot and MC results is expected if:")
        print("1. The optimization is nonlinear with respect to fixed parameters")
        print("2. The fixed parameter distributions are asymmetric")
        print("3. There are parameter interactions")
        print("\nOne-shot uses average fixed parameters from config file")
        print("Monte Carlo draws from full distributions of fixed parameters")
        print("="*50)
        
        # Run one-shot optimization to get the central values
        print("\n" + "="*50)
        print("ONE-SHOT OPTIMIZATION (Central Values)")
        print("="*50)
        
        # Use the same approach as the one-shot section
        with open(args.config_file, "r") as f:
            cfg_oneshot = yaml.safe_load(f)
        p_oneshot = extract_params_from_config(cfg_oneshot)
        central_oneshot = np.array([p_oneshot["k_sample"], p_oneshot["k_ins"], p_oneshot["k_coupler"]], dtype=float)
        
        # Build initial guesses around central value
        rng_oneshot = np.random.default_rng(42)
        guesses_oneshot = central_oneshot + rng_oneshot.normal(scale=args.noise_std, size=(args.n_starts, 3))
        
        best_result_oneshot = None
        best_rmse_oneshot = np.inf
        
        for guess in guesses_oneshot:
            res = minimize(lambda p: np.sum((
                FullSurrogateModel.load_model("outputs/full_surrogate_model.pkl").predict_temperature_curves(
                    build_full_parameter_vector(p, args.config_file).reshape(1, -1))[0][0] - exp_timeseries) ** 2),
                           x0=guess,
                           method='L-BFGS-B',
                           bounds=bounds)
            if not res.success:
                continue
            candidate_params = res.x
            surrogate = FullSurrogateModel.load_model("outputs/full_surrogate_model.pkl")
            candidate_curve, _, _, _ = surrogate.predict_temperature_curves(
                build_full_parameter_vector(candidate_params, args.config_file).reshape(1, -1))
            candidate_curve = candidate_curve[0]
            candidate_rmse = np.sqrt(np.mean((candidate_curve - exp_timeseries) ** 2))
            if candidate_rmse < best_rmse_oneshot:
                best_rmse_oneshot = candidate_rmse
                best_result_oneshot = {
                    'params': candidate_params,
                    'curve': candidate_curve,
                    'rmse': candidate_rmse,
                }
        
        if best_result_oneshot is None:
            raise RuntimeError("One-shot optimization failed!")
        
        oneshot_params = best_result_oneshot['params']
        oneshot_rmse = best_result_oneshot['rmse']
        
        print(f"One-shot best-fit parameters (central values):")
        print(f"  k_sample:   {oneshot_params[0]:.4f} W/m/K")
        print(f"  k_ins:      {oneshot_params[1]:.4f} W/m/K") 
        print(f"  k_coupler:  {oneshot_params[2]:.4f} W/m/K")
        print(f"One-shot RMSE: {oneshot_rmse:.6f}")
        
        # Calculate uncertainty from Monte Carlo results
        print("\n" + "="*50)
        print("UNCERTAINTY QUANTIFICATION")
        print("="*50)
        print("Central values (from one-shot optimization):")
        print(f"  k_sample:   {oneshot_params[0]:.4f} W/m/K")
        print(f"  k_ins:      {oneshot_params[1]:.4f} W/m/K") 
        print(f"  k_coupler:  {oneshot_params[2]:.4f} W/m/K")
        
        print("\nUncertainties (from Monte Carlo propagation):")
        for name, s in zip(["k_sample", "k_ins", "k_coupler"], stds):
            print(f"  {name:9s}: ±{s:.4f} W/m/K (1σ)")
        
        print("\nFinal results with uncertainties:")
        for name, central, std in zip(["k_sample", "k_ins", "k_coupler"], oneshot_params, stds):
            print(f"  {name:9s}: {central:.4f} ± {std:.4f} W/m/K")
        
        # Compare with Monte Carlo mean
        print("\nComparison with Monte Carlo mean:")
        for name, oneshot, mc_mean, mc_std in zip(["k_sample", "k_ins", "k_coupler"], oneshot_params, means, stds):
            bias = mc_mean - oneshot
            print(f"  {name:9s}: One-shot={oneshot:.4f}, MC_mean={mc_mean:.4f}, Bias={bias:+.4f}")
        
        print("\n" + "="*50)
        print("INTERPRETATION")
        print("="*50)
        print("Central values: Best fit assuming fixed parameters at their most likely values")
        print("Uncertainties: Account for uncertainty in fixed parameters")
        print("Bias: Difference between MC mean and one-shot result")
        print("If bias is large, consider using MC mean as central value instead")
        
        # Check if bias is significant
        max_bias = max(abs(mc_mean - oneshot) for oneshot, mc_mean in zip(oneshot_params, means))
        max_std = max(stds)
        bias_ratio = max_bias / max_std
        
        print(f"\nBias analysis:")
        print(f"  Maximum bias: {max_bias:.4f}")
        print(f"  Maximum uncertainty: {max_std:.4f}")
        print(f"  Bias/Uncertainty ratio: {bias_ratio:.2f}")
        
        if bias_ratio > 0.5:
            print(f"\n⚠️  WARNING: Large bias detected!")
            print(f"   Consider using Monte Carlo mean as central value:")
            for name, mc_mean, mc_std in zip(["k_sample", "k_ins", "k_coupler"], means, stds):
                print(f"   {name:9s}: {mc_mean:.4f} ± {mc_std:.4f} W/m/K")
        else:
            print(f"\n✅ Bias is small relative to uncertainty - one-shot approach is reasonable")
        
        print("="*50)
        
        # Create comparison plot
        print("\nCreating Monte Carlo comparison plot...")
        time_points = np.arange(len(exp_timeseries))
        
        plt.figure(figsize=(12, 8))
        
        # Plot experimental data
        plt.plot(time_points, exp_timeseries, 'k-', linewidth=3, label='Experimental', alpha=0.9)
        
        # Plot one-shot result (central values) - use the oneshot results we just computed
        oneshot_vector = build_full_parameter_vector(oneshot_params, args.config_file)
        oneshot_curve, _, _, _ = surrogate.predict_temperature_curves(oneshot_vector.reshape(1, -1))
        plt.plot(time_points, oneshot_curve[0], 'b-', linewidth=3, label='One-shot (Central Values)', alpha=0.9)
        
        # Plot MC mean with config fixed params
        plt.plot(time_points, test_curve, 'r--', linewidth=2, label='MC Mean + Config Fixed', alpha=0.8)
        
        # Plot best individual draw
        plt.plot(time_points, best_curve, 'g--', linewidth=2, label='Best MC Draw', alpha=0.8)
        
        # Plot a few random draws
        for i in np.random.choice(len(k_samples), min(5, len(k_samples)), replace=False):
            draw_vector = vector_from_dict(k_samples[i], fixed_params_samples[i])
            draw_curve, _, _, _ = surrogate.predict_temperature_curves(draw_vector.reshape(1, -1))
            plt.plot(time_points, draw_curve[0], 'gray', linewidth=1, alpha=0.2)
        
        plt.xlabel('Time Point Index')
        plt.ylabel('Normalized Temperature')
        plt.title(f'Comparison: One-shot vs Monte Carlo (One-shot RMSE = {oneshot_rmse:.6f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig('outputs/monte_carlo_fit.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return  # end script after MC path

    # Determine central guess
    if args.central is not None:
        central = np.array(args.central, dtype=float)
    else:
        # Use values from the YAML config
        with open(args.config_file, "r") as f:
            cfg_tmp = yaml.safe_load(f)
        p_tmp = extract_params_from_config(cfg_tmp)
        central = np.array([p_tmp["k_sample"], p_tmp["k_ins"], p_tmp["k_coupler"]], dtype=float)

    # Build initial guesses around central value
    rng = np.random.default_rng(42)
    guesses = central + rng.normal(scale=args.noise_std, size=(args.n_starts, 3))

    best_result = None
    best_rmse = np.inf

    bounds = [(2.8, 4.8), (7.0, 13.0), (300, 400)]  # Same bounds as Monte Carlo
    
    for guess in guesses:
        res = minimize(lambda p: np.sum((
            FullSurrogateModel.load_model("outputs/full_surrogate_model.pkl").predict_temperature_curves(
                build_full_parameter_vector(p, args.config_file).reshape(1, -1))[0][0] - exp_timeseries) ** 2),
                       x0=guess,
                       method='L-BFGS-B',
                       bounds=bounds)
        if not res.success:
            continue
        candidate_params = res.x
        surrogate = FullSurrogateModel.load_model("outputs/full_surrogate_model.pkl")
        candidate_curve, _, _, _ = surrogate.predict_temperature_curves(
            build_full_parameter_vector(candidate_params, args.config_file).reshape(1, -1))
        candidate_curve = candidate_curve[0]
        candidate_rmse = np.sqrt(np.mean((candidate_curve - exp_timeseries) ** 2))
        if candidate_rmse < best_rmse:
            best_rmse = candidate_rmse
            best_result = {
                'params': candidate_params,
                'curve': candidate_curve,
                'rmse': candidate_rmse,
            }
    
    if best_result is None:
        raise RuntimeError("All optimization attempts failed!")
    
    best_params = best_result['params']
    predicted_curve = best_result['curve']
    rmse = best_result['rmse']
    
    print("\n" + "="*50)
    print("LEAST SQUARES RESULTS")
    print("="*50)
    print(f"Best-fit parameters:")
    print(f"  k_sample:   {best_params[0]:.3f} W/m/K")
    print(f"  k_ins:      {best_params[1]:.3f} W/m/K") 
    print(f"  k_coupler:  {best_params[2]:.3f} W/m/K")
    print(f"RMSE: {rmse:.6f}")
    
    # Test the fit by reconstructing the curve
    print("\nTesting reconstructed curve with best parameters...")
    test_params = build_full_parameter_vector(best_params, args.config_file)  # Full 11-parameter vector in correct order
    surrogate = FullSurrogateModel.load_model("outputs/full_surrogate_model.pkl")
    predicted_curve, _, _, _ = surrogate.predict_temperature_curves(test_params.reshape(1, -1))
    predicted_curve = predicted_curve[0]
    
    # Compute RMSE
    rmse = np.sqrt(np.mean((predicted_curve - exp_timeseries)**2))
    print(f"RMSE vs experimental: {rmse:.6f}")
    
    # Create comparison plot
    print("\nCreating comparison plot...")
    time_points = np.arange(len(exp_timeseries))  # Time indices
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, exp_timeseries, 'b-', linewidth=2, label='Experimental', alpha=0.8)
    plt.plot(time_points, predicted_curve, 'r--', linewidth=2, label='Surrogate (Best Fit)', alpha=0.8)
    
    plt.xlabel('Time Point Index')
    plt.ylabel('Normalized Temperature')
    plt.title(f'Least Squares Fit: RMSE = {rmse:.6f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig('outputs/least_squares_fit.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nPlot saved to: outputs/least_squares_fit.png")
    print(f"Config file used: {args.config_file}")

if __name__ == "__main__":
    try:
        from tqdm import trange
    except ImportError:
        def trange(n, desc=None):
            if desc:
                print(desc)
            return range(n)
    # Ensure variable exists for type checkers
    trange  # type: ignore[misc]
    main() 