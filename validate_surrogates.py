#!/usr/bin/env python3
"""
Compare surrogate GP model predictions to actual simulations for random parameter sets.
Draws 10 random parameter sets, predicts with surrogate, runs actual simulation, projects actual curve to FPCA, and overlays both.
"""

import sys
import os


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

from analysis.uq_wrapper import (
    run_single_simulation,
    project_curve_to_fpca,
    reconstruct_curve_from_fpca,
    load_fpca_model,
    load_recast_training_data,
)
from train_surrogate_models import FullSurrogateModel, get_parameter_ranges
import warnings

import sys

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# Number of test runs
N_TEST = 15

OUTPUT_DIR = "outputs"

# -----------------------------------------------------------------------------
# LOAD MODELS & TRAINING DATA
# -----------------------------------------------------------------------------
surrogate = FullSurrogateModel.load_model(f"{OUTPUT_DIR}/full_surrogate_model.pkl")
fpca_model = surrogate.fpca_model
param_ranges = get_parameter_ranges()
param_names = list(param_ranges.keys())

# Load original training parameters to measure distance-to-training
try:
    recast_data = load_recast_training_data(f"{OUTPUT_DIR}/training_data_fpca.npz")
    X_train = recast_data["parameters"]
    X_train_scaled = surrogate.scaler.transform(X_train)
    nn_model = NearestNeighbors(n_neighbors=1).fit(X_train_scaled)
except Exception as e:
    print("[WARNING] Could not load training parameters for distance diagnostics:", e)
    nn_model = None

# Draw random parameter sets from the SAME distributions used for training
print(f"\nDrawing {N_TEST} test samples from the SAME distributions used for training...")

# Load parameter definitions from config file (same as training)
from analysis.config_utils import get_param_defs_from_config, get_param_mapping_from_config
param_defs = get_param_defs_from_config()

# Generate samples from the same distributions as training
test_samples = np.zeros((N_TEST, len(param_names)))
for i, param_def in enumerate(param_defs):
    if param_def["type"] == "lognormal":
        # Generate lognormal samples
        mu = np.log(param_def["center"])
        sigma = param_def["sigma_log"]
        log_samples = np.random.normal(mu, sigma, N_TEST)
        test_samples[:, i] = np.exp(log_samples)
    elif param_def["type"] == "uniform":
        # Generate uniform samples
        test_samples[:, i] = np.random.uniform(param_def["low"], param_def["high"], N_TEST)
    else:
        raise ValueError(f"Unknown parameter type: {param_def['type']}")

print(f"Generated test samples from training distributions:")
for i, param_def in enumerate(param_defs):
    samples_for_param = test_samples[:, i]
    print(f"  {param_def['name']}: mean={np.mean(samples_for_param):.2e}, std={np.std(samples_for_param):.2e}")

# Compare with the old uniform sampling approach
print(f"\nComparison with old uniform sampling approach:")
old_low = [param_ranges[name][0] for name in param_names]
old_high = [param_ranges[name][1] for name in param_names]
old_uniform_samples = np.random.uniform(low=old_low, high=old_high, size=(N_TEST, len(param_names)))

for i, param_def in enumerate(param_defs):
    old_samples = old_uniform_samples[:, i]
    new_samples = test_samples[:, i]
    print(f"  {param_def['name']}:")
    print(f"    Training dist: mean={np.mean(new_samples):.2e}, std={np.std(new_samples):.2e}")
    print(f"    Uniform range: mean={np.mean(old_samples):.2e}, std={np.std(old_samples):.2e}")
    print(f"    Range overlap: {np.min(new_samples):.2e} to {np.max(new_samples):.2e} vs {old_low[i]:.2e} to {old_high[i]:.2e}")

# Prepare containers for extended diagnostics
results = []
diagnostics = []  # list of dicts – one per successful simulation

# Load parameter mapping from config file
param_mapping = get_param_mapping_from_config()

for i, params in enumerate(test_samples):
    print(f"\nTest {i+1}/{N_TEST}")
    print(params)
    # Surrogate prediction
    surrogate_curve, surrogate_coeffs, _, _ = surrogate.predict_temperature_curves(params)
    surrogate_curve = surrogate_curve[0]  # shape (n_timepoints,)
    surrogate_coeffs = surrogate_coeffs[0]

    # Actual simulation
    sim_result = run_single_simulation(
        sample=params,
        param_defs=param_defs,
        param_mapping=param_mapping,
        simulation_index=i,
        config_path="configs/config_5_materials.yaml",
        suppress_print=True
    )
    if 'error' in sim_result:
        print(f"Simulation failed: {sim_result['error']}")
        results.append({'params': params, 'surrogate_curve': surrogate_curve, 'sim_curve': None, 'surrogate_coeffs': surrogate_coeffs, 'sim_coeffs': None})
        continue
    if 'watcher_data' not in sim_result or 'oside' not in sim_result['watcher_data']:
        print("Simulation did not return valid oside curve.")
        results.append({'params': params, 'surrogate_curve': surrogate_curve, 'sim_curve': None, 'surrogate_coeffs': surrogate_coeffs, 'sim_coeffs': None})
        continue
    sim_curve = sim_result['watcher_data']['oside']['normalized']
    # Project actual curve to FPCA
    sim_coeffs = project_curve_to_fpca(sim_curve, fpca_model)
    # Reconstruct from FPCA (should be nearly identical to sim_curve)
    sim_curve_reconstructed = reconstruct_curve_from_fpca(sim_coeffs, fpca_model)

    # ---------------------------------------------------------------------
    # DIAGNOSTIC METRICS
    # ---------------------------------------------------------------------
    # FPCA truncation error (simulation vs. reconstruction with identical PCs)
    fpca_trunc_err = sim_curve - sim_curve_reconstructed
    rmse_fpca_trunc = np.sqrt(np.mean(fpca_trunc_err ** 2))

    # Coefficient errors & GP predictive std
    coeff_err_vec = surrogate_coeffs - sim_coeffs
    coeff_err_l2 = np.linalg.norm(coeff_err_vec)
    coeff_err_abs = np.abs(coeff_err_vec)

    # GP posterior std for each component
    X_scaled = surrogate.scaler.transform(params.reshape(1, -1))
    gp_stds = [gp.predict(X_scaled, return_std=True)[1][0] for gp in surrogate.gps]

    # GP-induced curve error (with same truncation)
    gp_curve_err = surrogate_curve - sim_curve_reconstructed
    rmse_gp_curve = np.sqrt(np.mean(gp_curve_err ** 2))

    # Total curve error
    total_curve_err = surrogate_curve - sim_curve
    rmse_total = np.sqrt(np.mean(total_curve_err ** 2))

    # Distance to nearest training point (scaled parameter space)
    if nn_model is not None:
        dist, _ = nn_model.kneighbors(X_scaled)
        nearest_dist = dist[0][0]
    else:
        nearest_dist = np.nan

    diagnostics.append({
        "rmse_fpca_trunc": rmse_fpca_trunc,
        "rmse_gp_curve": rmse_gp_curve,
        "rmse_total": rmse_total,
        "coeff_err_l2": coeff_err_l2,
        **{f"coeff_err_pc{i+1}": coeff_err_abs[i] for i in range(len(coeff_err_vec))},
        **{f"gp_std_pc{i+1}": gp_stds[i] for i in range(len(gp_stds))},
        "nearest_dist": nearest_dist,
    })

    results.append({
        'params': params,
        'surrogate_curve': surrogate_curve,
        'sim_curve': sim_curve,
        'surrogate_coeffs': surrogate_coeffs,
        'sim_coeffs': sim_coeffs,
        'sim_curve_reconstructed': sim_curve_reconstructed
    })

# Plot all curves on the same plot
plt.figure(figsize=(12, 8))

# Plot all test cases on the same plot
for i, res in enumerate(results):
    if res['sim_curve'] is not None:
        plt.plot(res['sim_curve'], color='black', linewidth=2, alpha=0.7, 
                label='Actual Simulation' if i == 0 else "")
        plt.plot(res['sim_curve_reconstructed'], '--', color='gray', linewidth=1, alpha=0.7,
                label='Sim FPCA Reconstruction' if i == 0 else "")
    plt.plot(res['surrogate_curve'], color='tab:blue', linewidth=2, alpha=0.7,
            label='Surrogate Prediction' if i == 0 else "")

plt.title('Surrogate Validation: All Test Cases')
plt.xlabel('Time Step')
plt.ylabel('Normalized Temperature')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/surrogate_vs_simulation_all.png', dpi=200)
plt.close()
print(f"Saved combined overlay plot to outputs/surrogate_vs_simulation_all.png")

# -----------------------------------------------------------------------------
# SAVE & VISUALISE DIAGNOSTICS
# -----------------------------------------------------------------------------

if diagnostics:
    diag_df = pd.DataFrame(diagnostics)
    diag_csv_path = os.path.join(OUTPUT_DIR, "surrogate_diagnostics.csv")
    diag_df.to_csv(diag_csv_path, index=False)
    print(f"Saved diagnostic metrics to {diag_csv_path}")

    # Scatter: GP std vs. coefficient absolute error per PC
    num_pcs = len(surrogate.gps)
    for pc in range(num_pcs):
        plt.figure(figsize=(6, 4))
        plt.scatter(diag_df[f"gp_std_pc{pc+1}"], diag_df[f"coeff_err_pc{pc+1}"], c='tab:blue')
        plt.xlabel("GP posterior σ (PC%d)" % (pc + 1))
        plt.ylabel("|Coeff error| (PC%d)" % (pc + 1))
        plt.title(f"PC{pc+1}: Prediction uncert. vs. abs. error")
        plt.grid(alpha=0.3)
        fname = os.path.join(OUTPUT_DIR, f"gp_std_vs_coeff_err_pc{pc+1}.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"Saved scatter plot {fname}")

    # Scatter: distance to nearest training vs total curve RMSE
    if diag_df["nearest_dist"].notna().any():
        plt.figure(figsize=(6, 4))
        plt.scatter(diag_df["nearest_dist"], diag_df["rmse_total"], c='tab:green')
        plt.xlabel("Distance to nearest training point (scaled)")
        plt.ylabel("Curve RMSE (total)")
        plt.title("Extrapolation vs. error")
        plt.grid(alpha=0.3)
        fname = os.path.join(OUTPUT_DIR, "dist_vs_curve_rmse.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"Saved scatter plot {fname}")


# Summary of validation results
if diagnostics:
    diag_df = pd.DataFrame(diagnostics)
    
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total test samples: {len(diagnostics)}")
    print(f"Mean total curve RMSE: {np.mean(diag_df['rmse_total']):.6f}")
    print(f"Mean GP curve RMSE: {np.mean(diag_df['rmse_gp_curve']):.6f}")
    print(f"Mean FPCA truncation RMSE: {np.mean(diag_df['rmse_fpca_trunc']):.6f}")
    print(f"Mean coefficient L2 error: {np.mean(diag_df['coeff_err_l2']):.6f}")
    
    if diag_df["nearest_dist"].notna().any():
        print(f"Mean distance to nearest training point: {np.mean(diag_df['nearest_dist']):.4f}")
        print(f"Max distance to nearest training point: {np.max(diag_df['nearest_dist']):.4f}")
    
    print(f"\nThis validation used the SAME distributions as training data.")
    print(f"If the surrogate is working correctly, these results should be much better")
    print(f"than the previous uniform sampling approach.")

print("\nAll overlays and diagnostics complete. Check the outputs/ directory for results.") 