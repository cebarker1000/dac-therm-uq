#!/usr/bin/env python3
"""
Plot surrogate model sensitivity to key thermal conductivity parameters (k_sample, k_ins, k_int)
using the Edmund‐specific surrogate and experimental data.

For each parameter we sweep across its prior uniform range while fixing all
other parameters at their nominal (central) values.  Predicted temperature
curves are plotted together with the experimental curve so that the
sensitivity of the surrogate response to each k parameter is visually
apparent.

Output:
    • Figure saved as `surrogate_sensitivity_edmund.png`
    • Console prints basic RMSE statistics at extreme and midpoint values.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from train_surrogate_models import FullSurrogateModel
from analysis.config_utils import get_param_defs_from_config
import warnings

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def load_experimental_data():
    """Load Edmund experimental oside data and normalise like in the MCMC."""
    data = pd.read_csv("data/experimental/edmund_71Gpa_run1.csv")
    oside = data["oside"].values
    y_obs = (oside - oside[0]) / (data["temp"].max() - data["temp"].iloc[0])
    exp_time = data["time"].values
    return y_obs, exp_time


def interp_to_surrogate_grid(y, t):
    """Interpolate experimental curve to the Edmund surrogate time grid."""
    sim_t_final = 8.5e-6  # seconds (from Edmund config)
    sim_steps = 50
    t_grid = np.linspace(0, sim_t_final, sim_steps)
    f = interp1d(t, y, kind="linear", bounds_error=False, fill_value=(y[0], y[-1]))
    return f(t_grid), t_grid


# -----------------------------------------------------------------------------
# Main plotting routine
# -----------------------------------------------------------------------------

def main():
    print("Loading Edmund surrogate model …")
    surrogate = FullSurrogateModel.load_model(
        "outputs/edmund1/full_surrogate_model_int_ins_match.pkl"
    )

    print("Preparing experimental data …")
    y_obs, t_exp = load_experimental_data()
    y_obs_interp, t_grid = interp_to_surrogate_grid(y_obs, t_exp)

    # Get parameter definitions
    param_defs = get_param_defs_from_config("configs/distributions_edmund.yaml")
    param_names = [p["name"] for p in param_defs]

    # Identify k parameters of interest
    k_params = ["k_sample", "k_ins", "k_int"]
    k_ranges = {}
    baseline = np.zeros(len(param_defs))

    for i, pd in enumerate(param_defs):
        name = pd["name"]
        if name in k_params:
            # All k parameters are uniform in Edmund config
            k_ranges[name] = (pd["low"], pd["high"])
            # Baseline will be filled later by midpoint of its own range
        else:
            if pd["type"] == "normal":
                baseline[i] = pd["center"]
            elif pd["type"] == "lognormal":
                baseline[i] = pd["center"]
            elif pd["type"] == "uniform":
                baseline[i] = 0.5 * (pd["low"] + pd["high"])
            else:
                raise ValueError(f"Unknown distribution type: {pd['type']}")

    # Use midpoints for k parameters in baseline too (and record)
    for name, (lo, hi) in k_ranges.items():
        idx = param_names.index(name)
        baseline[idx] = 0.5 * (lo + hi)

    print("Baseline parameter vector constructed.")

    # ---------------------------------------------------------------------
    # Create plots
    # ---------------------------------------------------------------------
    n_k = len(k_params)
    fig, axes = plt.subplots(n_k, 1, figsize=(12, 4 * n_k))
    if n_k == 1:
        axes = [axes]

    for ax, k_name in zip(axes, k_params):
        k_min, k_max = k_ranges[k_name]
        sweep_values = np.linspace(k_min, k_max, 10)
        colors = plt.cm.viridis(np.linspace(0, 1, len(sweep_values)))

        # Plot experimental reference
        ax.plot(t_grid * 1e6, y_obs_interp, "k-", lw=2, label="Experimental")

        for j, k_val in enumerate(sweep_values):
            params = baseline.copy()
            params[param_names.index(k_name)] = k_val

            try:
                y_pred, _, _, _ = surrogate.predict_temperature_curves(params.reshape(1, -1))
                y_pred = y_pred[0]
                label = f"{k_name}={k_val:.1f}" if j in [0, len(sweep_values)//2, len(sweep_values)-1] else None
                ax.plot(t_grid * 1e6, y_pred, color=colors[j], lw=1.5, alpha=0.8, label=label)
            except Exception as e:
                print(f"Prediction failed for {k_name}={k_val}: {e}")

        ax.set_xlabel("Time (μs)")
        ax.set_ylabel("Normalised Temp")
        ax.set_title(f"Surrogate Sensitivity: {k_name}")
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    out_file = "surrogate_sensitivity_edmund.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"Sensitivity plots saved to {out_file}")
    plt.show()

    # ---------------------------------------------------------------------
    # Summary statistics
    # ---------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS SUMMARY (RMSE vs experimental)")
    print("=" * 60)

    for k_name in k_params:
        k_min, k_max = k_ranges[k_name]
        for k_val in [k_min, 0.5 * (k_min + k_max), k_max]:
            params = baseline.copy()
            params[param_names.index(k_name)] = k_val
            try:
                y_pred, _, _, _ = surrogate.predict_temperature_curves(params.reshape(1, -1))
                y_pred = y_pred[0]
                rmse = np.sqrt(np.mean((y_pred - y_obs_interp) ** 2))
                print(f"{k_name}={k_val:.1f}: RMSE={rmse:.4f}")
            except Exception as e:
                print(f"{k_name}={k_val:.1f}: ERROR - {e}")


if __name__ == "__main__":
    main() 