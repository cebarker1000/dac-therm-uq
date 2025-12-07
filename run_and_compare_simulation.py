#!/usr/bin/env python3
"""
Minimal re-implementation of the original `run_and_compare_simulation.py` that was
accidentally deleted.  Provides `SimulationComparer` with the plotting and helper
methods needed by `compare_single_parameter_set.py` plus a simple CLI to compare
one YAML configuration.
"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import seaborn as sns
from typing import Dict, Any

from train_surrogate_models import FullSurrogateModel

plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

class SimulationComparer:
    def __init__(self, config_file: str, surrogate_model_path: str = "outputs/full_surrogate_model.pkl"):
        self.config_file = config_file
        with open(config_file, "r") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

        # Surrogate
        self.surrogate = FullSurrogateModel.load_model(surrogate_model_path)

        # The surrogate model now has the correct time grid
        self.sim_time_grid = self.surrogate.time_grid
        self.sim_t_final = self.surrogate.t_final
        self.sim_num_steps = self.surrogate.num_steps

        # Experimental file: try to get from config, fallback to default
        if "heating" in self.config and "file" in self.config["heating"]:
            self.experimental_file = self.config["heating"]["file"]
        elif "output" in self.config and "analysis" in self.config["output"] and "experimental_data_file" in self.config["output"]["analysis"]:
            self.experimental_file = self.config["output"]["analysis"]["experimental_data_file"]
        else:
            # Fallback to default (hard-coded as in original script)
            self.experimental_file = "data/experimental/geballe_heat_data.csv"

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------
    def extract_parameters_from_config(self) -> Dict[str, float]:
        mats = self.config.get("mats", {})
        p: Dict[str, float] = {}
        if "sample" in mats:
            s = mats["sample"]
            p["d_sample"] = s.get("z", 1.84e-6)
            p["rho_cv_sample"] = s.get("rho_cv", 2.764828e6)
            p["k_sample"] = s.get("k", 3.8)
        if "p_coupler" in mats:
            c = mats["p_coupler"]
            p["d_coupler"] = c.get("z", 6.2e-8)
            p["rho_cv_coupler"] = c.get("rho_cv", 3.44552e6)
            p["k_coupler"] = c.get("k", 350.0)
        if "p_ins" in mats:
            ins = mats["p_ins"]
            p["d_ins_pside"] = ins.get("z", 6.3e-6)
            p["rho_cv_ins"] = ins.get("rho_cv", 2.764828e6)
            p["k_ins"] = ins.get("k", 10.0)
        if "o_ins" in mats:
            oins = mats["o_ins"]
            p["d_ins_oside"] = oins.get("z", 3.2e-6)
        if "heating" in self.config:
            p["fwhm"] = self.config["heating"].get("fwhm", 12e-6)
        return p

    # ------------------------------------------------------------------
    # Surrogate prediction
    # ------------------------------------------------------------------
    def get_surrogate_prediction(self, params: Dict[str, float]):
        """Return surrogate outputs including per-time-step uncertainty.

        Returns
        -------
        tuple
            (curve_mean, fpca_coeffs, fpca_uncerts, curve_uncerts) – each as 1-D arrays.
        """
        # Build vector in expected order for the surrogate
        default_order = self.surrogate.parameter_names
        vector = np.array([params[name] for name in default_order])

        curves, coeffs, fpca_uncerts, curve_uncerts = self.surrogate.predict_temperature_curves(
            vector.reshape(1, -1)
        )

        # Flatten batch dimension
        return curves[0], coeffs[0], fpca_uncerts[0], curve_uncerts[0]

    # ------------------------------------------------------------------
    # Experimental data utilities
    # ------------------------------------------------------------------
    def load_experimental_data(self):
        data = np.genfromtxt(self.experimental_file, delimiter=",", names=True)
        return data["time"], data["temp"], data["oside"]
    
    def estimate_variance_from_raw_data(self):
        """Estimate sensor variance from normalized raw (unsmoothed) experimental data.
        
        This computes the variance of the normalized oside data in the baseline region only,
        where the signal should be constant, so any variation is pure sensor noise.
        Matching the logic in uqpy_MCMC.py with --estimate-variance-from-raw.
        
        Returns:
        --------
        float
            Estimated variance in normalized units
        """
        import pandas as pd
        
        # Load experimental data
        data = pd.read_csv(self.experimental_file)
        oside_data = data["oside"].values
        times = data["time"].values
        
        # Determine which pside data to use for normalization (prefer temp_raw if available)
        if "temp_raw" in data.columns:
            temp_data = data["temp_raw"].values  # Use raw (unsmoothed) pside data
            print("Using temp_raw column for variance estimation")
        else:
            temp_data = data["temp"].values  # Use regular temp column
            print("Warning: temp_raw column not found, using temp column for variance estimation")
        
        # Compute baselines
        baseline_pside = self._compute_baseline(times, temp_data)
        baseline_oside = self._compute_baseline(times, oside_data)
        
        # Compute excursion from raw pside data
        excursion_pside = (temp_data - baseline_pside).max() - (temp_data - baseline_pside).min()
        if excursion_pside <= 0.0:
            raise ValueError("Temp excursion is zero – check experimental data")
        
        # Normalize oside data using raw pside excursion
        y_obs_raw = (oside_data - baseline_oside) / excursion_pside
        
        # Get baseline time window from config
        baseline_cfg = self.config.get('baseline', {})
        use_avg = bool(baseline_cfg.get('use_average', False))
        
        if use_avg:
            t_window = float(baseline_cfg.get('time_window', 0.0))
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
                
                print(f"Variance estimation from raw experimental data (baseline region only):")
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
        
        print(f"Variance estimation from raw experimental data:")
        print(f"  Estimated variance (normalized units): {variance:.6e}")
        print(f"  Standard deviation (normalized units): {np.sqrt(variance):.6e}")
        
        return variance

    def align_experimental_data(self, exp_time, exp_data, method="linear"):
        f = interp1d(exp_time, exp_data, kind=method, bounds_error=False,
                     fill_value=(exp_data[0], exp_data[-1]))
        return f(self.sim_time_grid)

    def _compute_baseline(self, time_series: np.ndarray, temp_series: np.ndarray) -> float:
        """Compute baseline temperature using either the first data point or an
        average over an initial time window, matching simulation_engine._compute_baseline.
        
        This ensures consistent baseline calculation across all scripts.
        """
        baseline_cfg = self.config.get('baseline', {})
        use_avg = bool(baseline_cfg.get('use_average', False))
        if not use_avg:
            return float(temp_series[0])

        t_window = float(baseline_cfg.get('time_window', 0.0))
        mask = time_series <= t_window
        if mask.any():
            return float(np.mean(temp_series[mask]))
        # Fallback – no points in window.
        return float(temp_series[0])

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def create_comparison_plot(self, *, sim_results: Dict[str, Any], surrogate_curve: np.ndarray,
                               curve_uncert: np.ndarray | None = None,
                               exp_time: np.ndarray, exp_temp: np.ndarray, exp_oside: np.ndarray,
                               params: Dict[str, float], output_dir: str = "outputs"):
        if "oside_temps" not in sim_results:
            raise ValueError("sim_results must contain 'oside_temps' (normalised) and 'time_grid'.")

        sim_curve = np.array(sim_results["oside_temps"])
        sim_time  = np.array(sim_results.get("time_grid", self.sim_time_grid))

        # `sim_curve` is already normalised by its p-side excursion in the minimal
        # simulation; the surrogate prediction is generated in the same scale, so
        # we can use them directly without further scaling.

        norm_sim = sim_curve
        norm_surr = surrogate_curve

        # Normalize experimental data using the same baseline averaging logic as simulations
        # Step 1: Compute baselines from original experimental data (before interpolation)
        baseline_pside = self._compute_baseline(exp_time, exp_temp)
        baseline_oside = self._compute_baseline(exp_time, exp_oside)
        
        # Step 2: Compute p-side excursion after baseline removal
        pside_shifted = exp_temp - baseline_pside
        pside_exc_exp = pside_shifted.max() - pside_shifted.min()
        if pside_exc_exp <= 0:
            raise ValueError("P-side excursion is zero after baseline removal – check experimental data")
        
        # Step 3: Normalize o-side: subtract its own baseline, divide by p-side excursion
        # (matching the normalization logic in simulation_engine.py)
        exp_oside_normalized = (exp_oside - baseline_oside) / pside_exc_exp

        # Step 4: Interpolate simulation and surrogate onto experimental time grid
        # (matching MCMC approach: restrict to overlapping region)
        t_min = max(sim_time.min(), exp_time.min(), self.sim_time_grid.min())
        t_max = min(sim_time.max(), exp_time.max(), self.sim_time_grid.max())
        overlap_mask = (exp_time >= t_min) & (exp_time <= t_max)
        exp_time_overlap = exp_time[overlap_mask]
        exp_oside_overlap = exp_oside_normalized[overlap_mask]
        
        # Interpolate simulation onto experimental grid
        sim_interp_func = interp1d(sim_time, norm_sim, kind='linear', 
                                   bounds_error=False, fill_value=np.nan)
        norm_sim_on_exp = sim_interp_func(exp_time_overlap)
        
        # Interpolate surrogate onto experimental grid
        surr_interp_func = interp1d(self.sim_time_grid, norm_surr, kind='linear',
                                    bounds_error=False, fill_value=np.nan)
        norm_surr_on_exp = surr_interp_func(exp_time_overlap)
        
        # Interpolate uncertainty if provided
        if curve_uncert is not None:
            uncert_interp_func = interp1d(self.sim_time_grid, curve_uncert, kind='linear',
                                         bounds_error=False, fill_value=np.nan)
            curve_uncert_on_exp = uncert_interp_func(exp_time_overlap)
        else:
            curve_uncert_on_exp = None

        # Estimate variance from raw experimental data (matching MCMC logic)
        variance = self.estimate_variance_from_raw_data()
        sigma = np.sqrt(variance)
        
        # Compute ±1 sigma bands around experimental data (on experimental grid)
        exp_upper = exp_oside_overlap + sigma
        exp_lower = exp_oside_overlap - sigma

        # Compute RMSE on experimental grid (overlapping region)
        rmse_sim = np.sqrt(np.mean((norm_sim_on_exp - exp_oside_overlap) ** 2))
        rmse_sur = np.sqrt(np.mean((norm_surr_on_exp - exp_oside_overlap) ** 2))

        plt.figure(figsize=(8, 5))
        plt.plot(exp_time_overlap, norm_sim_on_exp, label="2-D Simulation (interp to exp grid)", lw=2)
        plt.plot(exp_time_overlap, norm_surr_on_exp, "r--", label="Surrogate (interp to exp grid)", lw=2)
        plt.scatter(exp_time_overlap, exp_oside_overlap, c='green', s=20, alpha=0.7, 
                   label='Experimental (raw)', zorder=5)
        
        # Plot ±1 sigma bands
        plt.fill_between(exp_time_overlap, exp_lower, exp_upper, 
                        alpha=0.2, color='green', label=f'±1σ (σ={sigma:.4f})')
        plt.title("Normalized Comparison (on Experimental Time Grid)")
        plt.xlabel("Time (s)")
        plt.ylabel("Normalized Temperature")
        plt.legend()
        plt.grid(alpha=0.3)
        
        # --- uncertainty band -------------------------------------------------
        if curve_uncert_on_exp is not None:
            # Shade ±2σ around the surrogate mean
            upper = norm_surr_on_exp + 2 * curve_uncert_on_exp
            lower = norm_surr_on_exp - 2 * curve_uncert_on_exp
            plt.fill_between(exp_time_overlap, lower, upper, color="r", alpha=0.2,
                             label="Surrogate ±2σ")

            # Mark point of maximum σ for quick visual cue
            idx_max = int(np.argmax(curve_uncert_on_exp))
            plt.scatter(exp_time_overlap[idx_max], norm_surr_on_exp[idx_max], color="k", zorder=5)
            plt.text(exp_time_overlap[idx_max], norm_surr_on_exp[idx_max],
                     f"  max σ={curve_uncert_on_exp[idx_max]:.3f}",
                     va="bottom", ha="left", fontsize=8)

        plt.tight_layout()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        config_basename = os.path.splitext(os.path.basename(self.config_file))[0]
        out = os.path.join(output_dir, f"comparison_{config_basename}.png")
        plt.savefig(out, dpi=200)
        print(f"Comparison plot saved to {out}\nRMSE Sim={rmse_sim:.4f} | Surrogate={rmse_sur:.4f}")
        print(f"Overlapping region: [{t_min:.6e}, {t_max:.6e}] s ({len(exp_time_overlap)} points)")
        plt.close()

# -----------------------------------------------------------------------------
# Simple CLI to reproduce original behaviour for one config
# -----------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare surrogate with minimal simulation and experiment")
    parser.add_argument("config_file", help="YAML config with parameters")
    parser.add_argument("--surrogate", default="outputs/full_surrogate_model.pkl", help="Path to surrogate model")
    args = parser.parse_args()

    comp = SimulationComparer(args.config_file, surrogate_model_path=args.surrogate)
    params = comp.extract_parameters_from_config()

    # Minimal simulation via compare_single_parameter_set (import lazily)
    from analysis.uq_wrapper import run_single_simulation
    # Build sample vector in correct order
    param_defs_dummy = []  # not needed because we supply sample directly later

    # (Re-using parameter mapping from compare_single_parameter_set is overkill; just call
    # run_single_simulation through that script if desired.)
    print("Please use compare_single_parameter_set.py for full workflow.")

if __name__ == "__main__":
    main() 