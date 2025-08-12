#!/usr/bin/env python3
"""
Compare the surrogate prediction to a minimal 2-D simulation **for a single parameter set**.

This script re-uses the same simulation pathway as `validate_surrogates.py` (i.e. it
invokes `analysis.uq_wrapper.run_single_simulation`, which builds a fresh in-memory
mesh for every call) and the same plotting utilities as `run_and_compare_simulation.py`.

Usage
-----
python compare_single_parameter_set.py <config_yaml> \
       --surrogate outputs/full_surrogate_model.pkl

The YAML file specifies the material thicknesses / properties exactly like in
`run_and_compare_simulation.py`.
"""

import argparse
import os
from typing import Dict, Any

import numpy as np
from analysis.uq_wrapper import run_single_simulation, load_recast_training_data
from train_surrogate_models import FullSurrogateModel
from run_and_compare_simulation import SimulationComparer  # reuse plotting & helpers
from analysis.config_utils import get_param_defs_from_config, get_param_mapping_from_config


# -----------------------------------------------------------------------------
# Load parameter definitions and mapping from config file
# -----------------------------------------------------------------------------
param_defs = get_param_defs_from_config()
param_mapping = get_param_mapping_from_config()
param_names = [p["name"] for p in param_defs]

# -----------------------------------------------------------------------------
# Helper: extract parameters from YAML config (reuse logic from SimulationComparer)
# -----------------------------------------------------------------------------

def extract_params_from_config(cfg: Dict[str, Any]) -> Dict[str, float]:
    """Duplicate of SimulationComparer.extract_parameters_from_config() without printing."""
    params: Dict[str, float] = {}
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

# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Compare surrogate vs minimal simulation for one parameter set")
    parser.add_argument("config_file", help="YAML config defining parameter values")
    parser.add_argument("--surrogate", default="outputs/full_surrogate_model.pkl", help="Path to saved surrogate model")
    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # Load surrogate and helper comparer (for plotting utilities)
    # ---------------------------------------------------------------------
    comparer = SimulationComparer(config_file=args.config_file, surrogate_model_path=args.surrogate)
    surrogate = comparer.surrogate  # FullSurrogateModel instance

    # ---------------------------------------------------------------------
    # Build parameter dictionary & vector
    # ---------------------------------------------------------------------
    import yaml
    with open(args.config_file, "r") as f:
        cfg = yaml.safe_load(f)
    params_dict = extract_params_from_config(cfg)
    
    # Debug: print parameter values
    print("\nDEBUG: Parameter values from config file:")
    for name in param_names:
        print(f"  {name}: {params_dict[name]}")
    
    sample_vec = np.array([params_dict[name] for name in param_names])

    # ---------------------------------------------------------------------
    # Run minimal 2-D simulation via UQ wrapper (fresh mesh)
    # ---------------------------------------------------------------------
    sim_result = run_single_simulation(
        sample=sample_vec,
        param_defs=param_defs,
        param_mapping=param_mapping,
        simulation_index=0,
        config_path=args.config_file,
        suppress_print=False,
    )

    # Convert watcher_data to the fields expected by create_comparison_plot
    if 'watcher_data' in sim_result and 'oside' in sim_result['watcher_data']:
        oside_entry = sim_result['watcher_data']['oside']
        sim_result['oside_temps'] = np.array(oside_entry['normalized'])
        sim_result['time_grid'] = np.array(oside_entry['time'])
        # Also capture pside excursion for correct normalisation
        if 'pside' in sim_result['watcher_data']:
            pside_entry = sim_result['watcher_data']['pside']
            sim_result['pside_excursion'] = float(pside_entry['max_excursion'])

    # ---------------------------------------------------------------------
    # Surrogate prediction for same parameters (including uncertainty)
    # ---------------------------------------------------------------------
    surrogate_curve, fpca_scores, fpca_uncert, curve_uncert = comparer.get_surrogate_prediction(params_dict)

    # ---------------------------------------------------------------------
    # Experimental data & plotting
    # ---------------------------------------------------------------------
    exp_time, exp_temp, exp_oside = comparer.load_experimental_data()
    comparer.create_comparison_plot(
        sim_results=sim_result,
        surrogate_curve=surrogate_curve,
        curve_uncert=curve_uncert,
        exp_time=exp_time,
        exp_temp=exp_temp,
        exp_oside=exp_oside,
        params=params_dict,
    )

    # use the default parameter values just once
    param_sample = np.array([p['center'] if 'center' in p else (p['low']+p['high'])/2
                            for p in param_defs])          # param_defs already in namespace
    res = run_single_simulation(param_sample, param_defs, param_mapping,
                                suppress_print=True)

    watch = res['watcher_data']['oside']
    print("raw length:", len(watch['raw']))
    print("first 5 times :", watch['time'][:5])
    print("first 5 temps :", watch['raw'][:5])


if __name__ == "__main__":
    main() 