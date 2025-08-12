#!/usr/bin/env python3
"""
Script to compare FEM vs surrogate at multiple parameter combinations.

This script runs three different comparisons:
1. Central values of all distributions
2. Random combination of low values (1-2 sigma below mean)
3. Random combination of high values (1-2 sigma above mean)

For each combination, it runs both FEM and surrogate simulations and creates comparison plots.
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analysis.config_utils import get_param_defs_from_config, get_param_mapping_from_config
from train_surrogate_models import FullSurrogateModel


def get_central_values(param_defs):
    """Extract central values for all parameters"""
    central_values = {}
    
    for param_def in param_defs:
        name = param_def['name']
        param_type = param_def['type']
        
        if param_type == 'lognormal':
            central_values[name] = param_def['center']
        elif param_type == 'normal':
            central_values[name] = param_def['center']
        elif param_type == 'uniform':
            central_values[name] = (param_def['low'] + param_def['high']) / 2
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
    
    return central_values


def get_random_low_values(param_defs, sigma_factor=1.5):
    """Generate random low values (1-2 sigma below mean)"""
    low_values = {}
    
    for param_def in param_defs:
        name = param_def['name']
        param_type = param_def['type']
        
        if param_type == 'lognormal':
            center = param_def['center']
            sigma_log = param_def['sigma_log']
            # Generate random value in lower tail
            log_center = np.log(center)
            low_log = log_center - sigma_factor * sigma_log
            low_values[name] = np.exp(low_log)
            
        elif param_type == 'normal':
            center = param_def['center']
            sigma = param_def['sigma']
            # Generate random value in lower tail
            low_values[name] = center - sigma_factor * sigma
            
        elif param_type == 'uniform':
            low = param_def['low']
            high = param_def['high']
            # Generate random value in lower third
            low_values[name] = low + np.random.uniform(0, 0.3) * (high - low)
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
    
    return low_values


def get_random_high_values(param_defs, sigma_factor=1.5):
    """Generate random high values (1-2 sigma above mean)"""
    high_values = {}
    
    for param_def in param_defs:
        name = param_def['name']
        param_type = param_def['type']
        
        if param_type == 'lognormal':
            center = param_def['center']
            sigma_log = param_def['sigma_log']
            # Generate random value in upper tail
            log_center = np.log(center)
            high_log = log_center + sigma_factor * sigma_log
            high_values[name] = np.exp(high_log)
            
        elif param_type == 'normal':
            center = param_def['center']
            sigma = param_def['sigma']
            # Generate random value in upper tail
            high_values[name] = center + sigma_factor * sigma
            
        elif param_type == 'uniform':
            low = param_def['low']
            high = param_def['high']
            # Generate random value in upper third
            high_values[name] = high - np.random.uniform(0, 0.3) * (high - low)
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
    
    return high_values


def run_fem_simulation(param_values, config_path, param_defs, param_mapping):
    """Run FEM simulation with given parameter values"""
    from analysis.uq_wrapper import run_single_simulation
    import time
    
    print("\nRunning FEM simulation...")
    start_time = time.time()
    
    try:
        # Convert param_values dict to array format expected by run_single_simulation
        param_names = [param_def['name'] for param_def in param_defs]
        param_array = np.array([param_values[name] for name in param_names])
        
        result = run_single_simulation(
            sample=param_array,
            param_defs=param_defs,
            param_mapping=param_mapping,
            simulation_index=0,
            config_path=config_path
        )
        
        fem_elapsed = time.time() - start_time
        print(f"FEM simulation completed in {fem_elapsed:.2f} seconds")

        if 'watcher_data' in result and 'oside' in result['watcher_data']:
            fem_curve = result['watcher_data']['oside']['normalized']
            fem_time_array = result['watcher_data']['oside']['time']
            return fem_curve, fem_time_array, result
        else:
            print("ERROR: No watcher data found in FEM result")
            return None, None, result
            
    except Exception as e:
        print(f"ERROR in FEM simulation: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def run_surrogate_prediction(param_values, surrogate):
    """Run surrogate prediction with given parameter values"""
    # Filter out error_inflation parameter if present
    surrogate_param_values = {k: v for k, v in param_values.items() if k != 'error_inflation'}
    
    # Convert parameter values to array format expected by surrogate
    param_names = surrogate.parameter_names
    param_array = np.array([[surrogate_param_values[name] for name in param_names]])
    
    # Get predictions
    y_pred, _, _, curve_uncert = surrogate.predict_temperature_curves(param_array)
    
    return y_pred[0], curve_uncert[0]  # Return first (and only) prediction


def compare_fem_surrogate(param_values, config_path, surrogate, comparison_name, output_dir, param_defs, param_mapping):
    """Compare FEM vs surrogate for given parameter values"""
    print(f"\n{'='*60}")
    print(f"COMPARISON: {comparison_name}")
    print(f"{'='*60}")
    
    # Print parameter values
    print("Parameter values:")
    for name, value in param_values.items():
        if name != 'error_inflation':  # Skip error inflation for display
            print(f"  {name}: {value:.3e}")
    
    # Run surrogate prediction
    print("\nRunning surrogate prediction...")
    surrogate_pred, surrogate_uncert = run_surrogate_prediction(param_values, surrogate)
    
    print(f"Surrogate prediction shape: {surrogate_pred.shape}")
    print(f"Surrogate uncertainty shape: {surrogate_uncert.shape}")
    print(f"Prediction range: {surrogate_pred.min():.4f} to {surrogate_pred.max():.4f}")
    print(f"Uncertainty range: {surrogate_uncert.min():.4f} to {surrogate_uncert.max():.4f}")
    
    # Run FEM simulation
    print("\nRunning FEM simulation...")
    fem_curve, fem_time, fem_result = run_fem_simulation(param_values, config_path, param_defs, param_mapping)
    
    return surrogate_pred, surrogate_uncert, fem_curve, fem_time


def create_combined_fem_surrogate_plot(results, output_dir):
    """Create a single plot with all three FEM vs surrogate comparisons"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Colors for the three comparisons
    colors = {'central': 'blue', 'low': 'red', 'high': 'green'}
    labels = {'central': 'Central Values', 'low': 'Low Values', 'high': 'High Values'}
    
    # Time grid for surrogate (assuming 120 time steps)
    surrogate_time = np.linspace(0, 1, len(results['central'][0]))
    
    # Plot all three comparisons
    for comparison_type in ['central', 'low', 'high']:
        surrogate_pred, surrogate_uncert, fem_curve, fem_time = results[comparison_type]
        color = colors[comparison_type]
        label = labels[comparison_type]
        
        # Plot FEM results first (if available)
        if fem_curve is not None and fem_time is not None:
            # Normalize FEM time to [0,1] for comparison
            fem_time_norm = (fem_time - fem_time.min()) / (fem_time.max() - fem_time.min())
            ax1.plot(fem_time_norm, fem_curve, color=color, linestyle='-', 
                    label=f'FEM {label}', linewidth=2, alpha=0.8)
            
            # Plot surrogate prediction as dotted line over FEM
            # Interpolate surrogate to FEM time points for comparison
            from scipy.interpolate import interp1d
            interp_func = interp1d(surrogate_time, surrogate_pred, kind='linear', 
                                   bounds_error=False, fill_value="extrapolate")
            surrogate_interp = interp_func(fem_time_norm)
            
            ax1.plot(fem_time_norm, surrogate_interp, color=color, linestyle=':', 
                    label=f'Surrogate {label}', linewidth=2, alpha=0.8)
            
            # Calculate and display error metrics
            mse = np.mean((surrogate_interp - fem_curve)**2)
            rmse = np.sqrt(mse)
            max_error = np.max(np.abs(surrogate_interp - fem_curve))
            
            # Add error metrics to plot
            ax1.text(0.02, 0.95 - 0.1 * list(colors.keys()).index(comparison_type), 
                    f'{label}: RMSE={rmse:.4f}, Max={max_error:.4f}', 
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    color=color, fontsize=9)
        else:
            # If no FEM results, just plot surrogate
            ax1.plot(surrogate_time, surrogate_pred, color=color, linestyle=':', 
                    label=f'Surrogate {label}', linewidth=2, alpha=0.8)
        
        # Plot surrogate uncertainty as background
        ax1.fill_between(surrogate_time, 
                         surrogate_pred - surrogate_uncert, 
                         surrogate_pred + surrogate_uncert, 
                         alpha=0.1, color=color, label=f'Uncertainty {label}')
        
        # Plot uncertainty comparison
        ax2.plot(surrogate_time, surrogate_uncert, color=color, linestyle='-', 
                label=f'Uncertainty {label}', linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Normalized Time')
    ax1.set_ylabel('Normalized Temperature')
    ax1.set_title('FEM vs Surrogate Comparison: All Parameter Combinations')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Normalized Time')
    ax2.set_ylabel('Prediction Uncertainty')
    ax2.set_title('Uncertainty Comparison: All Parameter Combinations')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'fem_surrogate_comparison_all.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined comparison plot saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare FEM vs surrogate at multiple parameter combinations")
    parser.add_argument('--distributions_path', type=str, required=True, 
                       help="Path to the distributions YAML file")
    parser.add_argument('--config_path', type=str, required=True,
                       help="Path to the simulation config file")
    parser.add_argument('--surrogate_path', type=str, required=True,
                       help="Path to the surrogate model file")
    parser.add_argument('--output_dir', type=str, required=True,
                       help="Output directory for results")
    parser.add_argument('--seed', type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument('--sigma_factor', type=float, default=1.5,
                       help="Sigma factor for low/high value generation")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load components
    print("Loading components...")
    param_defs = get_param_defs_from_config(args.distributions_path)
    param_mapping = get_param_mapping_from_config(args.distributions_path)
    surrogate = FullSurrogateModel.load_model(args.surrogate_path)
    
    # Generate parameter combinations
    print("Generating parameter combinations...")
    
    # 1. Central values
    central_values = get_central_values(param_defs)
    print("Central values:")
    for name, value in central_values.items():
        if name != 'error_inflation':
            print(f"  {name}: {value:.3e}")
    
    # 2. Random low values
    low_values = get_random_low_values(param_defs, args.sigma_factor)
    print(f"\nRandom low values (sigma_factor={args.sigma_factor}):")
    for name, value in low_values.items():
        if name != 'error_inflation':
            print(f"  {name}: {value:.3e}")
    
    # 3. Random high values
    high_values = get_random_high_values(param_defs, args.sigma_factor)
    print(f"\nRandom high values (sigma_factor={args.sigma_factor}):")
    for name, value in high_values.items():
        if name != 'error_inflation':
            print(f"  {name}: {value:.3e}")
    
    # Run comparisons
    results = {}
    
    # Comparison 1: Central values
    results['central'] = compare_fem_surrogate(
        central_values, args.config_path, surrogate, "Central Values", output_dir, param_defs, param_mapping
    )
    
    # Comparison 2: Random low values
    results['low'] = compare_fem_surrogate(
        low_values, args.config_path, surrogate, "Random Low Values", output_dir, param_defs, param_mapping
    )
    
    # Comparison 3: Random high values
    results['high'] = compare_fem_surrogate(
        high_values, args.config_path, surrogate, "Random High Values", output_dir, param_defs, param_mapping
    )
    
    # Create combined plot with all three comparisons
    create_combined_fem_surrogate_plot(results, output_dir)
    
    # Save results
    print(f"\nSaving results to {output_dir}")
    
    # Save parameter values
    param_results = {
        'central': central_values,
        'low': low_values,
        'high': high_values,
        'metadata': {
            'sigma_factor': args.sigma_factor,
            'seed': args.seed,
            'distributions_path': args.distributions_path,
            'config_path': args.config_path,
            'surrogate_path': args.surrogate_path
        }
    }
    
    with open(output_dir / 'parameter_combinations.yaml', 'w') as f:
        yaml.dump(param_results, f, default_flow_style=False)
    
    # Save surrogate predictions
    np.savez(output_dir / 'surrogate_predictions.npz',
             central_pred=results['central'][0],
             central_uncert=results['central'][1],
             low_pred=results['low'][0],
             low_uncert=results['low'][1],
             high_pred=results['high'][0],
             high_uncert=results['high'][1])
    
    print("Comparison complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main() 