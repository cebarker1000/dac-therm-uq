#!/usr/bin/env python3
"""
Quick script to plot k_ins draws from Edmund MCMC results with log values on linear scale.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_edmund_k_ins():
    """Plot k_ins draws from Edmund MCMC results with log values on linear scale."""
    
    # Load MCMC results
    try:
        data = np.load('mcmc_results_edmund.npz')
        samples_full = data['samples_full']
        
        # Get parameter names from Edmund config
        from analysis.config_utils import get_param_defs_from_config
        param_defs = get_param_defs_from_config(config_path="configs/distributions_edmund.yaml")
        param_names = [param_def['name'] for param_def in param_defs]
        
        # Find k_ins index
        k_ins_idx = None
        for i, name in enumerate(param_names):
            if name == 'k_ins':
                k_ins_idx = i
                break
        
        if k_ins_idx is None:
            print("Error: k_ins parameter not found in MCMC results")
            return
        
        # Extract k_ins samples (handle both 2D and 3D formats)
        if len(samples_full.shape) == 3:
            # 3D format: (n_samples, n_chains, n_dimensions)
            k_ins_samples = samples_full[:, :, k_ins_idx].flatten()
        else:
            # 2D format: (n_samples, n_dimensions)
            k_ins_samples = samples_full[:, k_ins_idx]
        
        # Take log of k_ins values
        log_k_ins_samples = np.log10(k_ins_samples)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Histogram with linear spacing (log values treated as real values)
        plt.hist(log_k_ins_samples, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('log₁₀(k_ins) (W/m/K)')
        plt.ylabel('Frequency')
        plt.title('Edmund MCMC: k_ins Distribution (Log Values on Linear Scale)')
        plt.grid(True, alpha=0.3)
        
        # Add statistics for log values
        mean_log_val = np.mean(log_k_ins_samples)
        median_log_val = np.median(log_k_ins_samples)
        std_log_val = np.std(log_k_ins_samples)
        
        plt.axvline(mean_log_val, color='red', linestyle='--', label=f'Mean: {mean_log_val:.3f}')
        plt.axvline(median_log_val, color='green', linestyle='--', label=f'Median: {median_log_val:.3f}')
        plt.legend()
        
        # Add text box with statistics (both log and original values)
        stats_text = (f'Log₁₀ Values:\n'
                     f'Mean: {mean_log_val:.3f}\n'
                     f'Median: {median_log_val:.3f}\n'
                     f'Std: {std_log_val:.3f}\n\n'
                     f'Original Values:\n'
                     f'Mean: {10**mean_log_val:.2f} W/m/K\n'
                     f'Median: {10**median_log_val:.2f} W/m/K\n'
                     f'N: {len(k_ins_samples)}')
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('edmund_k_ins_distribution.png', dpi=300, bbox_inches='tight')
        print("Plot saved to: edmund_k_ins_distribution.png")
        plt.show()
        
        # Print summary statistics
        print(f"\nEdmund MCMC k_ins Summary (Log₁₀ Values):")
        print(f"Mean: {mean_log_val:.3f}")
        print(f"Median: {median_log_val:.3f}")
        print(f"Std: {std_log_val:.3f}")
        print(f"Min: {log_k_ins_samples.min():.3f}")
        print(f"Max: {log_k_ins_samples.max():.3f}")
        
        print(f"\nEdmund MCMC k_ins Summary (Original Values):")
        print(f"Mean: {10**mean_log_val:.2f} W/m/K")
        print(f"Median: {10**median_log_val:.2f} W/m/K")
        print(f"Min: {k_ins_samples.min():.2f} W/m/K")
        print(f"Max: {k_ins_samples.max():.2f} W/m/K")
        print(f"Total samples: {len(k_ins_samples)}")
        
    except FileNotFoundError:
        print("Error: mcmc_results.npz not found in outputs/ directory")
    except Exception as e:
        print(f"Error loading MCMC results: {e}")

if __name__ == '__main__':
    plot_edmund_k_ins() 