#!/usr/bin/env python3
"""
Standalone script to examine training curves from batch results.
Plots 4 random curves from the training data to understand what the surrogate model was trained on.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def load_batch_results(input_file):
    """
    Load batch results from .npz file.
    
    Parameters:
    -----------
    input_file : str
        Path to the .npz file containing batch results
        
    Returns:
    --------
    dict
        Dictionary containing the loaded data
    """
    print(f"Loading batch results from {input_file}...")
    data = np.load(input_file)
    
    # Extract the data
    batch_data = {
        'oside_curves': data['oside_curves'],
        'parameters': data['parameters'],
        'parameter_names': data['parameter_names'],
        'timing': data.get('timing', None),
        'simulation_indices': data.get('simulation_indices', None)
    }
    
    print(f"Loaded data:")
    print(f"  Oside curves shape: {batch_data['oside_curves'].shape}")
    print(f"  Parameters shape: {batch_data['parameters'].shape}")
    print(f"  Parameter names: {batch_data['parameter_names']}")
    
    if batch_data['timing'] is not None:
        timing_array = batch_data['timing']
        print(f"  Timing array shape: {timing_array.shape}")
        if len(timing_array) > 0:
            print(f"  First timing entry: total_loop_time={timing_array[0,0]:.3f}s, avg_step_time={timing_array[0,1]:.3f}s, num_steps={timing_array[0,2]}")
    
    return batch_data

def plot_random_curves(batch_data, n_curves=4, output_file=None):
    """
    Plot n random curves from the training data.
    
    Parameters:
    -----------
    batch_data : dict
        Dictionary containing the batch results data
    n_curves : int
        Number of random curves to plot
    output_file : str, optional
        Path to save the plot
    """
    oside_curves = batch_data['oside_curves']
    parameters = batch_data['parameters']
    parameter_names = batch_data['parameter_names']
    
    # Filter out failed simulations (those with NaN values)
    valid_mask = ~np.isnan(oside_curves).any(axis=1)
    valid_curves = oside_curves[valid_mask]
    valid_params = parameters[valid_mask]
    
    print(f"Valid curves: {len(valid_curves)} out of {len(oside_curves)}")
    
    if len(valid_curves) == 0:
        print("No valid curves found!")
        return
    
    # Choose random curves
    n_curves = min(n_curves, len(valid_curves))
    random_indices = np.random.choice(len(valid_curves), n_curves, replace=False)
    
    # Create time grid (assuming uniform time steps)
    # We'll use the number of time points in the curves
    n_time_points = valid_curves.shape[1]
    
    # Try to get timing info from the data
    if batch_data['timing'] is not None:
        # Timing data is stored as [total_loop_time, avg_step_time, num_steps] for each simulation
        timing_array = batch_data['timing']
        print(f"Timing array shape: {timing_array.shape}")
        
        # Get num_steps from the first valid simulation
        valid_timing = timing_array[~np.isnan(timing_array).any(axis=1)]
        if len(valid_timing) > 0:
            num_steps = int(valid_timing[0, 2])  # num_steps is the third column
            print(f"Found num_steps from timing data: {num_steps}")
        else:
            num_steps = n_time_points
            print(f"No valid timing data, using n_time_points: {num_steps}")
        
        # For now, assume t_final based on the config (7.5 μs)
        t_final = 7.5e-6
        dt = t_final / num_steps
        time_grid = np.arange(0, num_steps) * dt
        print(f"Using timing from data: t_final={t_final:.2e}s, num_steps={num_steps}, dt={dt:.2e}s")
    else:
        # Fallback: assume 7.5 μs over n_time_points steps
        t_final = 7.5e-6
        dt = t_final / n_time_points
        time_grid = np.arange(0, n_time_points) * dt
        print(f"Using fallback timing: t_final={t_final:.2e}s, num_steps={n_time_points}, dt={dt:.2e}s")
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Temperature curves
    colors = plt.cm.tab10(np.linspace(0, 1, n_curves))
    
    for i, idx in enumerate(random_indices):
        curve = valid_curves[idx]
        params = valid_params[idx]
        
        # Plot the curve
        ax1.plot(time_grid * 1e6, curve, color=colors[i], linewidth=2, 
                label=f'Curve {i+1} (idx {idx})')
        
        # Print parameter values for this curve
        print(f"\nCurve {i+1} (index {idx}) parameters:")
        for j, name in enumerate(parameter_names):
            print(f"  {name}: {params[j]:.3e}")
    
    ax1.set_xlabel('Time (μs)')
    ax1.set_ylabel('Normalized Temperature')
    ax1.set_title(f'Training Curves (4 random samples from {len(valid_curves)} valid curves)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Parameter distributions for the selected curves
    param_data = valid_params[random_indices]
    
    # Create subplots for each parameter
    n_params = len(parameter_names)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    
    # Create a new figure for parameter distributions
    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    axes_flat = axes.flatten()
    
    for i, (name, ax) in enumerate(zip(parameter_names, axes_flat)):
        values = param_data[:, i]
        
        # Create bar plot
        bars = ax.bar(range(n_curves), values, color=colors[:n_curves])
        ax.set_title(f'{name}')
        ax.set_xlabel('Curve Index')
        ax.set_ylabel('Value')
        ax.set_xticks(range(n_curves))
        ax.set_xticklabels([f'{j+1}' for j in range(n_curves)])
        
        # Add value labels on bars
        for j, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2e}', ha='center', va='bottom', fontsize=8)
    
    # Hide unused subplots
    for i in range(n_params, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    fig2.suptitle('Parameter Values for Selected Curves', fontsize=16)
    fig2.tight_layout()
    
    # Save plots
    if output_file:
        base_name = os.path.splitext(output_file)[0]
        fig.savefig(f"{base_name}_curves.png", dpi=300, bbox_inches='tight')
        fig2.savefig(f"{base_name}_parameters.png", dpi=300, bbox_inches='tight')
        print(f"Plots saved to {base_name}_curves.png and {base_name}_parameters.png")
    
    plt.show()
    
    return fig, fig2

def main():
    """Main function to examine training curves."""
    parser = argparse.ArgumentParser(description='Examine training curves from batch results')
    parser.add_argument('--input', type=str, 
                       default='outputs/geballe/80Gpa/run1/uq_batch_results.npz',
                       help='Path to the batch results .npz file')
    parser.add_argument('--n-curves', type=int, default=4,
                       help='Number of random curves to plot (default: 4)')
    parser.add_argument('--output', type=str, default=None,
                       help='Base name for output plot files (optional)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("EXAMINING TRAINING CURVES")
    print("=" * 60)
    print(f"Input file: {args.input}")
    print(f"Number of curves to plot: {args.n_curves}")
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found!")
        return
    
    # Load batch results
    batch_data = load_batch_results(args.input)
    
    # Plot random curves
    plot_random_curves(batch_data, n_curves=args.n_curves, output_file=args.output)
    
    print("\nExamination complete!")

if __name__ == "__main__":
    main() 