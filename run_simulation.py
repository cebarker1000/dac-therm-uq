"""
This script is the main entry point for running individual heat flow simulations.

It uses a YAML configuration file to define the simulation parameters, such as
material properties, boundary conditions, and timing settings. The script can
be run from the command line, and it provides several options for controlling
the simulation, such as rebuilding the mesh, visualizing the mesh, and
suppressing output.

After the simulation is complete, the script generates a set of plots to
visualize the results and compare them with experimental data. It also saves
the simulation results and the configuration file to an output directory for
reproducibility.
"""

import os
import sys
import yaml
import argparse
import time
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Baseline helper (same rules as core.simulation_engine)
# ------------------------------------------------------------


def _compute_baseline(times: np.ndarray, temps: np.ndarray, cfg: Dict[str, Any]):
    """Compute baseline according to cfg['baseline'] settings."""
    baseline_cfg = cfg.get('baseline', {})
    if not baseline_cfg.get('use_average', False):
        return float(temps[0])

    t_window = float(baseline_cfg.get('time_window', 0.0))
    mask = times <= t_window
    if mask.any():
        return float(np.mean(temps[mask]))
    return float(temps[0])

from core.simulation_engine import OptimizedSimulationEngine, suppress_output
from analysis import analysis_utils as au


def get_default_paths():
    """Get default paths from environment variables or config."""
    return {
        'mesh_dir': os.getenv('V2HEATFLOW_MESH_DIR', 'data/meshes'),
        'output_dir': os.getenv('V2HEATFLOW_OUTPUT_DIR', 'outputs'),
        'experimental_data': os.getenv('V2HEATFLOW_EXPERIMENTAL_DATA', None)
    }



def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Validate required sections
    required_sections = ['heating', 'mats', 'timing']
    for section in required_sections:
        if section not in cfg:
            raise ValueError(f"Missing required configuration section: {section}")
    
    return cfg


def setup_paths(cfg: Dict[str, Any], mesh_folder: str = None, output_folder: str = None) -> tuple:
    """Setup mesh and output folders based on configuration."""
    # Determine mesh folder
    if mesh_folder is None:
        mesh_folder = cfg.get('io', {}).get('mesh_path', 'meshes/default')
    
    # Determine output folder
    if output_folder is None:
        sim_name = cfg.get('simulation_name', 'default_simulation')
        output_folder = f'outputs/{sim_name}'
    
    # Create directories
    os.makedirs(mesh_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    
    return mesh_folder, output_folder


def run_simulation(cfg: Dict[str, Any], output_dir: str,
                  rebuild_mesh: bool = False, suppress_output_flag: bool = False,
                  no_plots: bool = False, no_xdmf: bool = False, mesh_vis: bool = False,
                  mesh_dir: Optional[str] = None, experimental_data: Optional[str] = None,
                  config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Run the heat flow simulation using the optimized engine.
    
    Parameters:
    -----------
    cfg : dict
        Configuration dictionary loaded from YAML
    output_dir : str
        Where to save simulation outputs
    rebuild_mesh : bool, optional
        Whether to rebuild the mesh and update material tags
    suppress_output_flag : bool, optional
        If True, suppress all print output
    no_plots : bool, optional
        If True, skip plotting temperature curves
    no_xdmf : bool, optional
        If True, skip XDMF file creation
    mesh_vis : bool, optional
        Whether to visualize the mesh
    mesh_dir : str, optional
        Override mesh directory path
    experimental_data : str, optional
        Override experimental data file path
    config_path : str, optional
        Path to the configuration file
    
    Returns:
    --------
    dict
        Simulation results including timing information
    """
    
    # Get default paths
    defaults = get_default_paths()
    
    # Determine mesh directory
    if mesh_dir is None:
        mesh_dir = cfg.get('io', {}).get('mesh_path', defaults['mesh_dir'])
    
    # Override experimental data path if provided
    if experimental_data is not None:
        cfg['output']['analysis']['experimental_data_file'] = experimental_data
    
    # Disable XDMF if requested
    if no_xdmf:
        cfg['output']['xdmf']['enabled'] = False
    
    # Create simulation engine
    engine = OptimizedSimulationEngine(cfg, mesh_dir, output_dir, config_path)
    
    # Run simulation
    results = engine.run(
        rebuild_mesh=rebuild_mesh,
        visualize_mesh=mesh_vis,
        suppress_print=suppress_output_flag
    )
    
    # Save configuration used for this run
    config_save_path = os.path.join(output_dir, 'used_config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    
    # Plot results if requested
    if not no_plots:
        with suppress_output(suppress_output_flag):
            plot_temperature_curves(cfg, output_dir)
    
    return results


def plot_temperature_curves(cfg, output_folder):
    """
    Generates and saves a set of plots to visualize the simulation results and
    compare them with experimental data.

    This function uses the utility functions from `analysis/analysis_utils.py`
    to load and process the data, create the plots, and calculate performance
    metrics.
    
    Parameters:
    -----------
    cfg : dict
        Configuration dictionary
    output_folder : str
        Path to the output folder
    """
    try:
        # Load and process data using the utility function
        processed_data = au.load_and_process_data(cfg, output_folder)

        # Plot normalized temperature curves
        plot_path = os.path.join(output_folder, 'temperature_curves.png')
        au.plot_temperature_curves(
            sim_time=processed_data['sim_time'],
            sim_pside=processed_data['sim_pside_normed'],
            sim_oside=processed_data['sim_oside_normed'],
            exp_time=processed_data['exp_time'],
            exp_pside=processed_data['exp_pside_normed'],
            exp_oside=processed_data['exp_oside_normed'],
            exp_pside_raw=processed_data['exp_pside_raw_normed'],
            save_path=plot_path,
            show_plot=True
        )

        # Create residual plot for oside data
        residual_plot_path = os.path.join(output_folder, 'residual_plot.png')
        au.plot_residuals(
            exp_time=processed_data['exp_time'],
            exp_data=processed_data['exp_oside_normed'],
            sim_time=processed_data['sim_time'],
            sim_data=processed_data['sim_oside_normed'],
            save_path=residual_plot_path,
            show_plot=True
        )

        # Calculate RMSE and residual variance for oside data
        oside_rmse = au.calculate_rmse(
            exp_time=processed_data['exp_time'],
            exp_data=processed_data['exp_oside_normed'],
            sim_time=processed_data['sim_time'],
            sim_data=processed_data['sim_oside_normed']
        )
        oside_residual_stats = au.calculate_residual_variance(
            exp_time=processed_data['exp_time'],
            exp_data=processed_data['exp_oside_normed'],
            sim_time=processed_data['sim_time'],
            sim_data=processed_data['sim_oside_normed']
        )
        
        # Print and save analysis results
        print_analysis_summary(output_folder, oside_rmse, oside_residual_stats)
        
    except Exception as e:
        print(f"Error plotting results: {e}")
        import traceback
        traceback.print_exc()

def print_analysis_summary(output_folder, rmse, residual_stats):
    """Prints and saves a summary of the analysis results."""
    summary = (
        f"--- Analysis Results ---\n"
        f"O-side RMSE: {rmse:.4f}\n"
        f"Residual Variance (sensor noise): {residual_stats['variance']:.6f}\n"
        f"Residual Std Dev (sensor noise): {residual_stats['std']:.6f}\n"
        f"------------------------\n"
    )
    print(summary)
    
    # Save residual statistics to a file
    residual_stats_path = os.path.join(output_folder, 'residual_statistics.txt')
    with open(residual_stats_path, 'w') as f:
        f.write(summary)
    print(f"Residual statistics saved to: {residual_stats_path}")


def print_timing_summary(results: Dict[str, Any]):
    """Print timing summary from simulation results."""
    timing = results.get('timing', {})
    
    print("\n" + "="*50)
    print("SIMULATION TIMING SUMMARY")
    print("="*50)
    print(f"Total simulation time: {timing.get('total_loop_time', 0):.2f} seconds")
    print(f"Average time per step: {timing.get('avg_step_time', 0):.4f} seconds")
    print(f"Number of time steps: {results.get('num_steps', 0)}")
    
    if timing.get('total_loop_time', 0) > 0:
        steps_per_second = results.get('num_steps', 0) / timing.get('total_loop_time', 1)
        print(f"Simulation speed: {steps_per_second:.2f} steps/second")
    
    print("="*50)


def main():
    """Main entry point for the optimized simulation runner."""
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description='Optimized heat flow simulation runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Define command-line arguments
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the YAML configuration file')
    parser.add_argument('--mesh-folder', type=str, default=None,
                        help='Override mesh folder path from config')
    parser.add_argument('--output-folder', type=str, default=None,
                        help='Override output folder path from config')
    parser.add_argument('--rebuild-mesh', action='store_true',
                        help='Rebuild the mesh')
    parser.add_argument('--visualize-mesh', action='store_true',
                        help='Visualize the mesh')
    parser.add_argument('--suppress-output', action='store_true',
                        help='Suppress all output')
    parser.add_argument('--no-plots', action='store_true',
                        help='Do not generate plots')
    
    # Parse the arguments
    args = parser.parse_args()
    
    try:
        # Load the simulation configuration from the specified YAML file
        cfg = load_config(args.config)
        
        # Set up the mesh and output paths, using command-line overrides if provided
        mesh_folder, output_folder = setup_paths(cfg, args.mesh_folder, args.output_folder)
        
        # Run the simulation with the specified configuration and options
        with suppress_output(args.suppress_output):
            print(f"Starting simulation with configuration: {args.config}")
            print(f"Mesh folder: {mesh_folder}")
            print(f"Output folder: {output_folder}")

            results = run_simulation(
                cfg=cfg,
                output_dir=output_folder,
                rebuild_mesh=args.rebuild_mesh,
                suppress_output_flag=args.suppress_output,
                no_plots=args.no_plots,
                mesh_vis=args.visualize_mesh,
                mesh_dir=mesh_folder,
                config_path=args.config
            )

            # Print a summary of the simulation timing
            print_timing_summary(results)
            print(f"\nSimulation completed successfully!")
            print(f"Results saved to: {output_folder}")

    except Exception as e:
        # Print an error message if the simulation fails
        print(f"Error running simulation: {e}")
        raise


if __name__ == '__main__':
    main() 