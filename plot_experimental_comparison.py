#!/usr/bin/env python3
"""
Plot comparison between pside and oside for experimental data files.

This script loads one or more experimental data files, normalizes them using
baseline averaging windows, and plots both pside and oside temperatures for
comparison.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional


def compute_baseline(times: np.ndarray, temps: np.ndarray, time_window: float) -> float:
    """Compute baseline temperature as average over the specified time window.
    
    Parameters:
    -----------
    times : np.ndarray
        Time values in seconds
    temps : np.ndarray
        Temperature values
    time_window : float
        Time window in seconds (all points with time <= time_window are used)
        
    Returns:
    --------
    float
        Baseline temperature (average over window, or first point if no window points)
    """
    mask = times <= time_window
    if mask.any():
        return float(np.mean(temps[mask]))
    return float(temps[0])


def load_and_normalize_data(data_path: str, baseline_window_us: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load experimental data and normalize pside and oside.
    
    Parameters:
    -----------
    data_path : str
        Path to CSV file with columns: time, temp (pside), oside
    baseline_window_us : float
        Baseline averaging window in microseconds
        
    Returns:
    --------
    times : np.ndarray
        Time values in seconds
    pside_norm : np.ndarray
        Normalized pside temperatures: (pside - baseline_pside) / excursion_pside
    oside_norm : np.ndarray
        Normalized oside temperatures: (oside - baseline_oside) / excursion_pside
    pside_raw : np.ndarray
        Raw pside temperatures (for reference)
    """
    # Load data
    data = pd.read_csv(data_path)
    
    # Check required columns
    required_cols = ['time', 'temp', 'oside']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in {data_path}: {missing_cols}. "
                        f"Available columns: {list(data.columns)}")
    
    # Extract data
    times = data['time'].values
    pside_data = data['temp'].values
    oside_data = data['oside'].values
    
    # Remove NaN values
    valid_mask = ~(np.isnan(times) | np.isnan(pside_data) | np.isnan(oside_data))
    times = times[valid_mask]
    pside_data = pside_data[valid_mask]
    oside_data = oside_data[valid_mask]
    
    # Convert baseline window from microseconds to seconds
    baseline_window_s = baseline_window_us * 1e-6
    
    # Compute baselines
    baseline_pside = compute_baseline(times, pside_data, baseline_window_s)
    baseline_oside = compute_baseline(times, oside_data, baseline_window_s)
    
    # Compute pside excursion
    pside_excursion = (pside_data - baseline_pside).max() - (pside_data - baseline_pside).min()
    if pside_excursion <= 0.0:
        raise ValueError(f"Pside excursion is zero in {data_path} – check experimental data")
    
    # Normalize both pside and oside using pside excursion
    pside_norm = (pside_data - baseline_pside) / pside_excursion
    oside_norm = (oside_data - baseline_oside) / pside_excursion
    
    return times, pside_norm, oside_norm, pside_data


def plot_comparison(file_data: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]], 
                    output_path: Optional[str] = None):
    """Plot normalized pside and oside comparison for multiple files.
    
    Creates two subplots: one showing all pside curves, one showing all oside curves,
    so both can be compared side-by-side. Adds vertical lines showing the baseline
    averaging cutoff for each time series.
    
    Parameters:
    -----------
    file_data : List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]]
        List of (label, times, pside_norm, oside_norm, pside_raw, baseline_window_us) tuples
    output_path : str, optional
        Path to save the plot. If None, displays interactively.
    """
    n_files = len(file_data)
    
    # Create figure with two subplots (pside and oside)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Use distinct colors for each file
    colors = plt.cm.Dark2(np.linspace(0, 1, n_files))
    
    # Get y-axis limits after plotting to properly position vertical lines
    # First, plot all curves to establish axis limits
    for i, (label, times, pside_norm, oside_norm, pside_raw, baseline_window_us) in enumerate(file_data):
        times_us = times * 1e6
        ax1.plot(times_us, pside_norm, label=f"{label}", 
                color=colors[i], linestyle='-', linewidth=1.5, alpha=0.8)
        ax2.plot(times_us, oside_norm, label=f"{label}", 
                color=colors[i], linestyle='--', linewidth=1.5, alpha=0.8)
    
    # Get y-axis limits
    ylim1 = ax1.get_ylim()
    ylim2 = ax2.get_ylim()
    
    # Add vertical lines at baseline cutoff for each time series
    for i, (label, times, pside_norm, oside_norm, pside_raw, baseline_window_us) in enumerate(file_data):
        # Draw vertical line at baseline cutoff (colored to match the time series)
        ax1.axvline(x=baseline_window_us, color=colors[i], linestyle=':', 
                   linewidth=1.5, alpha=0.6)
        ax2.axvline(x=baseline_window_us, color=colors[i], linestyle=':', 
                   linewidth=1.5, alpha=0.6)
    
    # Restore original y-axis limits (in case vertical lines affected them)
    ax1.set_ylim(ylim1)
    ax2.set_ylim(ylim2)
    
    ax1.set_ylabel('Normalized Temperature', fontsize=12)
    ax1.set_title('Normalized Pside Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)
    
    ax2.set_xlabel('Time (μs)', fontsize=12)
    ax2.set_ylabel('Normalized Temperature', fontsize=12)
    ax2.set_title('Normalized Oside Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot normalized comparison between pside and oside for experimental data files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two files with different baseline windows:
  python plot_experimental_comparison.py file1.csv 1000 file2.csv 2000
  
  # Compare multiple files:
  python plot_experimental_comparison.py file1.csv 1000 file2.csv 2000 file3.csv 1500
  
  # Save to file:
  python plot_experimental_comparison.py file1.csv 1000 file2.csv 2000 -o comparison.png
  
  # With custom labels:
  python plot_experimental_comparison.py file1.csv 1000 file2.csv 2000 --labels "Run 1" "Run 2"
        """
    )
    
    parser.add_argument('file_window_pairs', nargs='+', 
                       help='Pairs of (file_path baseline_window_us) for each experimental data file. '
                            'Baseline window is in microseconds. Must provide an even number of arguments.')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output path for the plot (if not specified, displays interactively)')
    parser.add_argument('--labels', '-l', nargs='+', type=str, default=None,
                       help='Custom labels for each file (defaults to filenames)')
    
    args = parser.parse_args()
    
    # Parse file/window pairs
    n_args = len(args.file_window_pairs)
    if n_args % 2 != 0:
        parser.error(f"Must provide an even number of arguments (file/window pairs). Got {n_args} arguments.")
    
    if n_args < 4:
        parser.error(f"Must provide at least 2 file/window pairs (4 arguments minimum). Got {n_args} arguments.")
    
    # Parse into file paths and windows
    files = []
    windows = []
    for i in range(0, n_args, 2):
        file_path = args.file_window_pairs[i]
        try:
            window_us = float(args.file_window_pairs[i + 1])
        except ValueError:
            parser.error(f"Invalid baseline window value: '{args.file_window_pairs[i + 1]}'. Must be a number.")
        files.append(file_path)
        windows.append(window_us)
    
    n_files = len(files)
    
    if args.labels and len(args.labels) != n_files:
        parser.error(f"Number of labels ({len(args.labels)}) must match number of files ({n_files})")
    
    # Generate labels if not provided
    if args.labels is None:
        labels = [Path(f).stem for f in files]
    else:
        labels = args.labels
    
    # Load and normalize data for each file
    file_data = []
    print("Loading and normalizing data files...")
    print("=" * 60)
    
    for i, (file_path, baseline_window_us, label) in enumerate(zip(files, windows, labels)):
        if not Path(file_path).exists():
            parser.error(f"File not found: {file_path}")
        
        print(f"\n[{i+1}/{n_files}] Processing: {file_path}")
        print(f"  Label: {label}")
        print(f"  Baseline window: {baseline_window_us:.1f} μs")
        
        try:
            times, pside_norm, oside_norm, pside_raw = load_and_normalize_data(file_path, baseline_window_us)
            
            print(f"  Loaded {len(times)} data points")
            print(f"  Time range: {times.min()*1e6:.1f} – {times.max()*1e6:.1f} μs")
            print(f"  Pside normalized range: [{pside_norm.min():.4f}, {pside_norm.max():.4f}]")
            print(f"  Oside normalized range: [{oside_norm.min():.4f}, {oside_norm.max():.4f}]")
            
            file_data.append((label, times, pside_norm, oside_norm, pside_raw, baseline_window_us))
            
        except Exception as e:
            parser.error(f"Error processing {file_path}: {e}")
    
    print("\n" + "=" * 60)
    print("Creating comparison plot...")
    
    # Create the plot
    plot_comparison(file_data, output_path=args.output)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

