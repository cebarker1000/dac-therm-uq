#!/usr/bin/env python3
"""
FPCA Decomposition Plotting Tool

This script provides comprehensive plotting functionality for Functional Principal Component Analysis (FPCA) 
decompositions. It can plot mean functions, eigenfunctions, parameter correlations with weights, 
and various diagnostic plots.

Usage:
    python plot_fpca_decomposition.py --input fpca_model.npz --output-dir plots/
    python plot_fpca_decomposition.py --input uq_batch_results.npz --build-model --output-dir plots/
"""

import sys
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg
from typing import Dict, Any, List, Optional, Tuple
import warnings

# Import from local modules
from analysis.uq_wrapper import load_batch_results, build_fpca_model, load_fpca_model

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_fpca_decomposition(input_file: str) -> Dict[str, Any]:
    """
    Load FPCA decomposition from file.
    
    Parameters:
    -----------
    input_file : str
        Path to the FPCA model file (.npz)
        
    Returns:
    --------
    Dict[str, Any]
        FPCA decomposition dictionary
    """
    print(f"Loading FPCA decomposition from {input_file}...")
    
    if input_file.endswith('.npz'):
        # Try to load as FPCA model first
        try:
            fpca_model = load_fpca_model(input_file)
            print("Loaded as FPCA model")
            return fpca_model
        except:
            # Try to load as batch results and build FPCA model
            try:
                print("Loading as batch results and building FPCA model...")
                fpca_model = build_fpca_model(input_file)
                return fpca_model
            except Exception as e:
                print(f"Error loading file: {e}")
                raise
    else:
        raise ValueError("Input file must be a .npz file")


def plot_mean_function(fpca_model: Dict[str, Any], 
                      time_scale: float = 1e6,
                      time_unit: str = "μs",
                      figsize: Tuple[int, int] = (10, 6),
                      output_dir: str = "plots") -> None:
    """
    Plot the mean function from FPCA decomposition.
    
    Parameters:
    -----------
    fpca_model : Dict[str, Any]
        FPCA model dictionary
    time_scale : float
        Scaling factor for time axis (default: 1e6 for microseconds)
    time_unit : str
        Unit for time axis (default: "μs")
    figsize : Tuple[int, int]
        Figure size
    output_dir : str
        Output directory for plots
    """
    mean_curve = fpca_model['mean_curve']
    time_points = np.arange(len(mean_curve)) * time_scale
    
    plt.figure(figsize=figsize)
    plt.plot(time_points, mean_curve, 'b-', linewidth=2, label='Mean Function')
    plt.xlabel(f'Time ({time_unit})')
    plt.ylabel('Temperature (Normalized)')
    plt.title('FPCA Mean Function')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'fpca_mean_function.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()


def plot_eigenfunctions(fpca_model: Dict[str, Any], 
                       n_components: int = 5,
                       time_scale: float = 1e6,
                       time_unit: str = "μs",
                       figsize: Tuple[int, int] = (15, 10),
                       output_dir: str = "plots") -> None:
    """
    Plot eigenfunctions from FPCA decomposition.
    
    Parameters:
    -----------
    fpca_model : Dict[str, Any]
        FPCA model dictionary
    n_components : int
        Number of eigenfunctions to plot
    time_scale : float
        Scaling factor for time axis
    time_unit : str
        Unit for time axis
    figsize : Tuple[int, int]
        Figure size
    output_dir : str
        Output directory for plots
    """
    eigenfunctions = fpca_model['eigenfunctions']
    eigenvalues = fpca_model['eigenvalues']
    explained_variance = fpca_model['explained_variance']
    
    n_plot = min(n_components, eigenfunctions.shape[1])
    time_points = np.arange(eigenfunctions.shape[0]) * time_scale
    
    # Create subplot grid
    n_cols = 2
    n_rows = (n_plot + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each eigenfunction
    for i in range(n_plot):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        eigenfunction = eigenfunctions[:, i]
        eigenvalue = eigenvalues[i]
        var_explained = explained_variance[i]
        
        ax.plot(time_points, eigenfunction, linewidth=2, 
               label=f'λ={eigenvalue:.3f}, {var_explained:.1%}')
        ax.set_title(f'Eigenfunction {i+1}')
        ax.set_xlabel(f'Time ({time_unit})')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Hide empty subplots
    for i in range(n_plot, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'fpca_eigenfunctions.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()


def plot_explained_variance(fpca_model: Dict[str, Any],
                           n_components: int = 10,
                           figsize: Tuple[int, int] = (12, 8),
                           output_dir: str = "plots") -> None:
    """
    Plot explained variance and cumulative variance.
    
    Parameters:
    -----------
    fpca_model : Dict[str, Any]
        FPCA model dictionary
    n_components : int
        Number of components to plot
    figsize : Tuple[int, int]
        Figure size
    output_dir : str
        Output directory for plots
    """
    explained_variance = fpca_model['explained_variance']
    cumulative_variance = fpca_model['cumulative_variance']
    eigenvalues = fpca_model['eigenvalues']
    
    n_plot = min(n_components, len(explained_variance))
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Individual explained variance
    axes[0, 0].bar(range(1, n_plot + 1), explained_variance[:n_plot], 
                   color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Explained Variance by Component')
    axes[0, 0].set_xlabel('Principal Component')
    axes[0, 0].set_ylabel('Explained Variance')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Cumulative explained variance
    axes[0, 1].plot(range(1, n_plot + 1), cumulative_variance[:n_plot], 
                    'bo-', linewidth=2, markersize=6)
    axes[0, 1].axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% threshold')
    axes[0, 1].axhline(y=0.99, color='orange', linestyle='--', alpha=0.7, label='99% threshold')
    axes[0, 1].set_title('Cumulative Explained Variance')
    axes[0, 1].set_xlabel('Number of Components')
    axes[0, 1].set_ylabel('Cumulative Explained Variance')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Eigenvalues (scree plot)
    axes[1, 0].semilogy(range(1, n_plot + 1), eigenvalues[:n_plot], 'ro-', linewidth=2)
    axes[1, 0].set_title('Eigenvalues (Scree Plot)')
    axes[1, 0].set_xlabel('Principal Component')
    axes[1, 0].set_ylabel('Eigenvalue (log scale)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Log-log plot of eigenvalues
    axes[1, 1].loglog(range(1, n_plot + 1), eigenvalues[:n_plot], 'go-', linewidth=2)
    axes[1, 1].set_title('Eigenvalues (Log-Log Scale)')
    axes[1, 1].set_xlabel('Principal Component')
    axes[1, 1].set_ylabel('Eigenvalue')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'fpca_explained_variance.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()


def compute_parameter_correlations(fpca_model: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
    """
    Compute correlations between FPCA scores and parameters.
    
    Parameters:
    -----------
    fpca_model : Dict[str, Any]
        FPCA model dictionary
        
    Returns:
    --------
    Tuple[np.ndarray, List[str]]
        Correlation matrix and parameter names
    """
    if 'training_scores' not in fpca_model or 'training_parameters' not in fpca_model:
        raise ValueError("FPCA model must contain training_scores and training_parameters")
    
    scores = fpca_model['training_scores']
    parameters = fpca_model['training_parameters']
    param_names = fpca_model['parameter_names']
    
    n_components = scores.shape[1]
    n_params = parameters.shape[1]
    
    # Compute correlations
    correlations = np.zeros((n_components, n_params))
    for i in range(n_components):
        for j in range(n_params):
            corr = np.corrcoef(scores[:, i], parameters[:, j])[0, 1]
            correlations[i, j] = corr
    
    return correlations, param_names


def plot_parameter_correlations(fpca_model: Dict[str, Any],
                               n_components: int = 5,
                               correlation_threshold: float = 0.3,
                               figsize: Tuple[int, int] = (15, 12),
                               output_dir: str = "plots") -> np.ndarray:
    """
    Plot parameter correlations with FPCA components.
    
    Parameters:
    -----------
    fpca_model : Dict[str, Any]
        FPCA model dictionary
    n_components : int
        Number of components to analyze
    correlation_threshold : float
        Threshold for highlighting significant correlations
    figsize : Tuple[int, int]
        Figure size
    output_dir : str
        Output directory for plots
        
    Returns:
    --------
    np.ndarray
        Correlation matrix
    """
    correlations, param_names = compute_parameter_correlations(fpca_model)
    
    n_plot = min(n_components, correlations.shape[0])
    
    # Create subplot grid
    n_cols = 2
    n_rows = (n_plot + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot correlations for each component
    for i in range(n_plot):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        corr_values = correlations[i, :]
        
        # Create bar plot
        bars = ax.bar(range(len(param_names)), corr_values, 
                     color='skyblue', alpha=0.7, edgecolor='navy', linewidth=0.5)
        
        # Color bars based on correlation strength
        for j, (bar, corr) in enumerate(zip(bars, corr_values)):
            if abs(corr) > correlation_threshold:
                bar.set_color('red' if corr > 0 else 'blue')
                bar.set_alpha(0.8)
        
        ax.set_title(f'PC{i+1} Parameter Correlations')
        ax.set_xlabel('Parameters')
        ax.set_ylabel('Pearson Correlation')
        ax.set_xticks(range(len(param_names)))
        ax.set_xticklabels(param_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.axhline(y=correlation_threshold, color='red', linestyle='--', alpha=0.5, 
                   label=f'Threshold ±{correlation_threshold}')
        ax.axhline(y=-correlation_threshold, color='red', linestyle='--', alpha=0.5)
        ax.legend()
        
        # Add correlation values on bars
        for j, corr in enumerate(corr_values):
            if abs(corr) > 0.1:  # Only show significant correlations
                ax.text(j, corr + (0.02 if corr >= 0 else -0.02), 
                       f'{corr:.3f}', ha='center', 
                       va='bottom' if corr >= 0 else 'top',
                       fontsize=8, fontweight='bold')
    
    # Hide empty subplots
    for i in range(n_plot, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'fpca_parameter_correlations.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return correlations


def plot_correlation_heatmap(fpca_model: Dict[str, Any],
                            n_components: int = 5,
                            figsize: Tuple[int, int] = (12, 8),
                            output_dir: str = "plots") -> np.ndarray:
    """
    Plot correlation heatmap between FPCA components and parameters.
    
    Parameters:
    -----------
    fpca_model : Dict[str, Any]
        FPCA model dictionary
    n_components : int
        Number of components to analyze
    figsize : Tuple[int, int]
        Figure size
    output_dir : str
        Output directory for plots
        
    Returns:
    --------
    np.ndarray
        Correlation matrix
    """
    correlations, param_names = compute_parameter_correlations(fpca_model)
    
    n_plot = min(n_components, correlations.shape[0])
    corr_subset = correlations[:n_plot, :]
    
    # Create heatmap
    plt.figure(figsize=figsize)
    
    # Create mask for significant correlations
    mask = np.abs(corr_subset) < 0.1  # Mask weak correlations
    
    sns.heatmap(corr_subset, 
                xticklabels=param_names,
                yticklabels=[f'PC{i+1}' for i in range(n_plot)],
                annot=True, 
                fmt='.3f',
                cmap='RdBu_r',
                center=0,
                mask=mask,
                cbar_kws={'label': 'Pearson Correlation'})
    
    plt.title('FPCA Component - Parameter Correlations')
    plt.xlabel('Parameters')
    plt.ylabel('Principal Components')
    plt.xticks(rotation=45, ha='right')
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'fpca_correlation_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return corr_subset


def plot_score_distributions(fpca_model: Dict[str, Any],
                            n_components: int = 4,
                            figsize: Tuple[int, int] = (15, 10),
                            output_dir: str = "plots") -> None:
    """
    Plot distributions of FPCA scores.
    
    Parameters:
    -----------
    fpca_model : Dict[str, Any]
        FPCA model dictionary
    n_components : int
        Number of components to plot
    figsize : Tuple[int, int]
        Figure size
    output_dir : str
        Output directory for plots
    """
    if 'training_scores' not in fpca_model:
        raise ValueError("FPCA model must contain training_scores")
    
    scores = fpca_model['training_scores']
    eigenvalues = fpca_model['eigenvalues']
    
    n_plot = min(n_components, scores.shape[1])
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Score distributions
    for i in range(min(n_plot, 3)):
        axes[0, 0].hist(scores[:, i], bins=30, alpha=0.7, 
                       label=f'PC{i+1} (σ={np.std(scores[:, i]):.3f})')
    
    axes[0, 0].set_title('Distribution of Principal Component Scores')
    axes[0, 0].set_xlabel('Score Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Score scatter plots
    if n_plot >= 2:
        scatter = axes[0, 1].scatter(scores[:, 0], scores[:, 1], alpha=0.6)
        axes[0, 1].set_xlabel('PC1 Score')
        axes[0, 1].set_ylabel('PC2 Score')
        axes[0, 1].set_title('PC1 vs PC2 Scores')
        axes[0, 1].grid(True, alpha=0.3)
    
    if n_plot >= 3:
        axes[1, 0].scatter(scores[:, 0], scores[:, 2], alpha=0.6)
        axes[1, 0].set_xlabel('PC1 Score')
        axes[1, 0].set_ylabel('PC3 Score')
        axes[1, 0].set_title('PC1 vs PC3 Scores')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].scatter(scores[:, 1], scores[:, 2], alpha=0.6)
        axes[1, 1].set_xlabel('PC2 Score')
        axes[1, 1].set_ylabel('PC3 Score')
        axes[1, 1].set_title('PC2 vs PC3 Scores')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'fpca_score_distributions.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()


def plot_comprehensive_summary(fpca_model: Dict[str, Any],
                             n_components: int = 4,
                             time_scale: float = 1e6,
                             time_unit: str = "μs",
                             figsize: Tuple[int, int] = (20, 16),
                             output_dir: str = "plots") -> None:
    """
    Create a comprehensive summary plot of FPCA decomposition.
    
    Parameters:
    -----------
    fpca_model : Dict[str, Any]
        FPCA model dictionary
    n_components : int
        Number of components to analyze
    time_scale : float
        Scaling factor for time axis
    time_unit : str
        Unit for time axis
    figsize : Tuple[int, int]
        Figure size
    output_dir : str
        Output directory for plots
    """
    n_plot = min(n_components, fpca_model['eigenfunctions'].shape[1])
    
    # Create large figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Mean function (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    mean_curve = fpca_model['mean_curve']
    time_points = np.arange(len(mean_curve)) * time_scale
    ax1.plot(time_points, mean_curve, 'b-', linewidth=2)
    ax1.set_title('Mean Function')
    ax1.set_xlabel(f'Time ({time_unit})')
    ax1.set_ylabel('Temperature')
    ax1.grid(True, alpha=0.3)
    
    # 2. Explained variance (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    explained_variance = fpca_model['explained_variance'][:n_plot]
    ax2.bar(range(1, n_plot + 1), explained_variance, color='skyblue', alpha=0.7)
    ax2.set_title('Explained Variance')
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Variance')
    ax2.grid(True, alpha=0.3)
    
    # 3. Eigenfunctions (middle row)
    eigenfunctions = fpca_model['eigenfunctions']
    eigenvalues = fpca_model['eigenvalues']
    time_points = np.arange(eigenfunctions.shape[0]) * time_scale
    
    for i in range(min(n_plot, 4)):
        ax = fig.add_subplot(gs[1, i])
        eigenfunction = eigenfunctions[:, i]
        eigenvalue = eigenvalues[i]
        var_explained = fpca_model['explained_variance'][i]
        
        ax.plot(time_points, eigenfunction, linewidth=2, 
               label=f'λ={eigenvalue:.3f}')
        ax.set_title(f'Eigenfunction {i+1}\n({var_explained:.1%} variance)')
        ax.set_xlabel(f'Time ({time_unit})')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # 4. Parameter correlations (bottom row)
    try:
        correlations, param_names = compute_parameter_correlations(fpca_model)
        
        for i in range(min(n_plot, 4)):
            ax = fig.add_subplot(gs[2, i])
            corr_values = correlations[i, :]
            
            # Create bar plot
            bars = ax.bar(range(len(param_names)), corr_values, 
                         color='skyblue', alpha=0.7)
            
            # Color significant correlations
            for j, (bar, corr) in enumerate(zip(bars, corr_values)):
                if abs(corr) > 0.3:
                    bar.set_color('red' if corr > 0 else 'blue')
                    bar.set_alpha(0.8)
            
            ax.set_title(f'PC{i+1} Parameter Correlations')
            ax.set_xlabel('Parameters')
            ax.set_ylabel('Correlation')
            ax.set_xticks(range(len(param_names)))
            ax.set_xticklabels(param_names, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
    except Exception as e:
        print(f"Warning: Could not plot parameter correlations: {e}")
        # Add placeholder text
        ax = fig.add_subplot(gs[2, 0])
        ax.text(0.5, 0.5, 'Parameter correlations\nnot available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Parameter Correlations')
    
    plt.suptitle('FPCA Decomposition Summary', fontsize=16, fontweight='bold')
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'fpca_comprehensive_summary.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()


def generate_fpca_report(fpca_model: Dict[str, Any], 
                        correlations: Optional[np.ndarray] = None,
                        output_dir: str = "plots") -> None:
    """
    Generate a comprehensive text report of FPCA analysis.
    
    Parameters:
    -----------
    fpca_model : Dict[str, Any]
        FPCA model dictionary
    correlations : Optional[np.ndarray]
        Parameter correlation matrix
    output_dir : str
        Output directory for report
    """
    report = []
    report.append("=" * 60)
    report.append("FPCA DECOMPOSITION ANALYSIS REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Basic information
    report.append("BASIC INFORMATION:")
    if 'training_curves' in fpca_model:
        report.append(f"Number of training curves: {len(fpca_model['training_curves'])}")
        report.append(f"Curve length: {fpca_model['training_curves'].shape[1]} time steps")
    report.append(f"Number of components: {fpca_model['n_components']}")
    report.append("")
    
    # Explained variance
    report.append("EXPLAINED VARIANCE:")
    for i in range(min(10, len(fpca_model['explained_variance']))):
        cum_var = fpca_model['cumulative_variance'][i]
        report.append(f"PC{i+1}: {fpca_model['explained_variance'][i]:.4f} "
                     f"(cumulative: {cum_var:.4f})")
    report.append("")
    
    # Components needed for different thresholds
    thresholds = [0.8, 0.9, 0.95, 0.99]
    report.append("COMPONENTS NEEDED FOR VARIANCE THRESHOLDS:")
    for threshold in thresholds:
        n_comp = np.argmax(fpca_model['cumulative_variance'] >= threshold) + 1
        report.append(f"{threshold*100}% variance: {n_comp} components")
    report.append("")
    
    # Score statistics
    if 'training_scores' in fpca_model:
        report.append("PRINCIPAL COMPONENT SCORE STATISTICS:")
        scores = fpca_model['training_scores']
        for i in range(min(5, scores.shape[1])):
            score_std = np.std(scores[:, i])
            score_range = np.max(scores[:, i]) - np.min(scores[:, i])
            report.append(f"PC{i+1}: std={score_std:.4f}, range={score_range:.4f}")
        report.append("")
    
    # Parameter correlations
    if correlations is not None:
        report.append("PARAMETER CORRELATIONS:")
        param_names = fpca_model.get('parameter_names', [f'Param_{i}' for i in range(correlations.shape[1])])
        
        for i in range(min(5, correlations.shape[0])):
            report.append(f"PC{i+1} correlations:")
            # Sort by absolute correlation
            sorted_corrs = sorted(enumerate(correlations[i, :]), 
                                key=lambda x: abs(x[1]), reverse=True)
            for j, (param_idx, corr) in enumerate(sorted_corrs[:5]):
                report.append(f"  {param_names[param_idx]:15s}: {corr:+.3f}")
            report.append("")
    
    # Save report
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'fpca_analysis_report.txt'), 'w') as f:
        f.write('\n'.join(report))
    
    # Print to console
    print('\n'.join(report))


def plot_mean_with_eigenfunctions(fpca_model: Dict[str, Any], 
                                 n_components: int = 4,
                                 time_scale: float = 1e6,
                                 time_unit: str = "μs",
                                 figsize: Tuple[int, int] = (15, 10),
                                 output_dir: str = "plots") -> None:
    """
    Plot the mean curve with eigenfunctions overlaid.
    
    Parameters:
    -----------
    fpca_model : Dict[str, Any]
        FPCA model dictionary
    n_components : int
        Number of eigenfunctions to overlay
    time_scale : float
        Scaling factor for time axis
    time_unit : str
        Unit for time axis
    figsize : Tuple[int, int]
        Figure size
    output_dir : str
        Output directory for plots
    """
    mean_curve = fpca_model['mean_curve']
    eigenfunctions = fpca_model['eigenfunctions']
    eigenvalues = fpca_model['eigenvalues']
    explained_variance = fpca_model['explained_variance']
    
    n_plot = min(n_components, eigenfunctions.shape[1])
    time_points = np.arange(len(mean_curve)) * time_scale
    
    # Create subplot grid
    n_cols = 2
    n_rows = (n_plot + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Colors for eigenfunctions
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Plot each eigenfunction overlaid on mean
    for i in range(n_plot):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        eigenfunction = eigenfunctions[:, i]
        eigenvalue = eigenvalues[i]
        var_explained = explained_variance[i]
        
        # Plot mean curve
        ax.plot(time_points, mean_curve, 'k-', linewidth=2, label='Mean Function', alpha=0.7)
        
        # Plot eigenfunction overlaid
        color = colors[i % len(colors)]
        ax.plot(time_points, mean_curve + eigenfunction, color=color, linewidth=2, 
               linestyle='--', label=f'PC{i+1} (+λ={eigenvalue:.3f})')
        ax.plot(time_points, mean_curve - eigenfunction, color=color, linewidth=2, 
               linestyle=':', label=f'PC{i+1} (-λ={eigenvalue:.3f})')
        
        ax.set_title(f'Mean + PC{i+1} ({var_explained:.1%} variance)')
        ax.set_xlabel(f'Time ({time_unit})')
        ax.set_ylabel('Temperature (Normalized)')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Hide empty subplots
    for i in range(n_plot, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'fpca_mean_with_eigenfunctions.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()


def plot_mean_with_eigenfunctions_combined(fpca_model: Dict[str, Any], 
                                         n_components: int = 4,
                                         time_scale: float = 1e6,
                                         time_unit: str = "μs",
                                         figsize: Tuple[int, int] = (12, 8),
                                         output_dir: str = "plots") -> None:
    """
    Plot the mean curve with all eigenfunctions overlaid in a single plot.
    
    Parameters:
    -----------
    fpca_model : Dict[str, Any]
        FPCA model dictionary
    n_components : int
        Number of eigenfunctions to overlay
    time_scale : float
        Scaling factor for time axis
    time_unit : str
        Unit for time axis
    figsize : Tuple[int, int]
        Figure size
    output_dir : str
        Output directory for plots
    """
    mean_curve = fpca_model['mean_curve']
    eigenfunctions = fpca_model['eigenfunctions']
    eigenvalues = fpca_model['eigenvalues']
    explained_variance = fpca_model['explained_variance']
    
    n_plot = min(n_components, eigenfunctions.shape[1])
    time_points = np.arange(len(mean_curve)) * time_scale
    
    # Create single plot
    plt.figure(figsize=figsize)
    
    # Colors for eigenfunctions
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Plot mean curve
    plt.plot(time_points, mean_curve, 'k-', linewidth=3, label='Mean Function', alpha=0.8)
    
    # Plot each eigenfunction overlaid
    for i in range(n_plot):
        eigenfunction = eigenfunctions[:, i]
        eigenvalue = eigenvalues[i]
        var_explained = explained_variance[i]
        color = colors[i % len(colors)]
        
        # Plot positive and negative variations
        plt.plot(time_points, mean_curve + eigenfunction, color=color, linewidth=2, 
               linestyle='--', alpha=0.7, 
               label=f'PC{i+1} (+λ={eigenvalue:.3f}, {var_explained:.1%})')
        plt.plot(time_points, mean_curve - eigenfunction, color=color, linewidth=2, 
               linestyle=':', alpha=0.7)
    
    plt.title('Mean Function with Principal Component Variations')
    plt.xlabel(f'Time ({time_unit})')
    plt.ylabel('Temperature (Normalized)')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'fpca_mean_with_eigenfunctions_combined.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()


def plot_mean_with_eigenfunctions_scaled(fpca_model: Dict[str, Any], 
                                       n_components: int = 4,
                                       time_scale: float = 1e6,
                                       time_unit: str = "μs",
                                       figsize: Tuple[int, int] = (15, 10),
                                       output_dir: str = "plots") -> None:
    """
    Plot the mean curve with eigenfunctions overlaid, scaled by their eigenvalues.
    
    Parameters:
    -----------
    fpca_model : Dict[str, Any]
        FPCA model dictionary
    n_components : int
        Number of eigenfunctions to overlay
    time_scale : float
        Scaling factor for time axis
    time_unit : str
        Unit for time axis
    figsize : Tuple[int, int]
        Figure size
    output_dir : str
        Output directory for plots
    """
    mean_curve = fpca_model['mean_curve']
    eigenfunctions = fpca_model['eigenfunctions']
    eigenvalues = fpca_model['eigenvalues']
    explained_variance = fpca_model['explained_variance']
    
    n_plot = min(n_components, eigenfunctions.shape[1])
    time_points = np.arange(len(mean_curve)) * time_scale
    
    # Create subplot grid
    n_cols = 2
    n_rows = (n_plot + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Colors for eigenfunctions
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Plot each eigenfunction overlaid on mean (scaled by eigenvalue)
    for i in range(n_plot):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        eigenfunction = eigenfunctions[:, i]
        eigenvalue = eigenvalues[i]
        var_explained = explained_variance[i]
        
        # Scale eigenfunction by eigenvalue
        scaled_eigenfunction = eigenfunction * np.sqrt(eigenvalue)
        
        # Plot mean curve
        ax.plot(time_points, mean_curve, 'k-', linewidth=2, label='Mean Function', alpha=0.7)
        
        # Plot scaled eigenfunction overlaid
        color = colors[i % len(colors)]
        ax.plot(time_points, mean_curve + scaled_eigenfunction, color=color, linewidth=2, 
               linestyle='--', label=f'PC{i+1} (+scaled)')
        ax.plot(time_points, mean_curve - scaled_eigenfunction, color=color, linewidth=2, 
               linestyle=':', label=f'PC{i+1} (-scaled)')
        
        ax.set_title(f'Mean + PC{i+1} (scaled, {var_explained:.1%} variance)')
        ax.set_xlabel(f'Time ({time_unit})')
        ax.set_ylabel('Temperature (Normalized)')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Hide empty subplots
    for i in range(n_plot, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'fpca_mean_with_eigenfunctions_scaled.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """
    Main function to run FPCA decomposition plotting.
    """
    parser = argparse.ArgumentParser(
        description='Plot FPCA decomposition components',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot from existing FPCA model
  python plot_fpca_decomposition.py --input fpca_model.npz --output-dir plots/
  
  # Build FPCA model from batch results and plot
  python plot_fpca_decomposition.py --input uq_batch_results.npz --build-model --output-dir plots/
  
  # Plot with custom number of components
  python plot_fpca_decomposition.py --input fpca_model.npz --components 6 --output-dir plots/
        """
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='Path to the input file (.npz)')
    parser.add_argument('--output-dir', type=str, default='plots',
                       help='Output directory for plots (default: plots)')
    parser.add_argument('--components', type=int, default=4,
                       help='Number of components to analyze (default: 4)')
    parser.add_argument('--build-model', action='store_true',
                       help='Build FPCA model from batch results')
    parser.add_argument('--time-scale', type=float, default=1e6,
                       help='Time scale factor (default: 1e6 for microseconds)')
    parser.add_argument('--time-unit', type=str, default='μs',
                       help='Time unit (default: μs)')
    parser.add_argument('--correlation-threshold', type=float, default=0.3,
                       help='Correlation threshold for highlighting (default: 0.3)')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Generate comprehensive summary plot')
    
    args = parser.parse_args()
    
    print(f"Starting FPCA decomposition plotting...")
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of components: {args.components}")
    
    # Load or build FPCA model
    if args.build_model:
        print("Building FPCA model from batch results...")
        fpca_model = build_fpca_model(args.input)
    else:
        fpca_model = load_fpca_decomposition(args.input)
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Basic plots
    plot_mean_function(fpca_model, time_scale=args.time_scale, 
                      time_unit=args.time_unit, output_dir=args.output_dir)
    
    plot_eigenfunctions(fpca_model, n_components=args.components,
                       time_scale=args.time_scale, time_unit=args.time_unit,
                       output_dir=args.output_dir)
    
    plot_explained_variance(fpca_model, n_components=args.components,
                           output_dir=args.output_dir)
    
    # Mean with eigenfunctions plots
    plot_mean_with_eigenfunctions(fpca_model, n_components=args.components,
                                 time_scale=args.time_scale, time_unit=args.time_unit,
                                 output_dir=args.output_dir)
    
    plot_mean_with_eigenfunctions_combined(fpca_model, n_components=args.components,
                                         time_scale=args.time_scale, time_unit=args.time_unit,
                                         output_dir=args.output_dir)
    
    plot_mean_with_eigenfunctions_scaled(fpca_model, n_components=args.components,
                                       time_scale=args.time_scale, time_unit=args.time_unit,
                                       output_dir=args.output_dir)
    
    # Parameter correlation plots
    try:
        correlations = plot_parameter_correlations(fpca_model, 
                                                n_components=args.components,
                                                correlation_threshold=args.correlation_threshold,
                                                output_dir=args.output_dir)
        
        plot_correlation_heatmap(fpca_model, n_components=args.components,
                               output_dir=args.output_dir)
    except Exception as e:
        print(f"Warning: Could not generate parameter correlation plots: {e}")
        correlations = None
    
    # Score distribution plots
    try:
        plot_score_distributions(fpca_model, n_components=args.components,
                               output_dir=args.output_dir)
    except Exception as e:
        print(f"Warning: Could not generate score distribution plots: {e}")
    
    # Comprehensive summary
    if args.comprehensive:
        plot_comprehensive_summary(fpca_model, n_components=args.components,
                                time_scale=args.time_scale, time_unit=args.time_unit,
                                output_dir=args.output_dir)
    
    # Generate report
    print("\nGenerating FPCA analysis report...")
    generate_fpca_report(fpca_model, correlations, output_dir=args.output_dir)
    
    print(f"\nFPCA decomposition plotting completed! Check the '{args.output_dir}' directory for:")
    print("- fpca_mean_function.png: Mean function plot")
    print("- fpca_eigenfunctions.png: Eigenfunction plots")
    print("- fpca_explained_variance.png: Variance analysis")
    print("- fpca_mean_with_eigenfunctions.png: Mean with eigenfunctions (individual)")
    print("- fpca_mean_with_eigenfunctions_combined.png: Mean with eigenfunctions (combined)")
    print("- fpca_mean_with_eigenfunctions_scaled.png: Mean with scaled eigenfunctions")
    print("- fpca_parameter_correlations.png: Parameter correlation plots")
    print("- fpca_correlation_heatmap.png: Correlation heatmap")
    print("- fpca_score_distributions.png: Score distribution plots")
    if args.comprehensive:
        print("- fpca_comprehensive_summary.png: Comprehensive summary")
    print("- fpca_analysis_report.txt: Text report")


if __name__ == "__main__":
    main() 