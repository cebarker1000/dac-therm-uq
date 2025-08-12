import sys
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from analysis.uq_wrapper import load_batch_results
from scipy import linalg
import warnings

import sys

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data(file_path="outputs/uq_batch_results.npz"):
    """
    Load the batch results and prepare data for FPCA.
    """
    print("Loading batch results...")
    data = load_batch_results(file_path)
    
    # Filter out failed simulations (those with NaN values)
    valid_mask = ~np.isnan(data['oside_curves']).any(axis=1)
    valid_curves = data['oside_curves'][valid_mask]
    valid_params = data['parameters'][valid_mask]
    
    print(f"Valid curves: {len(valid_curves)} out of {len(data['oside_curves'])}")
    print(f"Curve length: {valid_curves.shape[1]} time steps")
    
    return valid_curves, valid_params, data['parameter_names']

def compute_fpca(curves, n_components=None, center=True):
    """
    Compute Functional Principal Component Analysis.
    
    Parameters:
    -----------
    curves : np.ndarray
        2D array where each row is a curve (n_curves x n_timepoints)
    n_components : int, optional
        Number of components to compute. If None, compute all.
    center : bool
        Whether to center the data (remove mean)
        
    Returns:
    --------
    dict
        Dictionary containing FPCA results
    """
    print("Computing FPCA...")
    
    # Center the data if requested
    if center:
        mean_curve = np.mean(curves, axis=0)
        curves_centered = curves - mean_curve
    else:
        mean_curve = np.zeros(curves.shape[1])
        curves_centered = curves
    
    # Compute covariance matrix
    n_curves, n_timepoints = curves_centered.shape
    cov_matrix = np.cov(curves_centered.T)
    
    print(f"Covariance matrix shape: {cov_matrix.shape}")
    
    # Compute eigendecomposition
    eigenvalues, eigenfunctions = linalg.eigh(cov_matrix)
    
    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenfunctions = eigenfunctions[:, idx]
    
    # Normalize eigenfunctions
    eigenfunctions = eigenfunctions / np.sqrt(np.sum(eigenfunctions**2, axis=0))
    
    # Compute explained variance
    explained_variance = eigenvalues / np.sum(eigenvalues)
    cumulative_variance = np.cumsum(explained_variance)
    
    # Limit number of components if specified
    if n_components is not None:
        n_components = min(n_components, len(eigenvalues))
        eigenvalues = eigenvalues[:n_components]
        eigenfunctions = eigenfunctions[:, :n_components]
        explained_variance = explained_variance[:n_components]
        cumulative_variance = cumulative_variance[:n_components]
    
    # Compute scores (projections of curves onto eigenfunctions)
    scores = curves_centered @ eigenfunctions
    
    results = {
        'mean_curve': mean_curve,
        'eigenvalues': eigenvalues,
        'eigenfunctions': eigenfunctions,
        'explained_variance': explained_variance,
        'cumulative_variance': cumulative_variance,
        'scores': scores,
        'n_components': len(eigenvalues)
    }
    
    print(f"FPCA completed with {len(eigenvalues)} components")
    print(f"Explained variance: {explained_variance[:5]}")
    
    return results

def reconstruct_curves(fpca_results, n_components=None):
    """
    Reconstruct curves using specified number of components.
    
    Parameters:
    -----------
    fpca_results : dict
        Results from compute_fpca
    n_components : int, optional
        Number of components to use for reconstruction
        
    Returns:
    --------
    np.ndarray
        Reconstructed curves
    """
    if n_components is None:
        n_components = fpca_results['n_components']
    
    n_components = min(n_components, fpca_results['n_components'])
    
    # Use only the first n_components
    eigenfunctions = fpca_results['eigenfunctions'][:, :n_components]
    scores = fpca_results['scores'][:, :n_components]
    mean_curve = fpca_results['mean_curve']
    
    # Reconstruct
    reconstructed = scores @ eigenfunctions.T + mean_curve
    
    return reconstructed

def compute_reconstruction_error(original, reconstructed):
    """
    Compute reconstruction error metrics.
    
    Parameters:
    -----------
    original : np.ndarray
        Original curves
    reconstructed : np.ndarray
        Reconstructed curves
        
    Returns:
    --------
    dict
        Dictionary with error metrics
    """
    # Mean squared error
    mse = np.mean((original - reconstructed)**2)
    
    # Root mean squared error
    rmse = np.sqrt(mse)
    
    # Mean absolute error
    mae = np.mean(np.abs(original - reconstructed))
    
    # R-squared (coefficient of determination)
    ss_res = np.sum((original - reconstructed)**2)
    ss_tot = np.sum((original - np.mean(original))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Per-curve errors
    curve_errors = np.sqrt(np.mean((original - reconstructed)**2, axis=1))
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r_squared': r_squared,
        'curve_errors': curve_errors
    }

def plot_eigenfunctions(fpca_results, n_components=3, figsize=(15, 10), output_dir="outputs"):
    """
    Plot the first n eigenfunctions.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    eigenfunctions = fpca_results['eigenfunctions']
    eigenvalues = fpca_results['eigenvalues']
    explained_variance = fpca_results['explained_variance']
    time_points = np.arange(eigenfunctions.shape[0])
    
    # Plot eigenfunctions
    for i in range(min(n_components, 3)):
        axes[0, 0].plot(time_points, eigenfunctions[:, i], 
                       label=f'PC{i+1} (λ={eigenvalues[i]:.3f})', linewidth=2)
    
    axes[0, 0].set_title('First Three Eigenfunctions')
    axes[0, 0].set_xlabel('Time Steps')
    axes[0, 0].set_ylabel('Eigenfunction Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot explained variance
    n_plot = min(10, len(explained_variance))
    axes[0, 1].bar(range(1, n_plot + 1), explained_variance[:n_plot])
    axes[0, 1].set_title('Explained Variance by Component')
    axes[0, 1].set_xlabel('Principal Component')
    axes[0, 1].set_ylabel('Explained Variance')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot cumulative explained variance
    cumulative_variance = fpca_results['cumulative_variance']
    axes[1, 0].plot(range(1, len(cumulative_variance) + 1), 
                   cumulative_variance, 'bo-')
    axes[1, 0].axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% threshold')
    axes[1, 0].axhline(y=0.99, color='orange', linestyle='--', alpha=0.7, label='99% threshold')
    axes[1, 0].set_title('Cumulative Explained Variance')
    axes[1, 0].set_xlabel('Number of Components')
    axes[1, 0].set_ylabel('Cumulative Explained Variance')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot eigenvalues (scree plot)
    n_plot = min(20, len(eigenvalues))
    axes[1, 1].semilogy(range(1, n_plot + 1), eigenvalues[:n_plot], 'ro-')
    axes[1, 1].set_title('Eigenvalues (Scree Plot)')
    axes[1, 1].set_xlabel('Principal Component')
    axes[1, 1].set_ylabel('Eigenvalue (log scale)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fpca_eigenfunctions.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_reconstruction_examples(original_curves, fpca_results, n_examples=6, figsize=(15, 12), output_dir="outputs"):
    """
    Plot examples of original vs reconstructed curves.
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    # Select random examples
    n_curves = len(original_curves)
    example_indices = np.random.choice(n_curves, n_examples, replace=False)
    
    time_points = np.arange(original_curves.shape[1])
    
    for i, idx in enumerate(example_indices):
        # Original curve
        axes[i].plot(time_points, original_curves[idx], 'b-', linewidth=2, label='Original')
        
        # Reconstructed with different numbers of components
        for n_comp in [1, 3, 5]:
            reconstructed = reconstruct_curves(fpca_results, n_comp)
            axes[i].plot(time_points, reconstructed[idx], '--', linewidth=1, 
                        alpha=0.7, label=f'{n_comp} PCs')
        
        axes[i].set_title(f'Curve {idx}')
        axes[i].set_xlabel('Time Steps')
        axes[i].set_ylabel('Normalized Temperature')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fpca_reconstruction_examples.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_reconstruction_errors(original_curves, fpca_results, max_components=10, figsize=(12, 8), output_dir="outputs"):
    """
    Plot reconstruction errors as a function of number of components.
    """
    n_components_range = range(1, min(max_components + 1, fpca_results['n_components'] + 1))
    
    errors = []
    for n_comp in n_components_range:
        reconstructed = reconstruct_curves(fpca_results, n_comp)
        error_metrics = compute_reconstruction_error(original_curves, reconstructed)
        errors.append(error_metrics)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # RMSE
    rmse_values = [e['rmse'] for e in errors]
    axes[0, 0].plot(n_components_range, rmse_values, 'bo-')
    axes[0, 0].set_title('Root Mean Square Error')
    axes[0, 0].set_xlabel('Number of Components')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE
    mae_values = [e['mae'] for e in errors]
    axes[0, 1].plot(n_components_range, mae_values, 'ro-')
    axes[0, 1].set_title('Mean Absolute Error')
    axes[0, 1].set_xlabel('Number of Components')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].grid(True, alpha=0.3)
    
    # R-squared
    r_squared_values = [e['r_squared'] for e in errors]
    axes[1, 0].plot(n_components_range, r_squared_values, 'go-')
    axes[1, 0].set_title('R-squared')
    axes[1, 0].set_xlabel('Number of Components')
    axes[1, 0].set_ylabel('R²')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Explained variance
    explained_var = fpca_results['explained_variance'][:max_components]
    cumulative_var = fpca_results['cumulative_variance'][:max_components]
    axes[1, 1].plot(n_components_range, cumulative_var, 'mo-', label='Cumulative')
    axes[1, 1].plot(n_components_range, explained_var, 'co-', label='Individual')
    axes[1, 1].set_title('Explained Variance')
    axes[1, 1].set_xlabel('Number of Components')
    axes[1, 1].set_ylabel('Explained Variance')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fpca_reconstruction_errors.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return errors

def plot_score_distributions(fpca_results, n_components=3, figsize=(15, 10), output_dir="outputs"):
    """
    Plot distributions of the first n principal component scores.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    scores = fpca_results['scores']
    eigenvalues = fpca_results['eigenvalues']
    
    # Plot score distributions
    for i in range(min(n_components, 3)):
        axes[0, 0].hist(scores[:, i], bins=30, alpha=0.7, 
                       label=f'PC{i+1} (σ={np.std(scores[:, i]):.3f})')
    
    axes[0, 0].set_title('Distribution of Principal Component Scores')
    axes[0, 0].set_xlabel('Score Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot score scatter plots
    if n_components >= 2:
        axes[0, 1].scatter(scores[:, 0], scores[:, 1], alpha=0.6)
        axes[0, 1].set_xlabel('PC1 Score')
        axes[0, 1].set_ylabel('PC2 Score')
        axes[0, 1].set_title('PC1 vs PC2 Scores')
        axes[0, 1].grid(True, alpha=0.3)
    
    if n_components >= 3:
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
    plt.savefig(os.path.join(output_dir, 'fpca_scores.png'), dpi=300, bbox_inches='tight')
    plt.show()

def generate_fpca_report(fpca_results, original_curves, errors, param_names, output_dir="outputs"):
    """
    Generate a comprehensive FPCA report.
    """
    report = []
    report.append("=" * 60)
    report.append("FUNCTIONAL PCA ANALYSIS REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Basic information
    report.append("BASIC INFORMATION:")
    report.append(f"Number of curves: {len(original_curves)}")
    report.append(f"Curve length: {original_curves.shape[1]} time steps")
    report.append(f"Number of components computed: {fpca_results['n_components']}")
    report.append("")
    
    # Explained variance
    report.append("EXPLAINED VARIANCE:")
    for i in range(min(10, len(fpca_results['explained_variance']))):
        cum_var = fpca_results['cumulative_variance'][i]
        report.append(f"PC{i+1}: {fpca_results['explained_variance'][i]:.4f} "
                     f"(cumulative: {cum_var:.4f})")
    report.append("")
    
    # Components needed for different thresholds
    thresholds = [0.8, 0.9, 0.95, 0.99]
    report.append("COMPONENTS NEEDED FOR VARIANCE THRESHOLDS:")
    for threshold in thresholds:
        n_comp = np.argmax(fpca_results['cumulative_variance'] >= threshold) + 1
        report.append(f"{threshold*100}% variance: {n_comp} components")
    report.append("")
    
    # Reconstruction errors
    report.append("RECONSTRUCTION ERRORS:")
    for i, error in enumerate(errors):
        n_comp = i + 1
        report.append(f"{n_comp} components:")
        report.append(f"  RMSE: {error['rmse']:.6f}")
        report.append(f"  MAE:  {error['mae']:.6f}")
        report.append(f"  R²:   {error['r_squared']:.6f}")
    report.append("")
    
    # Score statistics
    report.append("PRINCIPAL COMPONENT SCORE STATISTICS:")
    scores = fpca_results['scores']
    for i in range(min(5, scores.shape[1])):
        score_std = np.std(scores[:, i])
        score_range = np.max(scores[:, i]) - np.min(scores[:, i])
        report.append(f"PC{i+1}: std={score_std:.4f}, range={score_range:.4f}")
    report.append("")
    
    # Save report
    with open(os.path.join(output_dir, 'fpca_report.txt'), 'w') as f:
        f.write('\n'.join(report))
    
    # Print to console
    print('\n'.join(report))

def main():
    """
    Main function to run FPCA analysis.
    """
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description='Analyze Functional PCA on training data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default Geballe data
  python analyze_fpca.py
  
  # Use Edmund data
  python analyze_fpca.py --input outputs/edmund1/uq_batch_results.npz --output-dir outputs/edmund1
  
  # Use custom data with different number of components
  python analyze_fpca.py --input my_data.npz --output-dir my_outputs --components 6
        """
    )
    
    parser.add_argument('--input', type=str, default='outputs/uq_batch_results.npz',
                       help='Path to the input .npz file (default: outputs/uq_batch_results.npz)')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory for results (default: outputs)')
    parser.add_argument('--components', type=int, default=4,
                       help='Number of components to compute (default: 4)')
    
    args = parser.parse_args()
    
    print(f"Starting FPCA analysis...")
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of components: {args.components}")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and prepare data
    curves, params, param_names = load_and_prepare_data(args.input)
    
    # Compute FPCA
    fpca_results = compute_fpca(curves, n_components=args.components)
    
    # Generate visualizations with custom output directory
    print("\nGenerating visualizations...")
    plot_eigenfunctions(fpca_results, output_dir=args.output_dir)
    plot_reconstruction_examples(curves, fpca_results, output_dir=args.output_dir)
    errors = plot_reconstruction_errors(curves, fpca_results, output_dir=args.output_dir)
    plot_score_distributions(fpca_results, output_dir=args.output_dir)
    
    # Generate report
    print("\nGenerating FPCA report...")
    generate_fpca_report(fpca_results, curves, errors, param_names, output_dir=args.output_dir)
    
    print(f"\nFPCA analysis completed! Check the '{args.output_dir}' directory for:")
    print("- fpca_eigenfunctions.png: Eigenfunctions and variance plots")
    print("- fpca_reconstruction_examples.png: Original vs reconstructed curves")
    print("- fpca_reconstruction_errors.png: Error metrics vs components")
    print("- fpca_scores.png: Score distributions and scatter plots")
    print("- fpca_report.txt: Comprehensive text report")
    
    # Return results for further analysis
    return fpca_results, errors

if __name__ == "__main__":
    fpca_results, errors = main() 