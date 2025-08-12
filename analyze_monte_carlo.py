#!/usr/bin/env python3
"""
Numeric diagnostics for Monte Carlo kappa samples.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from analysis.uq_wrapper import load_fpca_model
from analysis.config_utils import get_param_defs_from_config

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_pc_correlation_plots(mc_data, fpca_model, param_names, output_dir="outputs"):
    """
    Create correlation plots for each PC showing:
    1. Bar chart of parameter correlations with PC scores
    2. The PC eigenfunction curve
    
    Parameters:
    -----------
    mc_data : dict
        Monte Carlo data containing pc_scores_samples and fixed_params_samples
    fpca_model : dict
        FPCA model containing eigenfunctions and other data
    param_names : list
        List of parameter names
    output_dir : str
        Directory to save plots
    """
    print("\n" + "="*60)
    print("CREATING PC CORRELATION PLOTS")
    print("="*60)
    
    # Extract data
    pc_scores = mc_data['pc_scores_samples']  # shape (N, n_components)
    fixed_params_dicts = mc_data['fixed_params_samples']  # shape (N,) - array of dicts
    k_samples = mc_data['k_samples']  # shape (N, 3)
    
    # Convert fixed_params from array of dicts to 2D array
    param_defs = get_param_defs_from_config()
    fixed_param_names = [p["name"] for p in param_defs[:8]]  # First 8 are fixed params
    
    fixed_params = np.array([[sample[name] for name in fixed_param_names] 
                            for sample in fixed_params_dicts])
    
    # Combine all parameters for correlation analysis
    # Order: fixed_params (8) + k_samples (3) = 11 total
    all_params = np.hstack([fixed_params, k_samples])  # shape (N, 11)
    
    # Get parameter names
    param_defs = get_param_defs_from_config()
    full_param_names = [p["name"] for p in param_defs]
    
    print(f"PC scores shape: {pc_scores.shape}")
    print(f"All parameters shape: {all_params.shape}")
    print(f"Number of PCs: {pc_scores.shape[1]}")
    print(f"Number of parameters: {len(full_param_names)}")
    
    # Create figure with subplots for each PC
    n_components = pc_scores.shape[1]
    fig, axes = plt.subplots(n_components, 2, figsize=(16, 4 * n_components))
    
    # Time points for eigenfunction plots
    time_points = np.linspace(0, 7.5e-6, fpca_model['eigenfunctions'].shape[0])
    
    for pc_idx in range(n_components):
        print(f"\nAnalyzing PC{pc_idx + 1}...")
        
        # Get PC scores for this component
        pc_scores_this = pc_scores[:, pc_idx]
        
        # Calculate correlations with all parameters
        correlations = []
        for param_idx in range(all_params.shape[1]):
            corr = np.corrcoef(all_params[:, param_idx], pc_scores_this)[0, 1]
            correlations.append(corr)
        
        # Create bar chart of correlations
        ax1 = axes[pc_idx, 0] if n_components > 1 else axes[0]
        bars = ax1.bar(range(len(full_param_names)), correlations, 
                      color='skyblue', alpha=0.7, edgecolor='navy', linewidth=0.5)
        
        # Color bars based on correlation strength
        for i, (bar, corr) in enumerate(zip(bars, correlations)):
            if abs(corr) > 0.3:
                bar.set_color('red' if corr > 0 else 'blue')
                bar.set_alpha(0.8)
        
        ax1.set_title(f'PC{pc_idx + 1} Parameter Correlations', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Parameters')
        ax1.set_ylabel('Pearson Correlation')
        ax1.set_xticks(range(len(full_param_names)))
        ax1.set_xticklabels(full_param_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add correlation values on bars
        for i, corr in enumerate(correlations):
            ax1.text(i, corr + (0.02 if corr >= 0 else -0.02), 
                    f'{corr:.3f}', ha='center', va='bottom' if corr >= 0 else 'top',
                    fontsize=8, fontweight='bold')
        
        # Plot eigenfunction
        ax2 = axes[pc_idx, 1] if n_components > 1 else axes[1]
        eigenfunction = fpca_model['eigenfunctions'][:, pc_idx]
        explained_var = fpca_model['explained_variance'][pc_idx]
        
        ax2.plot(time_points * 1e6, eigenfunction, 'r-', linewidth=2, 
                label=f'PC{pc_idx + 1} ({explained_var:.1%} variance)')
        ax2.set_title(f'PC{pc_idx + 1} Eigenfunction', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time (μs)')
        ax2.set_ylabel('Eigenfunction Amplitude')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Print top correlations
        sorted_corrs = sorted(enumerate(correlations), key=lambda x: abs(x[1]), reverse=True)
        print(f"  Top correlations with PC{pc_idx + 1}:")
        for i, (param_idx, corr) in enumerate(sorted_corrs[:5]):
            print(f"    {full_param_names[param_idx]:15s}: {corr:+.3f}")
    
    plt.tight_layout()
    
    # Save plot
    plot_file = f"{output_dir}/pc_correlation_analysis.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nPC correlation analysis plot saved to: {plot_file}")
    plt.show()
    
    return correlations

# Load MC results
mc_data = np.load("outputs/propagated_k_values.npz", allow_pickle=True)
k_samples = mc_data['k_samples']  # shape (N, 3)
param_names = ["κ_sample", "κ_ins", "κ_coupler"]

# Best-fit point
best_draw_idx = mc_data['best_draw_idx'] if 'best_draw_idx' in mc_data else None
best_point = k_samples[best_draw_idx] if best_draw_idx is not None else None

# Fixed parameters used in MC (if available)
fixed_params_samples = mc_data['fixed_params_samples'] if 'fixed_params_samples' in mc_data else None

print("=" * 60)
print("MONTE CARLO NUMERIC DIAGNOSTICS")
print("=" * 60)
print(f"Number of samples: {len(k_samples)}")
print(f"Parameter names: {param_names}")

# 1. Empirical covariance & correlation
print("\n" + "=" * 40)
print("1. COVARIANCE & CORRELATION MATRICES")
print("=" * 40)

cov = np.cov(k_samples, rowvar=False)
corr = np.corrcoef(k_samples, rowvar=False)

print("Covariance matrix:")
for i, row in enumerate(cov):
    print(f"  [{row[0]:8.4f}, {row[1]:8.4f}, {row[2]:8.4f}]  # {param_names[i]}")

print("\nCorrelation matrix:")
for i, row in enumerate(corr):
    print(f"  [{row[0]:8.4f}, {row[1]:8.4f}, {row[2]:8.4f}]  # {param_names[i]}")

# 2. Principal directions & identifiability
print("\n" + "=" * 40)
print("2. PRINCIPAL DIRECTIONS & IDENTIFIABILITY")
print("=" * 40)

eigvals, eigvecs = np.linalg.eigh(cov)
order = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[order], eigvecs[:, order]

print(f"Eigenvalues: {eigvals}")
print(f"Condition number: {eigvals[0]/eigvals[-1]:.2e}")

print("\nPrincipal axes (columns):")
for i, (eigval, eigvec) in enumerate(zip(eigvals, eigvecs.T)):
    print(f"  PC{i+1} (λ={eigval:.4f}): [{eigvec[0]:8.4f}, {eigvec[1]:8.4f}, {eigvec[2]:8.4f}]")

# Check for identifiability issues
condition_number = eigvals[0] / eigvals[-1]
if condition_number > 100:
    print(f"\n⚠️  WARNING: Large condition number ({condition_number:.2e}) suggests poor identifiability")
    print("   This indicates a 'flat' direction in parameter space")
else:
    print(f"\n✅ Condition number ({condition_number:.2e}) suggests good identifiability")

# 3. Mahalanobis distance of the best-fit
if best_point is not None:
    print("\n" + "=" * 40)
    print("3. MAHALANOBIS DISTANCE OF BEST-FIT")
    print("=" * 40)
    
    mu = k_samples.mean(axis=0)
    delta = best_point - mu
    D2 = delta @ np.linalg.inv(cov) @ delta
    
    print(f"Best-fit point: {best_point}")
    print(f"Sample mean: {mu}")
    print(f"Deviation: {delta}")
    print(f"Mahalanobis D² of θ̂: {D2:.4f}")
    
    # Interpret the Mahalanobis distance
    if D2 < 3:
        print("✅ Best-fit is typical within the cloud (D² < 3)")
    elif D2 < 6:
        print("⚠️  Best-fit is somewhat atypical (3 ≤ D² < 6)")
    else:
        print("❌ Best-fit is very atypical (D² ≥ 6)")

# 4. Credible intervals (numeric)
print("\n" + "=" * 40)
print("4. CREDIBLE INTERVALS")
print("=" * 40)

ci68 = np.percentile(k_samples, [16, 84], axis=0)
ci95 = np.percentile(k_samples, [2.5, 97.5], axis=0)

print("68% Credible Intervals:")
for i, (low, high) in enumerate(ci68.T):
    print(f"  {param_names[i]}: [{low:.4f}, {high:.4f}]")

print("\n95% Credible Intervals:")
for i, (low, high) in enumerate(ci95.T):
    print(f"  {param_names[i]}: [{low:.4f}, {high:.4f}]")

# Additional summary statistics
print("\n" + "=" * 40)
print("5. SUMMARY STATISTICS")
print("=" * 40)

means = k_samples.mean(axis=0)
stds = k_samples.std(axis=0)

print("Parameter means and standard deviations:")
for i, (name, mean, std) in enumerate(zip(param_names, means, stds)):
    print(f"  {name}: {mean:.4f} ± {std:.4f}")

# Parameter ranges
mins = k_samples.min(axis=0)
maxs = k_samples.max(axis=0)

print("\nParameter ranges:")
for i, (name, min_val, max_val) in enumerate(zip(param_names, mins, maxs)):
    print(f"  {name}: [{min_val:.4f}, {max_val:.4f}]")

# 6. Fixed parameters used in MC
if fixed_params_samples is not None:
    print("\n" + "=" * 40)
    print("6. FIXED PARAMETERS USED IN MC")
    print("=" * 40)
    
    # Convert list of dicts to array for easier computation
    fixed_param_names = list(fixed_params_samples[0].keys())
    fixed_param_array = np.array([[sample[name] for name in fixed_param_names] 
                                 for sample in fixed_params_samples])
    
    print("Fixed parameter averages (from MC draws):")
    for i, name in enumerate(fixed_param_names):
        mean_val = np.mean(fixed_param_array[:, i])
        std_val = np.std(fixed_param_array[:, i])
        print(f"  {name}: {mean_val:.6e} ± {std_val:.6e}")
    
    print(f"\nNote: These are the averages of the {len(fixed_params_samples)} fixed parameter")
    print("draws used in the Monte Carlo process (not the kappa parameters above).")

# 7. PC Correlation Analysis (if PC scores are available)
if 'pc_scores_samples' in mc_data:
    print("\n" + "=" * 40)
    print("7. PC CORRELATION ANALYSIS")
    print("=" * 40)
    
    # Load FPCA model
    try:
        fpca_model = load_fpca_model("outputs/fpca_model.npz")
        print("FPCA model loaded successfully")
        
        # Get parameter names
        param_defs = get_param_defs_from_config()
        param_names = [p["name"] for p in param_defs]
        
        # Create PC correlation plots
        create_pc_correlation_plots(mc_data, fpca_model, param_names)
        
    except Exception as e:
        print(f"Could not create PC correlation plots: {e}")
        print("Make sure fpca_model.npz exists in outputs/ directory")
else:
    print("\n" + "=" * 40)
    print("7. PC CORRELATION ANALYSIS")
    print("=" * 40)
    print("PC scores not found in Monte Carlo data.")
    print("Run uqpy_ls.py with --mc flag to generate PC scores.")

print("\n" + "=" * 60)
print("DIAGNOSTICS COMPLETE")
print("=" * 60)
