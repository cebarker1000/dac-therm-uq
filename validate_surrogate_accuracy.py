#!/usr/bin/env python3
"""
Script to validate surrogate model accuracy against all training data points.
This helps identify systematic errors and regions where the surrogate performs poorly.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from analysis.uq_wrapper import load_recast_training_data, load_fpca_model, reconstruct_curve_from_fpca
from train_surrogate_models import FullSurrogateModel
from sklearn.metrics import mean_squared_error, r2_score
import time
import seaborn as sns

def load_training_data():
    """
    Load the training data used for the surrogate model.
    """
    print("Loading training data...")
    
    try:
        # Load recast training data
        recast_data = load_recast_training_data("outputs/edmund1/training_data_fpca_int_ins_match.npz")
        parameters = recast_data['parameters']
        fpca_scores = recast_data['fpca_scores']
        parameter_names = recast_data['parameter_names']
        
        # Load FPCA model to get original curves
        fpca_model = load_fpca_model("outputs/edmund1/fpca_model_int_ins_match.npz")
        
        print(f"Training data shape: {parameters.shape}")
        print(f"FPCA scores shape: {fpca_scores.shape}")
        print(f"Parameter names: {parameter_names}")
        
        return parameters, fpca_scores, parameter_names, fpca_model
        
    except Exception as e:
        print(f"ERROR loading training data: {e}")
        return None, None, None, None

def load_surrogate_model():
    """
    Load the surrogate model.
    """
    print("\nLoading surrogate model...")
    
    try:
        surrogate = FullSurrogateModel.load_model("outputs/edmund1/full_surrogate_model_int_ins_match.pkl")
        
        print(f"Surrogate parameter names: {surrogate.parameter_names}")
        print(f"Number of parameters: {surrogate.n_parameters}")
        print(f"Number of FPCA components: {surrogate.n_components}")
        
        return surrogate
        
    except Exception as e:
        print(f"ERROR loading surrogate model: {e}")
        return None

def reconstruct_original_curves(fpca_scores, fpca_model):
    """
    Reconstruct original temperature curves from FPCA scores.
    """
    print("Reconstructing original curves from FPCA scores...")
    
    original_curves = []
    for i, scores in enumerate(fpca_scores):
        curve = reconstruct_curve_from_fpca(scores, fpca_model)
        original_curves.append(curve)
    
    return np.array(original_curves)

def validate_surrogate_accuracy(surrogate, parameters, original_curves, parameter_names, 
                               max_samples=None, random_seed=42):
    """
    Validate surrogate accuracy against training data.
    
    Parameters:
    -----------
    surrogate : FullSurrogateModel
        The surrogate model to validate
    parameters : np.ndarray
        Training parameters
    original_curves : np.ndarray
        Original temperature curves from training data
    parameter_names : list
        Names of parameters
    max_samples : int, optional
        Maximum number of samples to test (for speed)
    random_seed : int
        Random seed for sampling
        
    Returns:
    --------
    dict
        Dictionary containing validation results
    """
    print(f"\nValidating surrogate accuracy...")
    
    n_samples = len(parameters)
    if max_samples is not None and max_samples < n_samples:
        # Randomly sample subset for faster validation
        np.random.seed(random_seed)
        indices = np.random.choice(n_samples, max_samples, replace=False)
        test_parameters = parameters[indices]
        test_curves = original_curves[indices]
        print(f"Testing on {max_samples} randomly selected samples out of {n_samples}")
    else:
        test_parameters = parameters
        test_curves = original_curves
        print(f"Testing on all {n_samples} samples")
    
    # Predict curves using surrogate
    print("Making surrogate predictions...")
    start_time = time.time()
    
    predicted_curves, predicted_fpca, fpca_uncertainties, curve_uncertainties = surrogate.predict_temperature_curves(test_parameters)
    
    prediction_time = time.time() - start_time
    print(f"Prediction completed in {prediction_time:.2f} seconds")
    print(f"Average time per prediction: {prediction_time/len(test_parameters):.4f} seconds")
    
    # Calculate accuracy metrics
    print("Calculating accuracy metrics...")
    
    rmse_values = []
    r2_values = []
    max_errors = []
    mean_errors = []
    
    for i in range(len(test_curves)):
        original = test_curves[i]
        predicted = predicted_curves[i]
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(original, predicted))
        r2 = r2_score(original, predicted)
        max_error = np.max(np.abs(original - predicted))
        mean_error = np.mean(np.abs(original - predicted))
        
        rmse_values.append(rmse)
        r2_values.append(r2)
        max_errors.append(max_error)
        mean_errors.append(mean_error)
    
    # Calculate overall statistics
    results = {
        'rmse_values': np.array(rmse_values),
        'r2_values': np.array(r2_values),
        'max_errors': np.array(max_errors),
        'mean_errors': np.array(mean_errors),
        'test_parameters': test_parameters,
        'original_curves': test_curves,
        'predicted_curves': predicted_curves,
        'predicted_fpca': predicted_fpca,
        'fpca_uncertainties': fpca_uncertainties,
        'curve_uncertainties': curve_uncertainties,
        'parameter_names': parameter_names,
        'prediction_time': prediction_time,
        'n_samples_tested': len(test_parameters)
    }
    
    return results

def print_accuracy_summary(results):
    """
    Print summary of accuracy metrics.
    """
    print("\n" + "="*60)
    print("SURROGATE ACCURACY SUMMARY")
    print("="*60)
    
    rmse_values = results['rmse_values']
    r2_values = results['r2_values']
    max_errors = results['max_errors']
    mean_errors = results['mean_errors']
    
    print(f"Number of samples tested: {results['n_samples_tested']}")
    print(f"Total prediction time: {results['prediction_time']:.2f} seconds")
    print(f"Average time per prediction: {results['prediction_time']/results['n_samples_tested']:.4f} seconds")
    
    print(f"\nRMSE Statistics:")
    print(f"  Mean RMSE: {np.mean(rmse_values):.6f}")
    print(f"  Median RMSE: {np.median(rmse_values):.6f}")
    print(f"  Std RMSE: {np.std(rmse_values):.6f}")
    print(f"  Min RMSE: {np.min(rmse_values):.6f}")
    print(f"  Max RMSE: {np.max(rmse_values):.6f}")
    print(f"  95th percentile RMSE: {np.percentile(rmse_values, 95):.6f}")
    
    print(f"\nR² Statistics:")
    print(f"  Mean R²: {np.mean(r2_values):.6f}")
    print(f"  Median R²: {np.median(r2_values):.6f}")
    print(f"  Std R²: {np.std(r2_values):.6f}")
    print(f"  Min R²: {np.min(r2_values):.6f}")
    print(f"  Max R²: {np.max(r2_values):.6f}")
    print(f"  5th percentile R²: {np.percentile(r2_values, 5):.6f}")
    
    print(f"\nMax Error Statistics:")
    print(f"  Mean max error: {np.mean(max_errors):.6f}")
    print(f"  Median max error: {np.median(max_errors):.6f}")
    print(f"  Std max error: {np.std(max_errors):.6f}")
    print(f"  Min max error: {np.min(max_errors):.6f}")
    print(f"  Max max error: {np.max(max_errors):.6f}")
    
    print(f"\nMean Error Statistics:")
    print(f"  Mean mean error: {np.mean(mean_errors):.6f}")
    print(f"  Median mean error: {np.median(mean_errors):.6f}")
    print(f"  Std mean error: {np.std(mean_errors):.6f}")
    
    # Count samples with poor performance
    poor_rmse_threshold = np.percentile(rmse_values, 95)  # Top 5% worst
    poor_r2_threshold = np.percentile(r2_values, 5)       # Bottom 5% worst
    
    n_poor_rmse = np.sum(rmse_values > poor_rmse_threshold)
    n_poor_r2 = np.sum(r2_values < poor_r2_threshold)
    
    print(f"\nPoor Performance Analysis:")
    print(f"  Samples with RMSE > {poor_rmse_threshold:.6f} (95th percentile): {n_poor_rmse}")
    print(f"  Samples with R² < {poor_r2_threshold:.6f} (5th percentile): {n_poor_r2}")

def create_accuracy_plots(results):
    """
    Create plots showing surrogate accuracy.
    """
    print("\nCreating accuracy plots...")
    
    rmse_values = results['rmse_values']
    r2_values = results['r2_values']
    max_errors = results['max_errors']
    mean_errors = results['mean_errors']
    test_parameters = results['test_parameters']
    original_curves = results['original_curves']
    predicted_curves = results['predicted_curves']
    parameter_names = results['parameter_names']
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: RMSE distribution
    ax1 = plt.subplot(3, 3, 1)
    ax1.hist(rmse_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(np.mean(rmse_values), color='red', linestyle='--', label=f'Mean: {np.mean(rmse_values):.6f}')
    ax1.axvline(np.median(rmse_values), color='green', linestyle='--', label=f'Median: {np.median(rmse_values):.6f}')
    ax1.set_xlabel('RMSE')
    ax1.set_ylabel('Frequency')
    ax1.set_title('RMSE Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: R² distribution
    ax2 = plt.subplot(3, 3, 2)
    ax2.hist(r2_values, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.axvline(np.mean(r2_values), color='red', linestyle='--', label=f'Mean: {np.mean(r2_values):.6f}')
    ax2.axvline(np.median(r2_values), color='green', linestyle='--', label=f'Median: {np.median(r2_values):.6f}')
    ax2.set_xlabel('R²')
    ax2.set_ylabel('Frequency')
    ax2.set_title('R² Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Max error distribution
    ax3 = plt.subplot(3, 3, 3)
    ax3.hist(max_errors, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    ax3.axvline(np.mean(max_errors), color='red', linestyle='--', label=f'Mean: {np.mean(max_errors):.6f}')
    ax3.axvline(np.median(max_errors), color='green', linestyle='--', label=f'Median: {np.median(max_errors):.6f}')
    ax3.set_xlabel('Max Error')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Max Error Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: RMSE vs R² scatter
    ax4 = plt.subplot(3, 3, 4)
    ax4.scatter(rmse_values, r2_values, alpha=0.6, s=20)
    ax4.set_xlabel('RMSE')
    ax4.set_ylabel('R²')
    ax4.set_title('RMSE vs R²')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Parameter vs RMSE (for each parameter)
    ax5 = plt.subplot(3, 3, 5)
    n_params = min(3, len(parameter_names))  # Show first 3 parameters
    colors = ['blue', 'red', 'green']
    
    for i in range(n_params):
        ax5.scatter(test_parameters[:, i], rmse_values, alpha=0.6, s=20, 
                   color=colors[i], label=parameter_names[i])
    
    ax5.set_xlabel('Parameter Value')
    ax5.set_ylabel('RMSE')
    ax5.set_title('RMSE vs Parameters (First 3)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Parameter vs R² (for each parameter)
    ax6 = plt.subplot(3, 3, 6)
    for i in range(n_params):
        ax6.scatter(test_parameters[:, i], r2_values, alpha=0.6, s=20, 
                   color=colors[i], label=parameter_names[i])
    
    ax6.set_xlabel('Parameter Value')
    ax6.set_ylabel('R²')
    ax6.set_title('R² vs Parameters (First 3)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Example curves (best case)
    ax7 = plt.subplot(3, 3, 7)
    best_idx = np.argmax(r2_values)
    best_r2 = r2_values[best_idx]
    best_rmse = rmse_values[best_idx]
    
    # Create time array
    n_timepoints = original_curves.shape[1]
    sim_t_final = 8.5e-6  # seconds
    time_array = np.linspace(0, sim_t_final, n_timepoints)
    
    ax7.plot(time_array * 1e6, original_curves[best_idx], 'b-', linewidth=2, label='Original')
    ax7.plot(time_array * 1e6, predicted_curves[best_idx], 'r--', linewidth=2, label='Surrogate')
    ax7.set_xlabel('Time (μs)')
    ax7.set_ylabel('Normalized Temperature')
    ax7.set_title(f'Best Case (R²={best_r2:.4f}, RMSE={best_rmse:.6f})')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Example curves (worst case)
    ax8 = plt.subplot(3, 3, 8)
    worst_idx = np.argmin(r2_values)
    worst_r2 = r2_values[worst_idx]
    worst_rmse = rmse_values[worst_idx]
    
    ax8.plot(time_array * 1e6, original_curves[worst_idx], 'b-', linewidth=2, label='Original')
    ax8.plot(time_array * 1e6, predicted_curves[worst_idx], 'r--', linewidth=2, label='Surrogate')
    ax8.set_xlabel('Time (μs)')
    ax8.set_ylabel('Normalized Temperature')
    ax8.set_title(f'Worst Case (R²={worst_r2:.4f}, RMSE={worst_rmse:.6f})')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Plot 9: Error vs time for worst case
    ax9 = plt.subplot(3, 3, 9)
    error_curve = np.abs(original_curves[worst_idx] - predicted_curves[worst_idx])
    ax9.plot(time_array * 1e6, error_curve, 'g-', linewidth=2)
    ax9.set_xlabel('Time (μs)')
    ax9.set_ylabel('Absolute Error')
    ax9.set_title('Error vs Time (Worst Case)')
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("surrogate_accuracy_validation.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_detailed_error_analysis(results):
    """
    Create detailed error analysis plots.
    """
    print("\nCreating detailed error analysis...")
    
    rmse_values = results['rmse_values']
    r2_values = results['r2_values']
    test_parameters = results['test_parameters']
    parameter_names = results['parameter_names']
    
    # Find worst performing samples
    worst_indices = np.argsort(r2_values)[:10]  # 10 worst
    best_indices = np.argsort(r2_values)[-10:]  # 10 best
    
    print(f"\nWorst 10 samples (by R²):")
    for i, idx in enumerate(worst_indices):
        print(f"  {i+1}. Sample {idx}: R²={r2_values[idx]:.6f}, RMSE={rmse_values[idx]:.6f}")
        print(f"     Parameters: {dict(zip(parameter_names, test_parameters[idx]))}")
    
    print(f"\nBest 10 samples (by R²):")
    for i, idx in enumerate(best_indices):
        print(f"  {i+1}. Sample {idx}: R²={r2_values[idx]:.6f}, RMSE={rmse_values[idx]:.6f}")
        print(f"     Parameters: {dict(zip(parameter_names, test_parameters[idx]))}")
    
    # Create parameter range analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Parameter ranges for best vs worst samples
    ax1 = axes[0, 0]
    n_params = len(parameter_names)
    x_pos = np.arange(n_params)
    width = 0.35
    
    best_params = test_parameters[best_indices]
    worst_params = test_parameters[worst_indices]
    
    best_means = np.mean(best_params, axis=0)
    worst_means = np.mean(worst_params, axis=0)
    best_stds = np.std(best_params, axis=0)
    worst_stds = np.std(worst_params, axis=0)
    
    ax1.bar(x_pos - width/2, best_means, width, yerr=best_stds, 
            label='Best 10 samples', alpha=0.7, capsize=5)
    ax1.bar(x_pos + width/2, worst_means, width, yerr=worst_stds, 
            label='Worst 10 samples', alpha=0.7, capsize=5)
    
    ax1.set_xlabel('Parameters')
    ax1.set_ylabel('Mean Parameter Value')
    ax1.set_title('Parameter Values: Best vs Worst Samples')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(parameter_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: RMSE vs parameter values (scatter for all parameters)
    ax2 = axes[0, 1]
    for i, name in enumerate(parameter_names):
        ax2.scatter(test_parameters[:, i], rmse_values, alpha=0.6, s=20, label=name)
    
    ax2.set_xlabel('Parameter Value')
    ax2.set_ylabel('RMSE')
    ax2.set_title('RMSE vs All Parameters')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: R² vs parameter values (scatter for all parameters)
    ax3 = axes[1, 0]
    for i, name in enumerate(parameter_names):
        ax3.scatter(test_parameters[:, i], r2_values, alpha=0.6, s=20, label=name)
    
    ax3.set_xlabel('Parameter Value')
    ax3.set_ylabel('R²')
    ax3.set_title('R² vs All Parameters')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative distribution of R²
    ax4 = axes[1, 1]
    sorted_r2 = np.sort(r2_values)
    cumulative = np.arange(1, len(sorted_r2) + 1) / len(sorted_r2)
    ax4.plot(sorted_r2, cumulative, 'b-', linewidth=2)
    ax4.axhline(0.95, color='red', linestyle='--', alpha=0.7, label='95% threshold')
    ax4.axhline(0.99, color='green', linestyle='--', alpha=0.7, label='99% threshold')
    
    # Find R² values at thresholds
    r2_95 = np.percentile(r2_values, 5)  # 95% of samples have R² > this value
    r2_99 = np.percentile(r2_values, 1)  # 99% of samples have R² > this value
    
    ax4.axvline(r2_95, color='red', linestyle=':', alpha=0.7, label=f'R²={r2_95:.4f} (5th percentile)')
    ax4.axvline(r2_99, color='green', linestyle=':', alpha=0.7, label=f'R²={r2_99:.4f} (1st percentile)')
    
    ax4.set_xlabel('R²')
    ax4.set_ylabel('Cumulative Probability')
    ax4.set_title('Cumulative Distribution of R²')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("surrogate_detailed_error_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def main():
    """
    Main function to validate surrogate accuracy.
    """
    print("=" * 60)
    print("SURROGATE ACCURACY VALIDATION")
    print("=" * 60)
    
    # Load training data
    parameters, fpca_scores, parameter_names, fpca_model = load_training_data()
    
    if parameters is None:
        print("ERROR: Could not load training data!")
        return
    
    # Load surrogate model
    surrogate = load_surrogate_model()
    
    if surrogate is None:
        print("ERROR: Could not load surrogate model!")
        return
    
    # Reconstruct original curves
    original_curves = reconstruct_original_curves(fpca_scores, fpca_model)
    
    # Validate surrogate accuracy
    # Use max_samples=1000 for faster testing, or None for full validation
    results = validate_surrogate_accuracy(
        surrogate, parameters, original_curves, parameter_names, 
        max_samples=1000, random_seed=42
    )
    
    # Print accuracy summary
    print_accuracy_summary(results)
    
    # Create plots
    fig1 = create_accuracy_plots(results)
    fig2 = create_detailed_error_analysis(results)
    
    # Save results
    print(f"\nSaving results...")
    np.savez("surrogate_validation_results.npz",
             rmse_values=results['rmse_values'],
             r2_values=results['r2_values'],
             max_errors=results['max_errors'],
             mean_errors=results['mean_errors'],
             test_parameters=results['test_parameters'],
             parameter_names=results['parameter_names'],
             prediction_time=results['prediction_time'],
             n_samples_tested=results['n_samples_tested'])
    
    print(f"Results saved to: surrogate_validation_results.npz")
    print(f"Plots saved as:")
    print(f"  - surrogate_accuracy_validation.png")
    print(f"  - surrogate_detailed_error_analysis.png")
    
    print("\nValidation complete!")

if __name__ == "__main__":
    main() 