#!/usr/bin/env python3
"""
Script to examine the training data ranges and parameter coverage of the surrogate model.
This helps identify if the surrogate was trained on appropriate parameter ranges.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from analysis.uq_wrapper import load_recast_training_data, load_fpca_model
from train_surrogate_models import FullSurrogateModel
from analysis.config_utils import get_param_defs_from_config
import seaborn as sns

def load_training_data():
    """
    Load the training data used for the surrogate model.
    """
    print("Loading training data...")
    
    try:
        # Load recast training data
        recast_data = load_recast_training_data("outputs/edmund1/training_data_fpca_narrow_k.npz")
        parameters = recast_data['parameters']
        fpca_scores = recast_data['fpca_scores']
        parameter_names = recast_data['parameter_names']
        
        print(f"Training data shape: {parameters.shape}")
        print(f"FPCA scores shape: {fpca_scores.shape}")
        print(f"Parameter names: {parameter_names}")
        
        return parameters, fpca_scores, parameter_names
        
    except Exception as e:
        print(f"ERROR loading training data: {e}")
        return None, None, None

def load_surrogate_model():
    """
    Load the surrogate model to get its parameter information.
    """
    print("\nLoading surrogate model...")
    
    try:
        surrogate = FullSurrogateModel.load_model("outputs/edmund1/full_surrogate_model_narrow_k.pkl")
        
        print(f"Surrogate parameter names: {surrogate.parameter_names}")
        print(f"Surrogate parameter ranges: {surrogate.param_ranges}")
        print(f"Number of parameters: {surrogate.n_parameters}")
        print(f"Number of FPCA components: {surrogate.n_components}")
        
        return surrogate
        
    except Exception as e:
        print(f"ERROR loading surrogate model: {e}")
        return None

def load_current_config_params():
    """
    Load current parameter definitions from Edmund config.
    """
    print("\nLoading current Edmund config parameters...")
    
    try:
        param_defs = get_param_defs_from_config("configs/distributions_edmund.yaml")
        param_names = [param_def['name'] for param_def in param_defs]
        
        # Extract parameter ranges from config
        param_ranges = {}
        for param_def in param_defs:
            name = param_def['name']
            if param_def['type'] == 'uniform':
                param_ranges[name] = (param_def['low'], param_def['high'])
            elif param_def['type'] == 'normal':
                # For normal distributions, use ±3σ range
                center = param_def['center']
                sigma = param_def['sigma']
                param_ranges[name] = (center - 3*sigma, center + 3*sigma)
            elif param_def['type'] == 'lognormal':
                # For lognormal, use ±3σ_log range
                center = param_def['center']
                sigma_log = param_def['sigma_log']
                param_ranges[name] = (center * np.exp(-3*sigma_log), center * np.exp(3*sigma_log))
        
        print(f"Current config parameter names: {param_names}")
        print(f"Current config parameter ranges: {param_ranges}")
        
        return param_names, param_ranges, param_defs
        
    except Exception as e:
        print(f"ERROR loading current config: {e}")
        return None, None, None

def analyze_parameter_coverage(parameters, parameter_names, surrogate_ranges, current_ranges):
    """
    Analyze parameter coverage and ranges.
    """
    print("\n" + "="*60)
    print("PARAMETER COVERAGE ANALYSIS")
    print("="*60)
    
    # Calculate statistics for each parameter
    for i, name in enumerate(parameter_names):
        if i < parameters.shape[1]:
            values = parameters[:, i]
            min_val = np.min(values)
            max_val = np.max(values)
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            print(f"\n{name}:")
            print(f"  Training data range: [{min_val:.3e}, {max_val:.3e}]")
            print(f"  Training data mean: {mean_val:.3e}")
            print(f"  Training data std: {std_val:.3e}")
            
            # Compare with surrogate ranges
            if name in surrogate_ranges:
                surr_min, surr_max = surrogate_ranges[name]
                print(f"  Surrogate range: [{surr_min:.3e}, {surr_max:.3e}]")
                
                # Check if training data covers surrogate range
                coverage = (min_val <= surr_min and max_val >= surr_max)
                print(f"  Training covers surrogate range: {coverage}")
            
            # Compare with current config ranges
            if name in current_ranges:
                curr_min, curr_max = current_ranges[name]
                print(f"  Current config range: [{curr_min:.3e}, {curr_max:.3e}]")
                
                # Check if training data covers current range
                coverage = (min_val <= curr_min and max_val >= curr_max)
                print(f"  Training covers current range: {coverage}")
                
                if not coverage:
                    print(f"  WARNING: Training data does not cover current config range!")
                    print(f"    Missing: {curr_min:.3e} to {min_val:.3e} and/or {max_val:.3e} to {curr_max:.3e}")

def check_parameter_mismatch(training_names, surrogate_names, current_names):
    """
    Check for parameter mismatches between training, surrogate, and current config.
    """
    print("\n" + "="*60)
    print("PARAMETER MISMATCH ANALYSIS")
    print("="*60)
    
    training_set = set(training_names)
    surrogate_set = set(surrogate_names)
    current_set = set(current_names)
    
    print(f"Training parameters: {training_set}")
    print(f"Surrogate parameters: {surrogate_set}")
    print(f"Current config parameters: {current_set}")
    
    # Check for missing parameters
    missing_in_surrogate = training_set - surrogate_set
    missing_in_current = training_set - current_set
    extra_in_current = current_set - training_set
    
    if missing_in_surrogate:
        print(f"\nWARNING: Parameters in training but missing in surrogate: {missing_in_surrogate}")
    
    if missing_in_current:
        print(f"\nWARNING: Parameters in training but missing in current config: {missing_in_current}")
    
    if extra_in_current:
        print(f"\nWARNING: Parameters in current config but not in training: {extra_in_current}")
        print(f"  This includes the new k_int parameter!")
    
    if not missing_in_surrogate and not missing_in_current and not extra_in_current:
        print(f"\nSUCCESS: All parameter sets match!")

def create_parameter_plots(parameters, parameter_names, surrogate_ranges, current_ranges):
    """
    Create plots showing parameter distributions and ranges.
    """
    print("\nCreating parameter analysis plots...")
    
    n_params = len(parameter_names)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, name in enumerate(parameter_names):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        if i < parameters.shape[1]:
            values = parameters[:, i]
            
            # Plot histogram
            ax.hist(values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            
            # Add range lines
            min_val, max_val = np.min(values), np.max(values)
            ax.axvline(min_val, color='red', linestyle='--', alpha=0.8, label=f'Min: {min_val:.2e}')
            ax.axvline(max_val, color='red', linestyle='--', alpha=0.8, label=f'Max: {max_val:.2e}')
            
            # Add surrogate range if available
            if name in surrogate_ranges:
                surr_min, surr_max = surrogate_ranges[name]
                ax.axvline(surr_min, color='green', linestyle=':', alpha=0.8, label=f'Surr Min: {surr_min:.2e}')
                ax.axvline(surr_max, color='green', linestyle=':', alpha=0.8, label=f'Surr Max: {surr_max:.2e}')
            
            # Add current config range if available
            if name in current_ranges:
                curr_min, curr_max = current_ranges[name]
                ax.axvline(curr_min, color='orange', linestyle='-.', alpha=0.8, label=f'Config Min: {curr_min:.2e}')
                ax.axvline(curr_max, color='orange', linestyle='-.', alpha=0.8, label=f'Config Max: {curr_max:.2e}')
            
            ax.set_title(f'{name} Distribution')
            ax.set_xlabel('Parameter Value')
            ax.set_ylabel('Frequency')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'No data for {name}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{name} (No Data)')
    
    # Hide unused subplots
    for i in range(n_params, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig("surrogate_training_parameter_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_correlation_plot(parameters, parameter_names):
    """
    Create correlation plot between parameters.
    """
    print("\nCreating parameter correlation plot...")
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame(parameters, columns=parameter_names)
    
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f', cbar_kws={'label': 'Correlation'})
    plt.title('Parameter Correlation Matrix (Training Data)')
    plt.tight_layout()
    plt.savefig("surrogate_training_correlations.png", dpi=300, bbox_inches='tight')
    plt.show()

def analyze_fpca_coverage(fpca_scores):
    """
    Analyze FPCA score coverage.
    """
    print("\n" + "="*60)
    print("FPCA SCORE ANALYSIS")
    print("="*60)
    
    n_components = fpca_scores.shape[1]
    
    print(f"Number of FPCA components: {n_components}")
    print(f"Number of training samples: {fpca_scores.shape[0]}")
    
    for i in range(n_components):
        scores = fpca_scores[:, i]
        min_score = np.min(scores)
        max_score = np.max(scores)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        print(f"\nPC{i+1}:")
        print(f"  Range: [{min_score:.6f}, {max_score:.6f}]")
        print(f"  Mean: {mean_score:.6f}")
        print(f"  Std: {std_score:.6f}")

def main():
    """
    Main function to analyze surrogate training data.
    """
    print("=" * 60)
    print("SURROGATE TRAINING DATA ANALYSIS")
    print("=" * 60)
    
    # Load training data
    parameters, fpca_scores, training_names = load_training_data()
    
    if parameters is None:
        print("ERROR: Could not load training data!")
        return
    
    # Load surrogate model
    surrogate = load_surrogate_model()
    
    if surrogate is None:
        print("ERROR: Could not load surrogate model!")
        return
    
    # Load current config parameters
    current_names, current_ranges, param_defs = load_current_config_params()
    
    if current_names is None:
        print("ERROR: Could not load current config!")
        return
    
    # Analyze parameter coverage
    analyze_parameter_coverage(parameters, training_names, surrogate.param_ranges, current_ranges)
    
    # Check for parameter mismatches
    check_parameter_mismatch(training_names, surrogate.parameter_names, current_names)
    
    # Analyze FPCA coverage
    analyze_fpca_coverage(fpca_scores)
    
    # Create plots
    fig1 = create_parameter_plots(parameters, training_names, surrogate.param_ranges, current_ranges)
    create_correlation_plot(parameters, training_names)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print(f"Training data samples: {parameters.shape[0]}")
    print(f"Training parameters: {len(training_names)}")
    print(f"Surrogate parameters: {len(surrogate.parameter_names)}")
    print(f"Current config parameters: {len(current_names)}")
    
    # Check if k_int is missing
    if 'k_int' in current_names and 'k_int' not in training_names:
        print("\nCRITICAL ISSUE: k_int parameter is in current config but NOT in training data!")
        print("This means the surrogate model was not trained with the k_int parameter.")
        print("The surrogate will not be able to properly handle k_int variations.")
        print("\nRECOMMENDATION: Retrain the surrogate model with k_int included.")
    
    print(f"\nPlots saved as:")
    print(f"  - surrogate_training_parameter_analysis.png")
    print(f"  - surrogate_training_correlations.png")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 