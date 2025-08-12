#!/usr/bin/env python3
"""
Create a full surrogate GP model using all training data.
Fits a GP to each FPCA coefficient and tests predictions within parameter ranges.
"""

import sys
import os


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from analysis.uq_wrapper import load_recast_training_data, load_fpca_model, reconstruct_curve_from_fpca
import pickle
import warnings
import argparse
import os
from sklearn.model_selection import train_test_split

import sys

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FullSurrogateModel:
    """
    Full surrogate model that combines FPCA and GP for each component.
    """
    
    def __init__(self, fpca_model, gps, scaler, parameter_names, param_ranges, 
                 t_final, num_steps):
        """
        Initialize the surrogate model.
        
        Parameters:
        -----------
        fpca_model : dict
            The FPCA model
        gps : list
            List of trained GP models (one per FPCA component)
        scaler : StandardScaler
            Fitted scaler for input parameters
        parameter_names : list
            Names of the parameters
        param_ranges : dict
            Dictionary of parameter ranges for validation
        t_final : float
            Final time of the simulation
        num_steps : int
            Number of time steps in the simulation
        """
        self.fpca_model = fpca_model
        self.gps = gps
        self.scaler = scaler
        self.parameter_names = parameter_names
        self.param_ranges = param_ranges
        self.n_components = fpca_model['n_components']
        self.n_parameters = len(parameter_names)
        self.t_final = t_final
        self.num_steps = num_steps
        
        # Correctly define the time grid
        dt = t_final / num_steps
        self.time_grid = np.arange(1, num_steps + 1) * dt
        
    def predict_fpca_coefficients(self, X):
        """
        Predict FPCA coefficients for given parameter values.
        
        Parameters:
        -----------
        X : np.ndarray
            Parameter values (n_samples, n_parameters)
            
        Returns:
        --------
        np.ndarray
            Predicted FPCA coefficients (n_samples, n_components)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Scale inputs
        X_scaled = self.scaler.transform(X)
        
        # Predict each component
        predictions = []
        uncertainties = []
        
        for i, gp in enumerate(self.gps):
            pred, std = gp.predict(X_scaled, return_std=True)
            predictions.append(pred)
            uncertainties.append(std)
        
        return np.column_stack(predictions), np.column_stack(uncertainties)
    
    def predict_temperature_curves(self, X):
        """
        Predict full temperature curves for given parameter values.
        
        Parameters:
        -----------
        X : np.ndarray
            Parameter values (n_samples, n_parameters)
            
        Returns:
        --------
        np.ndarray
            Predicted temperature curves (n_samples, n_timepoints)
        """
        # Get FPCA coefficients
        fpca_coeffs, fpca_uncertainties = self.predict_fpca_coefficients(X)
        
        # Reconstruct curves
        curves = []
        for coeffs in fpca_coeffs:
            curve = reconstruct_curve_from_fpca(coeffs, self.fpca_model)
            curves.append(curve)
        
        # Compute curve-level uncertainties
        curve_uncertainties = self.predict_curve_uncertainty(fpca_uncertainties)
        
        return np.array(curves), fpca_coeffs, fpca_uncertainties, curve_uncertainties
    
    def predict_curve_uncertainty(self, fpca_uncertainties):
        """
        Compute curve-level uncertainty from FPCA coefficient uncertainties.
        
        Parameters:
        -----------
        fpca_uncertainties : np.ndarray (n_samples, n_components)
            GP uncertainties for each FPCA component
            
        Returns:
        --------
        np.ndarray (n_samples, n_timepoints)
            Curve-level uncertainties at each time point
        """
        eigenfunctions = self.fpca_model['eigenfunctions']  # (n_time, n_components)
        n_time = eigenfunctions.shape[0]
        
        curve_uncertainties = []
        for i in range(fpca_uncertainties.shape[0]):
            # Diagonal matrix of component variances
            Σ_coeffs = np.diag(fpca_uncertainties[i]**2)  # (n_comp, n_comp)
            
            # Sandwich formula: Φ × Σ_coeffs × Φᵀ
            Σ_curve = eigenfunctions @ Σ_coeffs @ eigenfunctions.T  # (n_time, n_time)
            
            # Extract diagonal (variances) and take square root
            σ_curve = np.sqrt(np.diag(Σ_curve))  # (n_time,)
            curve_uncertainties.append(σ_curve)
        
        return np.array(curve_uncertainties)
    
    def check_fpca_independence(self, test_samples, n_samples=100):
        """
        Check for independence between FPCA coefficients by computing correlations.
        
        Parameters:
        -----------
        test_samples : np.ndarray
            Parameter samples to test with
        n_samples : int
            Number of samples to use for correlation analysis
            
        Returns:
        --------
        dict
            Dictionary containing correlation matrices and independence metrics
        """
        print(f"\nChecking FPCA coefficient independence with {n_samples} samples...")
        
        # Use a subset of test samples if we have more than needed
        if len(test_samples) > n_samples:
            indices = np.random.choice(len(test_samples), n_samples, replace=False)
            samples_subset = test_samples[indices]
        else:
            samples_subset = test_samples
        
        # Get predictions
        fpca_coeffs, fpca_uncertainties = self.predict_fpca_coefficients(samples_subset)
        
        # Compute correlation matrices
        coeff_corr = np.corrcoef(fpca_coeffs.T)  # (n_components, n_components)
        
        uncertainty_corr = np.corrcoef(fpca_uncertainties.T)  # (n_components, n_components)
        
        # Extract off-diagonal elements (these should be close to zero for independence)
        n_components = fpca_coeffs.shape[1]
        coeff_off_diag = coeff_corr[np.triu_indices(n_components, k=1)]
        uncertainty_off_diag = uncertainty_corr[np.triu_indices(n_components, k=1)]
        
        # Compute independence metrics
        coeff_max_corr = np.max(np.abs(coeff_off_diag))
        coeff_mean_corr = np.mean(np.abs(coeff_off_diag))
        uncertainty_max_corr = np.max(np.abs(uncertainty_off_diag))
        uncertainty_mean_corr = np.mean(np.abs(uncertainty_off_diag))
        
        # Print results
        print(f"FPCA Coefficient Independence Check:")
        print(f"  Coefficient correlations:")
        print(f"    Max absolute correlation: {coeff_max_corr:.4f}")
        print(f"    Mean absolute correlation: {coeff_mean_corr:.4f}")
        print(f"  Uncertainty correlations:")
        print(f"    Max absolute correlation: {uncertainty_max_corr:.4f}")
        print(f"    Mean absolute correlation: {uncertainty_mean_corr:.4f}")
        
        # Independence assessment
        independence_threshold = 0.1  # Threshold for "practically independent"
        coeff_independent = coeff_max_corr < independence_threshold
        uncertainty_independent = uncertainty_max_corr < independence_threshold
        
        print(f"  Independence assessment (threshold: {independence_threshold}):")
        print(f"    Coefficients independent: {coeff_independent}")
        print(f"    Uncertainties independent: {uncertainty_independent}")
        
        if not coeff_independent:
            print(f"    WARNING: Strong correlations detected in coefficients!")
            print(f"    Largest correlation: {coeff_max_corr:.4f}")
        
        if not uncertainty_independent:
            print(f"    WARNING: Strong correlations detected in uncertainties!")
            print(f"    Largest correlation: {uncertainty_max_corr:.4f}")
        
        return {
            'coeff_correlation_matrix': coeff_corr,
            'uncertainty_correlation_matrix': uncertainty_corr,
            'coeff_max_correlation': coeff_max_corr,
            'coeff_mean_correlation': coeff_mean_corr,
            'uncertainty_max_correlation': uncertainty_max_corr,
            'uncertainty_mean_correlation': uncertainty_mean_corr,
            'coeff_independent': coeff_independent,
            'uncertainty_independent': uncertainty_independent,
            'test_samples_used': samples_subset
        }
    
    def check_fpca_orthogonality(self):
        """
        Check FPCA orthogonality by re-projecting training data and computing correlations.
        The correlation matrix should be essentially zero off-diagonal if FPCA is properly centered.
        """
        print(f"\nChecking FPCA orthogonality with training data...")
        
        # Get training scores from the FPCA model
        training_scores = self.fpca_model['training_scores']  # (n_samples, n_components)
        
        print(f"Training scores shape: {training_scores.shape}")
        
        # Compute correlation matrix of training scores
        training_corr = np.corrcoef(training_scores.T)  # (n_components, n_components)
        
        # Extract off-diagonal elements
        n_components = training_scores.shape[1]
        off_diag = training_corr[np.triu_indices(n_components, k=1)]
        
        # Compute orthogonality metrics
        max_corr = np.max(np.abs(off_diag))
        mean_corr = np.mean(np.abs(off_diag))
        std_corr = np.std(off_diag)
        
        # Print results
        print(f"FPCA Orthogonality Check (Training Data):")
        print(f"  Correlation matrix shape: {training_corr.shape}")
        print(f"  Max absolute off-diagonal correlation: {max_corr:.6f}")
        print(f"  Mean absolute off-diagonal correlation: {mean_corr:.6f}")
        print(f"  Std of off-diagonal correlations: {std_corr:.6f}")
        
        # Orthogonality assessment
        orthogonality_threshold = 1e-6  # Very strict threshold for numerical orthogonality
        is_orthogonal = max_corr < orthogonality_threshold
        
        print(f"  Orthogonality assessment (threshold: {orthogonality_threshold}):")
        print(f"    FPCA scores orthogonal: {is_orthogonal}")
        
        if not is_orthogonal:
            print(f"    WARNING: FPCA scores are NOT orthogonal!")
            print(f"    Largest off-diagonal correlation: {max_corr:.6f}")
            print(f"    This suggests a problem with centering or eigenfunction computation")
            
            # Show the full correlation matrix for debugging
            print(f"  Full correlation matrix:")
            for i in range(n_components):
                row_str = "  ".join([f"{training_corr[i,j]:8.3e}" for j in range(n_components)])
                print(f"    PC{i+1}: {row_str}")
        else:
            print(f"    SUCCESS: FPCA scores are properly orthogonal")
        
        # Check if diagonal elements are close to 1.0 (as expected for correlation matrix)
        diag_elements = np.diag(training_corr)
        diag_deviation = np.max(np.abs(diag_elements - 1.0))
        print(f"  Diagonal elements deviation from 1.0: {diag_deviation:.6f}")
        
        return {
            'correlation_matrix': training_corr,
            'max_off_diagonal_correlation': max_corr,
            'mean_off_diagonal_correlation': mean_corr,
            'std_off_diagonal_correlation': std_corr,
            'is_orthogonal': is_orthogonal,
            'diagonal_deviation': diag_deviation,
            'training_scores': training_scores
        }
    
    def validate_parameters(self, X):
        """
        Validate that parameters are within defined ranges.
        
        Parameters:
        -----------
        X : np.ndarray
            Parameter values to validate
            
        Returns:
        --------
        bool
            True if all parameters are within ranges
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        for i, (param_name, (min_val, max_val)) in enumerate(self.param_ranges.items()):
            if np.any(X[:, i] < min_val) or np.any(X[:, i] > max_val):
                return False
        return True
    
    def save_model(self, filepath):
        """
        Save the surrogate model to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        model_data = {
            'fpca_model': self.fpca_model,
            'gps': self.gps,
            'scaler': self.scaler,
            'parameter_names': self.parameter_names,
            'param_ranges': self.param_ranges,
            'n_components': self.n_components,
            'n_parameters': self.n_parameters,
            't_final': self.t_final,
            'num_steps': self.num_steps
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Surrogate model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load a surrogate model from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
            
        Returns:
        --------
        FullSurrogateModel
            Loaded surrogate model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Get timing information and **ensure** the number of steps matches the
        # length of the FPCA eigenfunctions (i.e. the curve length).  This
        # avoids inconsistencies where *num_steps* was copied from the YAML
        # config but the FPCA was built at a different resolution.

        fpca_model = model_data['fpca_model']

        # Total simulated time – fall back to stored value or a sensible default
        t_final = fpca_model.get('t_final', model_data.get('t_final', 8.5e-6))

        # Derive *num_steps* directly from the eigenfunction array to guarantee
        # consistency with reconstructed curves.  If the key exists but is
        # inconsistent we silently override it.
        n_time_from_fpca = fpca_model['eigenfunctions'].shape[0]
        num_steps = int(n_time_from_fpca)

        # Optionally warn if the stored num_steps disagrees (helps debugging)
        stored_num_steps = model_data.get('num_steps', None)
        if stored_num_steps is not None and stored_num_steps != num_steps:
            print(
                f"Warning: stored num_steps={stored_num_steps} does not match "
                f"FPCA eigenfunction length ({num_steps}). Using {num_steps}."
            )
        
        return cls(
            fpca_model=fpca_model,
            gps=model_data['gps'],
            scaler=model_data['scaler'],
            parameter_names=model_data['parameter_names'],
            param_ranges=model_data['param_ranges'],
            t_final=t_final,
            num_steps=num_steps
        )

def create_gp_model(kernel_type='rbf', n_dimensions=None):
    """
    Create a Gaussian Process model with specified kernel.
    
    Parameters:
    -----------
    kernel_type : str
        Type of kernel to use ('rbf' or 'matern')
    n_dimensions : int
        Number of input dimensions for the kernel
    """
    if n_dimensions is None:
        raise ValueError("n_dimensions must be specified")
    
    if kernel_type == 'rbf':
        kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(n_dimensions)) + WhiteKernel(noise_level=1e-6)
    elif kernel_type == 'matern':
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(n_dimensions), nu=1.5) + WhiteKernel(noise_level=1e-6)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-10,  # Reduced since WhiteKernel handles noise
        n_restarts_optimizer=10,
        random_state=42
    )
    
    return gp

def train_surrogate_model(input_path="outputs/edmund1/training_data_fpca_int_ins_match.npz", 
                         fpca_model_path="outputs/edmund1/fpca_model_int_ins_match.npz",
                         output_path="outputs/edmund1/full_surrogate_model_int_ins_match.pkl",
                         training_config="configs/config_5_materials.yaml",
                         test_fraction: float = 0.2,
                         random_state: int = 42):
    """
    Train the full surrogate model using all training data.
    
    Parameters:
    -----------
    input_path : str
        Path to the training data file
    fpca_model_path : str
        Path to the FPCA model file
    output_path : str
        Path to save the surrogate model
    training_config : str
        Path to the config file used to generate training data
    test_fraction : float
        Fraction of data to use for testing
    random_state : int
        Random state for train/test split
    
    Returns:
    --------
    tuple
        (FullSurrogateModel, training_metrics, test_metrics)
    """
    print("=" * 60)
    print("TRAINING FULL SURROGATE MODEL")
    print("=" * 60)
    
    # Load training data
    print("\n1. Loading training data...")
    recast_data = load_recast_training_data(input_path)
    X_full = recast_data['parameters']
    y_full = recast_data['fpca_scores']
    parameter_names = recast_data['parameter_names']
    print(f"Training data: {len(X_full)} samples, {X_full.shape[1]} parameters, {y_full.shape[1]} FPCA components")

    # Load the training config to get timing parameters
    print(f"\n1.5. Loading training config: {training_config}")
    import yaml
    with open(training_config, 'r') as f:
        training_config_data = yaml.safe_load(f)
    
    t_final = float(training_config_data['timing']['t_final'])
    num_steps = int(training_config_data['timing']['num_steps'])
    print(f"Training config timing: t_final={t_final:.2e}s, num_steps={num_steps}")

    # Split into train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=test_fraction, random_state=random_state
    )
    print(f"Train / test split: {len(X_train)} train | {len(X_test)} test (fraction {test_fraction})")

    # To calculate the R² on the full curves, we need the original curves for the test set.
    # We can reconstruct them from the y_test (FPCA scores) and the FPCA model.
    fpca_model = load_fpca_model(fpca_model_path)
    
    # Reconstruct the original curves for the test set
    true_curves_test = np.array([reconstruct_curve_from_fpca(coeffs, fpca_model) for coeffs in y_test])

    # Get the mean curve from the FPCA model
    mean_curve = fpca_model['mean_curve']

    # Load FPCA model
    print("\n2. Loading FPCA model...")
    fpca_model = load_fpca_model(fpca_model_path)

    # Scale inputs based on training set only
    print("\n3. Scaling input parameters...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Get parameter ranges from Edmund config instead of hardcoded values
    from analysis.config_utils import get_param_defs_from_config
    param_defs = get_param_defs_from_config(config_path="configs/distributions_edmund.yaml")
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

    # Train GP for each FPCA component
    print("\n4. Training GP models for each FPCA component...")
    gps = []
    training_metrics = []
    test_metrics = []


    for i in range(fpca_model['n_components']):
        print(f"\nTraining GP for PC{i+1}...")
        gp = create_gp_model('rbf', n_dimensions=X_train_scaled.shape[1])
        gp.fit(X_train_scaled, y_train[:, i])

        # Evaluate training performance
        y_pred_train, _ = gp.predict(X_train_scaled, return_std=True)
        train_r2 = r2_score(y_train[:, i], y_pred_train)
        train_rmse = np.sqrt(mean_squared_error(y_train[:, i], y_pred_train))

        # Evaluate test performance and capture predictive std-devs
        y_pred_test, y_std_test = gp.predict(X_test_scaled, return_std=True)
        test_r2 = r2_score(y_test[:, i], y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred_test))



        print(f"  Training R²: {train_r2:.6f} | RMSE: {train_rmse:.6f}")
        print(f"  Test     R²: {test_r2:.6f} | RMSE: {test_rmse:.6f}")
        print(f"  Optimized kernel: {gp.kernel_}")

        gps.append(gp)
        training_metrics.append({'r2': train_r2, 'rmse': train_rmse, 'kernel': gp.kernel_})
        test_metrics.append({'r2': test_r2, 'rmse': test_rmse})

    # Create surrogate model with the GPs trained on the training split
    print("\n5. Creating surrogate model...")
    
    # Get the correct number of time steps from the FPCA model's eigenfunctions
    # This ensures the surrogate model matches the actual resolution of the training data
    correct_num_steps = fpca_model['eigenfunctions'].shape[0]
    print(f"FPCA model eigenfunctions shape: {fpca_model['eigenfunctions'].shape}")
    print(f"Using {correct_num_steps} time steps (derived from FPCA model)")
    
    # Use timing parameters from the training config (not defaults)
    # Store them in the FPCA model for future reference
    fpca_model['t_final'] = t_final
    fpca_model['num_steps'] = num_steps
    
    print(f"Using timing parameters from training config: t_final={t_final:.2e}s, num_steps={num_steps}")
    
    surrogate = FullSurrogateModel(
        fpca_model=fpca_model,
        gps=gps,
        scaler=scaler,
        parameter_names=parameter_names,
        param_ranges=param_ranges,
        t_final=t_final,
        num_steps=num_steps
    )

    # Predict curves for the test set
    predicted_curves_test, _, _, _ = surrogate.predict_temperature_curves(X_test)
    
    # Calculate the overall R² score on the full curves
    # We need to flatten the arrays to treat all time points as independent samples for the R² calculation
    r2_full_curves = r2_score(true_curves_test.flatten(), predicted_curves_test.flatten())
    print(f"\nOverall R² score on full test curves: {r2_full_curves:.6f}")

    # Calculate R² on residuals (subtracting the mean curve)
    true_residuals = true_curves_test - mean_curve
    predicted_residuals = predicted_curves_test - mean_curve
    r2_residuals = r2_score(true_residuals.flatten(), predicted_residuals.flatten())
    print(f"Overall R² score on residuals (mean-subtracted): {r2_residuals:.6f}")


    # Save the model
    print("\n6. Saving surrogate model...")
    surrogate.save_model(output_path)

    # Summary
    print(f"\n{'='*60}")
    print("SURROGATE MODEL TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"Input parameters: {X_full.shape[1]}")
    print(f"FPCA components: {y_full.shape[1]}")

    print("\nPerformance per component:")
    for i, (tr, te) in enumerate(zip(training_metrics, test_metrics)):
        print(f"  PC{i+1}: Train R²={tr['r2']:.4f}, RMSE={tr['rmse']:.4e} | "
              f"Test R²={te['r2']:.4f}, RMSE={te['rmse']:.4e}")

    print(f"\nOverall R² on reconstructed test curves: {r2_full_curves:.6f}")
    print(f"Overall R² on residuals (mean-subtracted): {r2_residuals:.6f}")


    return surrogate, training_metrics, test_metrics

def main():
    """
    Main function to create the full surrogate model.
    """
    print("Creating full surrogate GP model...")
    
    # Train the surrogate model
    surrogate, training_metrics, test_metrics = train_surrogate_model(
        training_config="configs/config_5_materials.yaml"
    )
    
    # Final summary
    print(f"\n{'='*60}")
    print("FULL SURROGATE MODEL COMPLETED!")
    print(f"{'='*60}")
    print(f"Model saved to: outputs/edmund1/full_surrogate_model_int_ins_match.pkl")
    print(f"All training metrics R² > 0.99: {all(m['r2'] > 0.99 for m in training_metrics)}")
    print(f"All test metrics R² > 0.99: {all(m['r2'] > 0.99 for m in test_metrics)}")
    print(f"Model ready for use in UQ analysis!")
    
    return surrogate, training_metrics, test_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a full surrogate GP model")
    parser.add_argument("--input_path", type=str, default="outputs/edmund1/training_data_fpca_int_ins_match.npz", help="Path to training data")
    parser.add_argument("--fpca_model_path", type=str, default="outputs/edmund1/fpca_model_int_ins_match.npz", help="Path to FPCA model")
    parser.add_argument("--output_path", type=str, default="outputs/edmund1/full_surrogate_model_int_ins_match.pkl", help="Path to save the surrogate model")
    parser.add_argument("--training_config", type=str, default="configs/config_5_materials.yaml", help="Config file used to generate training data")
    parser.add_argument("--test_fraction", type=float, default=0.2, help="Fraction of data to use for testing")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for train/test split")
    args = parser.parse_args()
    
    print(f"Using input path: {args.input_path}")
    print(f"Using FPCA model path: {args.fpca_model_path}")
    print(f"Using output path: {args.output_path}")
    print(f"Using training config: {args.training_config}")
    
    # Train the surrogate model
    surrogate, training_metrics, test_metrics = train_surrogate_model(
        input_path=args.input_path,
        fpca_model_path=args.fpca_model_path,
        output_path=args.output_path,
        training_config=args.training_config,
        test_fraction=args.test_fraction,
        random_state=args.random_state
    )
    
    # Final summary
    print(f"\n{'='*60}")
    print("FULL SURROGATE MODEL COMPLETED!")
    print(f"{'='*60}")
    print(f"Model saved to: {args.output_path}")
    print(f"All training metrics R² > 0.99: {all(m['r2'] > 0.99 for m in training_metrics)}")
    print(f"All test metrics R² > 0.99: {all(m['r2'] > 0.99 for m in test_metrics)}")
    print(f"Model ready for use in UQ analysis!") 