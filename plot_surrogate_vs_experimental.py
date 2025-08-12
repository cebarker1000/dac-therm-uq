#!/usr/bin/env python3
"""
Simple script to run surrogate model with specific parameters and plot against experimental data.
Useful for manual fine-tuning of thermal conductivity parameters.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from train_surrogate_models import FullSurrogateModel
import warnings

warnings.filterwarnings('ignore')

class SurrogatePlotter:
    """
    Simple class to run surrogate model and plot against experimental data.
    """
    
    def __init__(self, surrogate_model_path="outputs/full_surrogate_model.pkl"):
        """
        Initialize the plotter.
        
        Parameters:
        -----------
        surrogate_model_path : str
            Path to the saved surrogate model
        """
        print("Loading surrogate model...")
        self.surrogate = FullSurrogateModel.load_model(surrogate_model_path)
        
        # The surrogate model now has the correct time grid
        self.sim_time_grid = self.surrogate.time_grid
        self.sim_t_final = self.surrogate.t_final
        self.sim_num_steps = self.surrogate.num_steps
        
        print(f"Surrogate model loaded with {self.surrogate.n_components} FPCA components")
        print(f"Parameter names: {self.surrogate.parameter_names}")
        print(f"Simulation time grid: {self.sim_num_steps} steps from 0 to {self.sim_t_final:.2e} s")
    
    def load_geballe_data(self, data_file="data/experimental/geballe_heat_data.csv"):
        """
        Load Geballe heat data from CSV file.
        
        Returns:
        --------
        tuple
            (time_array, temp_array, oside_array) - experimental time, temp, and oside data
        """
        print(f"Loading Geballe heat data from {data_file}...")
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Geballe data file not found: {data_file}")
        
        try:
            # Load CSV data
            df = pd.read_csv(data_file)
            
            # Check if required columns exist
            required_cols = ['time', 'temp', 'oside']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' not found in Geballe data. Available columns: {list(df.columns)}")
            
            # Extract data
            time_array = df['time'].values
            temp_array = df['temp'].values
            oside_array = df['oside'].values
            
            # Remove any NaN values
            valid_mask = ~(np.isnan(time_array) | np.isnan(temp_array) | np.isnan(oside_array))
            time_array = time_array[valid_mask]
            temp_array = temp_array[valid_mask]
            oside_array = oside_array[valid_mask]
            
            print(f"Loaded {len(time_array)} data points")
            print(f"Time range: {time_array[0]:.2e} to {time_array[-1]:.2e} s")
            print(f"Temp range: {temp_array.min():.2f} to {temp_array.max():.2f} K")
            print(f"Oside range: {oside_array.min():.2f} to {oside_array.max():.2f} K")
            
            return time_array, temp_array, oside_array
            
        except Exception as e:
            print(f"Error loading Geballe data: {e}")
            raise
    
    def align_experimental_data(self, exp_time, exp_data, method='linear'):
        """
        Align experimental data to simulation time grid using interpolation.
        
        Parameters:
        -----------
        exp_time : np.ndarray
            Experimental time points
        exp_data : np.ndarray
            Experimental data points
        method : str
            Interpolation method ('linear', 'cubic', 'nearest')
            
        Returns:
        --------
        np.ndarray
            Interpolated data on simulation time grid
        """
        print(f"Aligning experimental data to simulation time grid using {method} interpolation...")
        print(f"Experimental time range: {exp_time[0]:.2e} to {exp_time[-1]:.2e} s")
        print(f"Simulation time range: {self.sim_time_grid[0]:.2e} to {self.sim_time_grid[-1]:.2e} s")
        
        # Handle time range mismatch
        # If experimental data starts after simulation start, pad with initial value
        if exp_time[0] > self.sim_time_grid[0]:
            print(f"Experimental data starts after simulation start. Padding early times with initial value.")
            # Find simulation time points before experimental data starts
            early_mask = self.sim_time_grid < exp_time[0]
            late_mask = self.sim_time_grid >= exp_time[0]
            
            # Create interpolation function for the overlapping range
            interp_func = interp1d(exp_time, exp_data, kind=method, 
                                  bounds_error=False, fill_value=(exp_data[0], exp_data[-1]))
            
            # Interpolate the overlapping range
            aligned_data = np.zeros_like(self.sim_time_grid)
            aligned_data[early_mask] = exp_data[0]  # Use initial value for early times
            aligned_data[late_mask] = interp_func(self.sim_time_grid[late_mask])
            
        else:
            # Standard interpolation if time ranges overlap properly
            interp_func = interp1d(exp_time, exp_data, kind=method, 
                                  bounds_error=False, fill_value=exp_data[0])
            aligned_data = interp_func(self.sim_time_grid)
        
        # Check for any remaining NaN values
        if np.any(np.isnan(aligned_data)):
            print("Warning: NaN values detected after interpolation. Using nearest neighbor for extrapolation.")
            # Fall back to nearest neighbor for extrapolation
            interp_func_nearest = interp1d(exp_time, exp_data, kind='nearest', 
                                         bounds_error=False, fill_value=exp_data[0])
            aligned_data = interp_func_nearest(self.sim_time_grid)
        
        print(f"Interpolated to {len(aligned_data)} time points")
        print(f"Aligned data range: {aligned_data.min():.2f} to {aligned_data.max():.2f}")
        
        return aligned_data
    
    def normalize_experimental_data(self, data_array, reference_array=None):
        """
        Normalize experimental data using the same normalization as the surrogate training data.
        
        Parameters:
        -----------
        data_array : np.ndarray
            Data array to normalize (temp or oside)
        reference_array : np.ndarray, optional
            Reference array for normalization (if None, uses data_array itself)
            
        Returns:
        --------
        np.ndarray
            Normalized data array
        """
        if reference_array is None:
            reference_array = data_array
        
        # Use the same normalization as training data: subtract initial value and divide by max excursion
        # This matches the surrogate training normalization
        initial_value = data_array[0]  # Start from initial value
        max_excursion = np.max(reference_array) - np.min(reference_array)
        normalized = (data_array - initial_value) / max_excursion
        
        return normalized
    
    def get_default_parameters(self, use_mean_values=True):
        """
        Get default parameter values for the surrogate model.
        
        Parameters:
        -----------
        use_mean_values : bool
            If True, use mean values from training data. If False, use center values from parameter ranges.
            
        Returns:
        --------
        dict
            Dictionary of parameter names and their default values
        """
        if use_mean_values:
            # Load training data to get mean values
            try:
                from analysis.uq_wrapper import load_recast_training_data
                training_data = load_recast_training_data("outputs/training_data_fpca.npz")
                parameters = training_data['parameters']
                param_names = training_data['parameter_names']
                
                # Calculate mean values
                mean_values = np.mean(parameters, axis=0)
                defaults = dict(zip(param_names, mean_values))
                
                print("Using mean values from training data:")
                for name, value in defaults.items():
                    print(f"  {name}: {value:.6e}")
                
                return defaults
                
            except Exception as e:
                print(f"Warning: Could not load training data for mean values: {e}")
                print("Falling back to center values from parameter ranges...")
                use_mean_values = False
        
        if not use_mean_values:
            # Use center values from parameter ranges
            defaults = {
                "d_sample": 1.84e-6,      # Center of lognormal distribution
                "rho_cv_sample": 2764828,  # Center of lognormal distribution
                "rho_cv_coupler": 3445520, # Center of lognormal distribution
                "rho_cv_ins": 2764828,     # Center of lognormal distribution
                "d_coupler": 6.2e-8,       # Center of lognormal distribution
                "d_ins_oside": 3.2e-6,     # Center of lognormal distribution
                "d_ins_pside": 6.3e-6,     # Center of lognormal distribution
                "fwhm": 13.2e-6,             # Center of lognormal distribution
                "k_sample": 3.8,           # Center of uniform range (2.8, 4.8)
                "k_ins": 10.0,             # Center of uniform range (7.0, 13.0)
                "k_coupler": 350.0,        # Center of uniform range (300, 400)
            }
            
            print("Using center values from parameter ranges:")
            for name, value in defaults.items():
                print(f"  {name}: {value:.6e}")
            
            return defaults
    
    def run_surrogate_with_params(self, k_sample, k_ins=10.0, k_coupler=352.0, use_mean_values=True):
        """
        Run surrogate model with specific thermal conductivity parameters.
        
        Parameters:
        -----------
        k_sample : float
            Sample thermal conductivity (W/m/K)
        k_ins : float
            Insulator thermal conductivity (W/m/K)
        k_coupler : float
            Coupler thermal conductivity (W/m/K)
        use_mean_values : bool
            Whether to use mean values from training data for other parameters
            
        Returns:
        --------
        tuple
            (predicted_curve, fpca_scores, uncertainties) - surrogate model outputs
        """
        # Get default values for all parameters
        default_params = self.get_default_parameters(use_mean_values)
        
        # Update with the specified k values
        default_params['k_sample'] = k_sample
        default_params['k_ins'] = k_ins
        default_params['k_coupler'] = k_coupler
        
        # Create parameter vector in the correct order
        param_vector = np.array([default_params[name] for name in self.surrogate.parameter_names])
        
        print(f"\nRunning surrogate model with parameters:")
        print(f"  k_sample: {k_sample:.2f} W/m/K")
        print(f"  k_ins: {k_ins:.2f} W/m/K")
        print(f"  k_coupler: {k_coupler:.2f} W/m/K")
        print(f"  Other parameters: {'mean values' if use_mean_values else 'center values'}")
        
        # Run surrogate model
        predicted_curve, fpca_scores, uncertainties, _ = self.surrogate.predict_temperature_curves(param_vector.reshape(1, -1))
        
        # Extract single curve (remove batch dimension)
        predicted_curve = predicted_curve[0]
        fpca_scores = fpca_scores[0]
        uncertainties = uncertainties[0]
        
        print(f"Predicted curve range: {predicted_curve.min():.4f} to {predicted_curve.max():.4f}")
        print(f"FPCA scores: {fpca_scores}")
        print(f"Uncertainties: {uncertainties}")
        
        return predicted_curve, fpca_scores, uncertainties
    
    def plot_oside_comparison(self, exp_time, exp_temp, exp_oside, predicted_curve, 
                             k_sample, k_ins, k_coupler):
        """
        Plot surrogate prediction against oside experimental data.
        
        Parameters:
        -----------
        exp_time : np.ndarray
            Experimental time points
        exp_temp : np.ndarray
            Experimental temp data (used for normalization reference)
        exp_oside : np.ndarray
            Experimental oside data
        predicted_curve : np.ndarray
            Surrogate model prediction
        k_sample, k_ins, k_coupler : float
            Thermal conductivity parameters used
        """
        print(f"\nCreating comparison plot for oside data...")
        
        # Align oside experimental data to simulation time grid
        aligned_oside = self.align_experimental_data(exp_time, exp_oside)
        
        # Normalize oside data using temp data as reference (same as surrogate training)
        normalized_oside = self.normalize_experimental_data(aligned_oside, exp_temp)
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((normalized_oside - predicted_curve)**2))
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Normalized comparison
        ax1.plot(self.sim_time_grid, normalized_oside, 'b-', linewidth=2, label='Experimental (oside)')
        ax1.plot(self.sim_time_grid, predicted_curve, 'r--', linewidth=2, label='Surrogate Prediction')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Normalized Temperature')
        ax1.set_title(f'Oside Comparison (RMSE: {rmse:.6f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Original scale comparison
        # Denormalize predicted curve to original scale
        oside_initial = exp_oside[0]
        temp_max_excursion = np.max(exp_temp) - np.min(exp_temp)
        denorm_predicted = predicted_curve * temp_max_excursion + oside_initial
        denorm_oside = normalized_oside * temp_max_excursion + oside_initial
        
        ax2.plot(self.sim_time_grid, denorm_oside, 'b-', linewidth=2, label='Experimental (oside)')
        ax2.plot(self.sim_time_grid, denorm_predicted, 'r--', linewidth=2, label='Surrogate Prediction')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Temperature (K)')
        ax2.set_title('Oside - Original Scale')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add parameter info
        param_text = f'k_sample: {k_sample:.2f} W/m/K\nk_ins: {k_ins:.2f} W/m/K\nk_coupler: {k_coupler:.2f} W/m/K'
        fig.text(0.02, 0.02, param_text, fontsize=10, verticalalignment='bottom', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'outputs/surrogate_vs_oside_k{k_sample:.1f}.png', dpi=200, bbox_inches='tight')
        plt.show()
        
        print(f"Plot saved as outputs/surrogate_vs_oside_k{k_sample:.1f}.png")
        print(f"RMSE: {rmse:.6f}")
        
        return rmse
    


def main():
    """
    Example usage - modify the k_sample value to fine-tune the fit.
    """
    print("Surrogate Model vs Experimental Data Plotter")
    print("=" * 50)
    
    # Initialize plotter
    plotter = SurrogatePlotter()
    
    # Load experimental data
    try:
        exp_time, exp_temp, exp_oside = plotter.load_geballe_data()
    except Exception as e:
        print(f"Error loading experimental data: {e}")
        return
    
    # Set your thermal conductivity parameters here
    # Modify these values to fine-tune the fit
    k_sample = 3.4  # Try different values: 3.0, 3.5, 4.0, 4.5, etc.
    k_ins = 12    # Fixed
    k_coupler = 10.0  # Fixed
    
    print(f"\nTesting parameters: k_sample={k_sample}, k_ins={k_ins}, k_coupler={k_coupler}")
    
    # Run surrogate model with mean values for other parameters
    predicted_curve, fpca_scores, uncertainties = plotter.run_surrogate_with_params(
        k_sample, k_ins, k_coupler, use_mean_values=True
    )
    
    # Plot comparison against oside data only
    rmse = plotter.plot_oside_comparison(
        exp_time, exp_temp, exp_oside, predicted_curve, k_sample, k_ins, k_coupler
    )
    
    print(f"\nSummary:")
    print(f"  k_sample: {k_sample:.2f} W/m/K")
    print(f"  RMSE vs oside: {rmse:.6f}")
    print(f"\nTo try different k_sample values, modify the k_sample variable in main()")

if __name__ == "__main__":
    main() 