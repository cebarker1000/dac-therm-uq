"""
This module provides a set of wrapper functions for running the heat flow
simulation within a UQpy-based uncertainty quantification (UQ) workflow.

It includes functions for:

- Creating a simulation configuration from a parameter sample.
- Running single and batch simulations using the `OptimizedSimulationEngine`.
- Extracting and processing the simulation results.
- Building a Functional Principal Component Analysis (FPCA) model from the
  simulation results.
- Projecting the simulation results onto the FPCA basis to generate a
  low-dimensional representation of the data.
"""

import yaml
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from core.simulation_engine import OptimizedSimulationEngine
from scipy import linalg

import sys
sys.path.append("..")



def load_base_config(config_path: str = "configs/config_5_materials.yaml") -> Dict[str, Any]:
    """
    Load the base configuration from YAML file.
    
    Parameters:
    -----------
    config_path : str
        Path to the base configuration file
        
    Returns:
    --------
    Dict[str, Any]
        Base configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_config_from_sample(sample: np.ndarray, 
                             param_defs: List[Dict[str, Any]],
                             param_mapping: Dict[str, List[tuple]],
                             base_config: Optional[Dict[str, Any]] = None,
                             config_path: str = "configs/config_5_materials.yaml") -> Dict[str, Any]:
    """
    Convert a parameter sample to a simulation configuration.
    
    Parameters:
    -----------
    sample : np.ndarray
        Array of parameter values in the same order as param_defs
    param_defs : List[Dict[str, Any]]
        List of parameter definitions
    param_mapping : Dict[str, List[tuple]]
        Mapping from parameter names to config locations
    base_config : Dict[str, Any], optional
        Base configuration to modify. If None, loads from config_path.
    config_path : str
        Path to base configuration file (used if base_config is None)
        
    Returns:
    --------
    Dict[str, Any]
        Configuration dictionary with updated parameter values
    """
    if base_config is None:
        config = load_base_config(config_path)
    else:
        config = base_config.copy()
    
    # Get parameter names in order
    param_names = [p["name"] for p in param_defs]
    
    # Update configuration with sample values
    for name, value in zip(param_names, sample):
        if name in param_mapping:
            # Apply parameter to all mapped locations
            for mapping in param_mapping[name]:
                # Navigate to the target location in config
                current = config
                for key in mapping[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                # Set the value
                current[mapping[-1]] = float(value)
    
    return config


def run_single_simulation(sample: np.ndarray, 
                         param_defs: List[Dict[str, Any]],
                         param_mapping: Dict[str, List[tuple]],
                         simulation_index: int = 0,
                         base_config: Optional[Dict[str, Any]] = None,
                         config_path: str = "configs/config_5_materials.yaml",
                         suppress_print: bool = True) -> Dict[str, Any]:
    """
    Run a single simulation with the given parameter sample.
    
    Parameters:
    -----------
    sample : np.ndarray
        Array of parameter values
    param_defs : List[Dict[str, Any]]
        List of parameter definitions
    param_mapping : Dict[str, List[tuple]]
        Mapping from parameter names to config locations
    simulation_index : int
        Index for this simulation (for tracking purposes)
    base_config : Dict[str, Any], optional
        Base configuration to modify
    config_path : str
        Path to base configuration file
    suppress_print : bool
        Whether to suppress print output during simulation
        
    Returns:
    --------
    Dict[str, Any]
        Simulation results including watcher data and timing
    """
    # Create configuration from sample
    config = create_config_from_sample(sample, param_defs, param_mapping, base_config, config_path)
    
    # For minimal runs, we don't need actual folders since everything is in memory
    # But the constructor still requires these parameters
    mesh_folder = "/tmp"  # Dummy folder - not actually used in minimal mode
    output_folder = "/tmp"  # Dummy folder - not actually used in minimal mode
    
    try:
        # Create simulation engine
        engine = OptimizedSimulationEngine(config, mesh_folder, output_folder)
        
        # Run minimal simulation
        result = engine.run_minimal(suppress_print=suppress_print)
        
        # Add simulation metadata
        result['simulation_index'] = simulation_index
        result['parameters'] = dict(zip([p["name"] for p in param_defs], sample))
        
        return result
        
    except Exception as e:
        print(f"Simulation {simulation_index} failed: {e}")
        return {
            'simulation_index': simulation_index,
            'error': str(e),
            'parameters': dict(zip([p["name"] for p in param_defs], sample))
        }


def run_batch_simulations(samples: np.ndarray,
                         param_defs: List[Dict[str, Any]],
                         param_mapping: Dict[str, List[tuple]],
                         base_config: Optional[Dict[str, Any]] = None,
                         config_path: str = "configs/config_5_materials.yaml",
                         suppress_print: bool = True,
                         progress_callback=None) -> List[Dict[str, Any]]:
    """
    Run batch simulations for multiple parameter samples.
    
    Parameters:
    -----------
    samples : np.ndarray
        2D array where each row is a parameter sample
    param_defs : List[Dict[str, Any]]
        List of parameter definitions
    param_mapping : Dict[str, List[tuple]]
        Mapping from parameter names to config locations
    base_config : Dict[str, Any], optional
        Base configuration to modify
    config_path : str
        Path to base configuration file
    suppress_print : bool
        Whether to suppress print output during simulations
    progress_callback : callable, optional
        Callback function for progress reporting (called with current index, total)
        
    Returns:
    --------
    List[Dict[str, Any]]
        List of results for each simulation
    """
    results = []
    
    for i, sample in enumerate(samples):
        if progress_callback:
            progress_callback(i, len(samples))
        
        result = run_single_simulation(
            sample, param_defs, param_mapping, i, base_config, config_path, suppress_print
        )
        results.append(result)
    
    return results


def extract_oside_curves(results: List[Dict[str, Any]]) -> np.ndarray:
    """
    Extract oside temperature curves from simulation results.
    
    Parameters:
    -----------
    results : List[Dict[str, Any]]
        Results from batch simulations
        
    Returns:
    --------
    np.ndarray
        2D array where each row is a normalized oside temperature curve
    """
    curves = []
    successful_curves = []
    
    # First pass: collect all successful curves and find the maximum length
    max_length = 0
    for i, result in enumerate(results):
        if 'watcher_data' in result and 'oside' in result['watcher_data']:
            normalized_curve = result['watcher_data']['oside']['normalized']
            curve_length = len(normalized_curve)
            max_length = max(max_length, curve_length)
            successful_curves.append(normalized_curve)
            print(f"Curve {i}: length = {curve_length}")
        else:
            print(f"Curve {i}: failed simulation or missing data")
    
    if not successful_curves:
        print("WARNING: No successful curves found!")
        return np.array([])
    
    print(f"Maximum curve length: {max_length}")
    print(f"Number of successful curves: {len(successful_curves)}")
    
    # Second pass: pad all curves to the same length
    for i, result in enumerate(results):
        if 'watcher_data' in result and 'oside' in result['watcher_data']:
            normalized_curve = result['watcher_data']['oside']['normalized']
            curve_length = len(normalized_curve)
            
            if curve_length < max_length:
                # Pad with the last value
                padded_curve = np.pad(normalized_curve, (0, max_length - curve_length), 
                                     mode='edge')
                curves.append(padded_curve)
                print(f"Curve {i}: padded from {curve_length} to {max_length}")
            else:
                curves.append(normalized_curve)
                print(f"Curve {i}: length {curve_length} (no padding needed)")
        else:
            # Handle failed simulations - fill with NaN
            curves.append(np.full(max_length, np.nan))
            print(f"Curve {i}: filled with NaN (failed simulation)")
    
    print(f"Final array shape: {len(curves)} curves, max length {max_length}")
    return np.array(curves)


def save_batch_results(results: List[Dict[str, Any]], 
                      param_defs: List[Dict[str, Any]],
                      output_file: str = "uq_batch_results.npz"):
    """
    Save batch results to a compressed numpy file.
    
    Parameters:
    -----------
    results : List[Dict[str, Any]]
        Results from batch simulations
    param_defs : List[Dict[str, Any]]
        List of parameter definitions
    output_file : str
        Output file path
    """
    # Extract oside curves
    print("Extracting oside curves...")
    oside_curves = extract_oside_curves(results)
    
    if len(oside_curves) == 0:
        print("ERROR: No oside curves extracted!")
        return
    
    print(f"Oside curves shape: {oside_curves.shape}")
    
    # Extract parameters
    param_names = [p["name"] for p in param_defs]
    parameters = np.array([result.get('parameters', {}) for result in results])
    param_array = np.array([[parameters[i].get(name, np.nan) for name in param_names] 
                           for i in range(len(parameters))])
    
    print(f"Parameter array shape: {param_array.shape}")
    
    # Extract timing information
    timing_data = []
    for result in results:
        if 'timing' in result:
            timing_data.append([
                result['timing'].get('total_loop_time', np.nan),
                result['timing'].get('avg_step_time', np.nan),
                result['timing'].get('num_steps', np.nan)
            ])
        else:
            timing_data.append([np.nan, np.nan, np.nan])
    
    # Save to file
    np.savez_compressed(
        output_file,
        oside_curves=oside_curves,
        parameters=param_array,
        parameter_names=param_names,
        timing=np.array(timing_data),
        simulation_indices=np.array([r.get('simulation_index', i) 
                                   for i, r in enumerate(results)])
    )
    print(f"UQ batch results saved to {output_file}")


def load_batch_results(input_file: str = "uq_batch_results.npz") -> Dict[str, np.ndarray]:
    """
    Load batch results from a compressed numpy file.
    
    Parameters:
    -----------
    input_file : str
        Input file path
    
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary with loaded data
    """
    data = np.load(input_file)
    return {key: data[key] for key in data.keys()}


def build_fpca_model(input_file: str = "outputs/uq_batch_results.npz", 
                    min_components: int = 4,
                    variance_threshold: float = 0.99) -> Dict[str, Any]:
    """
    Build a Functional PCA model from batch results.
    
    Parameters:
    -----------
    input_file : str
        Path to the .npz file containing batch results
    min_components : int
        Minimum number of components to use (default: 4)
    variance_threshold : float
        Minimum cumulative variance to explain (default: 0.99)
        
    Returns:
    --------
    Dict[str, Any]
        FPCA model containing:
        - mean_curve: mean temperature curve
        - eigenfunctions: principal component functions
        - eigenvalues: eigenvalues for each component
        - explained_variance: explained variance for each component
        - cumulative_variance: cumulative explained variance
        - n_components: number of components used
        - training_curves: original training curves
        - training_scores: scores for training curves
    """
    print(f"Building FPCA model from {input_file}...")
    
    # Load batch results
    data = load_batch_results(input_file)
    
    # Filter out failed simulations (those with NaN values)
    valid_mask = ~np.isnan(data['oside_curves']).any(axis=1)
    valid_curves = data['oside_curves'][valid_mask]
    
    print(f"Using {len(valid_curves)} valid curves out of {len(data['oside_curves'])}")
    
    # Center the data
    mean_curve = np.mean(valid_curves, axis=0)
    curves_centered = valid_curves - mean_curve
    
    # Compute covariance matrix
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
    
    # Determine number of components
    n_components_variance = int(np.argmax(cumulative_variance >= variance_threshold) + 1)
    n_components = max(min_components, n_components_variance)
    
    print(f"Variance threshold {variance_threshold} requires {n_components_variance} components")
    print(f"Using {n_components} components (min: {min_components})")
    print(f"Explained variance: {cumulative_variance[n_components-1]:.4f}")
    
    # Limit to selected number of components
    eigenvalues = eigenvalues[:n_components]
    eigenfunctions = eigenfunctions[:, :n_components]
    explained_variance = explained_variance[:n_components]
    cumulative_variance = cumulative_variance[:n_components]
    
    # Compute scores for training data
    training_scores = curves_centered @ eigenfunctions
    
    # Build model
    fpca_model = {
        'mean_curve': mean_curve,
        'eigenfunctions': eigenfunctions,
        'eigenvalues': eigenvalues,
        'explained_variance': explained_variance,
        'cumulative_variance': cumulative_variance,
        'n_components': n_components,
        'training_curves': valid_curves,
        'training_scores': training_scores,
        'parameter_names': data['parameter_names'],
        'training_parameters': data['parameters'][valid_mask]
    }
    
    print(f"FPCA model built successfully with {n_components} components")
    return fpca_model


def project_curve_to_fpca(curve: np.ndarray, fpca_model: Dict[str, Any]) -> np.ndarray:
    """
    Project a single temperature curve onto the FPCA basis.
    
    Parameters:
    -----------
    curve : np.ndarray
        Temperature curve to project (1D array)
    fpca_model : Dict[str, Any]
        FPCA model from build_fpca_model
        
    Returns:
    --------
    np.ndarray
        FPCA coefficients (scores) for the curve
    """
    # Center the curve using the model's mean
    curve_centered = curve - fpca_model['mean_curve']
    
    # Project onto eigenfunctions
    scores = curve_centered @ fpca_model['eigenfunctions']
    
    return scores


def reconstruct_curve_from_fpca(scores: np.ndarray, fpca_model: Dict[str, Any]) -> np.ndarray:
    """
    Reconstruct a temperature curve from FPCA coefficients.
    
    Parameters:
    -----------
    scores : np.ndarray
        FPCA coefficients (scores)
    fpca_model : Dict[str, Any]
        FPCA model from build_fpca_model
        
    Returns:
    --------
    np.ndarray
        Reconstructed temperature curve
    """
    # Use only the first n_components
    n_components = min(len(scores), fpca_model['n_components'])
    eigenfunctions = fpca_model['eigenfunctions'][:, :n_components]
    scores_limited = scores[:n_components]
    
    # Reconstruct
    reconstructed = scores_limited @ eigenfunctions.T + fpca_model['mean_curve']
    
    return reconstructed


def save_fpca_model(fpca_model: Dict[str, Any], output_file: str = "outputs/fpca_model.npz"):
    """
    Save FPCA model to a compressed numpy file.
    
    Parameters:
    -----------
    fpca_model : Dict[str, Any]
        FPCA model from build_fpca_model
    output_file : str
        Output file path
    """
    np.savez_compressed(
        output_file,
        mean_curve=fpca_model['mean_curve'],
        eigenfunctions=fpca_model['eigenfunctions'],
        eigenvalues=fpca_model['eigenvalues'],
        explained_variance=fpca_model['explained_variance'],
        cumulative_variance=fpca_model['cumulative_variance'],
        n_components=fpca_model['n_components'],
        training_scores=fpca_model['training_scores'],
        parameter_names=fpca_model['parameter_names'],
        training_parameters=fpca_model['training_parameters']
    )
    print(f"FPCA model saved to {output_file}")


def load_fpca_model(input_file: str = "outputs/fpca_model.npz") -> Dict[str, Any]:
    """
    Load FPCA model from a compressed numpy file.
    
    Parameters:
    -----------
    input_file : str
        Input file path
    
    Returns:
    --------
    Dict[str, Any]
        Loaded FPCA model
    """
    data = np.load(input_file)
    
    fpca_model = {
        'mean_curve': data['mean_curve'],
        'eigenfunctions': data['eigenfunctions'],
        'eigenvalues': data['eigenvalues'],
        'explained_variance': data['explained_variance'],
        'cumulative_variance': data['cumulative_variance'],
        'n_components': int(data['n_components']),
        'training_scores': data['training_scores'],
        'parameter_names': data['parameter_names'],
        'training_parameters': data['training_parameters']
    }
    
    print(f"FPCA model loaded from {input_file}")
    return fpca_model


def recast_training_data_to_fpca(input_file: str = "outputs/uq_batch_results.npz",
                                fpca_model: Optional[Dict[str, Any]] = None,
                                fpca_model_file: Optional[str] = None,
                                output_file: str = "outputs/training_data_fpca.npz") -> Dict[str, Any]:
    """
    Recast all training data in terms of FPCA coefficients.
    
    Parameters:
    -----------
    input_file : str
        Path to the .npz file containing batch results
    fpca_model : Dict[str, Any], optional
        FPCA model (if None, will load from fpca_model_file or build new one)
    fpca_model_file : str, optional
        Path to saved FPCA model file
    output_file : str
        Output file path for recast data
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - parameters: original parameter values
        - fpca_scores: FPCA coefficients for each simulation
        - parameter_names: names of parameters
        - fpca_model: the FPCA model used
        - valid_mask: mask indicating which simulations were successful
    """
    print(f"Recasting training data to FPCA space...")
    
    # Load batch results
    data = load_batch_results(input_file)
    
    # Load or build FPCA model
    if fpca_model is None:
        if fpca_model_file is not None:
            print(f"Loading FPCA model from {fpca_model_file}")
            fpca_model = load_fpca_model(fpca_model_file)
        else:
            print("Building new FPCA model...")
            fpca_model = build_fpca_model(input_file)
    
    # Filter out failed simulations
    valid_mask = ~np.isnan(data['oside_curves']).any(axis=1)
    valid_curves = data['oside_curves'][valid_mask]
    valid_params = data['parameters'][valid_mask]
    
    print(f"Processing {len(valid_curves)} valid curves out of {len(data['oside_curves'])}")
    
    # Project all curves to FPCA space
    fpca_scores = []
    for i, curve in enumerate(valid_curves):
        scores = project_curve_to_fpca(curve, fpca_model)
        fpca_scores.append(scores)
    
    fpca_scores = np.array(fpca_scores)
    
    print(f"FPCA scores shape: {fpca_scores.shape}")
    print(f"Parameter shape: {valid_params.shape}")
    
    # Save recast data
    np.savez_compressed(
        output_file,
        parameters=valid_params,
        fpca_scores=fpca_scores,
        parameter_names=data['parameter_names'],
        valid_mask=valid_mask,
        n_components=fpca_model['n_components']
    )
    
    print(f"Recast training data saved to {output_file}")
    
    return {
        'parameters': valid_params,
        'fpca_scores': fpca_scores,
        'parameter_names': data['parameter_names'],
        'fpca_model': fpca_model,
        'valid_mask': valid_mask,
        'n_components': fpca_model['n_components']
    }


def load_recast_training_data(input_file: str = "outputs/training_data_fpca.npz") -> Dict[str, Any]:
    """
    Load recast training data from FPCA space.
    
    Parameters:
    -----------
    input_file : str
        Input file path
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing recast training data
    """
    data = np.load(input_file)
    
    recast_data = {
        'parameters': data['parameters'],
        'fpca_scores': data['fpca_scores'],
        'parameter_names': data['parameter_names'],
        'valid_mask': data['valid_mask'],
        'n_components': int(data['n_components'])
    }
    
    print(f"Recast training data loaded from {input_file}")
    print(f"Number of samples: {len(recast_data['parameters'])}")
    print(f"Number of FPCA components: {recast_data['n_components']}")
    
    return recast_data 