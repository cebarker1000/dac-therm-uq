"""
This module provides a set of utility functions for loading and parsing the
YAML configuration files that define the parameter distributions and mappings
for the uncertainty quantification (UQ) analysis.

It includes functions for:

- Loading the distributions configuration from a YAML file.
- Extracting parameter definitions and mappings in the format expected by
  the UQ scripts.
- Creating UQpy distribution objects from the parameter definitions.
- Transforming parameters between real-space and log-space.
"""

import yaml
from typing import Dict, List, Any, Tuple
import numpy as np
from UQpy.distributions.collection import Uniform, Normal, Lognormal


def load_distributions_config(config_path: str = "configs/distributions.yaml") -> Dict[str, Any]:
    """
    Load the distributions configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the loaded configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_param_defs_from_config(config_path: str = "configs/distributions.yaml") -> List[Dict[str, Any]]:
    """
    Extract parameter definitions from config file in the format expected by existing code.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        List of parameter definition dictionaries
    """
    config = load_distributions_config(config_path)
    param_defs = []
    
    for param_name, param_config in config['parameters'].items():
        param_def = {
            "name": param_name,
            "type": param_config["type"]
        }
        
        # Add type-specific parameters
        if param_config["type"] == "lognormal":
            param_def["center"] = param_config["center"]
            param_def["sigma_log"] = param_config["sigma_log"]
        elif param_config["type"] == "normal":
            param_def["center"] = param_config["center"]
            param_def["sigma"] = param_config["sigma"]
        elif param_config["type"] == "uniform":
            param_def["low"] = param_config["low"]
            param_def["high"] = param_config["high"]
        
        param_defs.append(param_def)
    
    return param_defs


def get_param_mapping_from_config(config_path: str = "configs/distributions.yaml") -> Dict[str, List[Tuple]]:
    """
    Extract parameter mapping from config file in the format expected by existing code.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary mapping parameter names to lists of config paths
    """
    config = load_distributions_config(config_path)
    param_mapping = {}
    
    for param_name, mappings in config['parameter_mapping'].items():
        # Convert list of lists to list of tuples
        param_mapping[param_name] = [tuple(mapping) for mapping in mappings]
    
    return param_mapping


def get_sampling_config(config_path: str = "configs/distributions.yaml") -> Dict[str, Any]:
    """
    Extract sampling configuration from config file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing sampling configuration
    """
    config = load_distributions_config(config_path)
    return config.get('sampling', {})


def get_output_config(config_path: str = "configs/distributions.yaml") -> Dict[str, Any]:
    """
    Extract output configuration from config file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing output configuration
    """
    config = load_distributions_config(config_path)
    return config.get('output', {})


def create_uqpy_distributions(param_defs: List[Dict[str, Any]]) -> List:
    """
    Create UQpy distribution objects from parameter definitions.
    
    Args:
        param_defs: List of parameter definition dictionaries
        
    Returns:
        List of UQpy distribution objects
    """
    distributions = []
    for p in param_defs:
        if p["type"] == "lognormal":
            sigma = p["sigma_log"]    
            center = p["center"] 
            # For lognormal, s=sigma_log (shape), scale=center (geometric mean)
            distributions.append(Lognormal(s=sigma, scale=center))
        elif p["type"] == "normal":
            distributions.append(Normal(loc=p["center"], scale=p["sigma"]))
        elif p["type"] == "uniform":
            distributions.append(Uniform(loc=p["low"], scale=p["high"] - p["low"]))
        else:
            raise ValueError(f"Unknown type: {p['type']}")
    
    return distributions


def get_fixed_params_from_config(config_path: str = "configs/distributions.yaml") -> np.ndarray:
    """
    Get the fixed parameter values (excluding k values) from config file.
    These are the centers of the distributions for the nuisance parameters.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Array of fixed parameter values in the order: [d_sample, rho_cv_sample, rho_cv_coupler, rho_cv_ins, d_coupler, d_ins_pside, d_ins_oside, fwhm]
    """
    config = load_distributions_config(config_path)
    
    # Define the order of fixed parameters (excluding k values)
    fixed_param_names = ['d_sample', 'rho_cv_sample', 'rho_cv_coupler', 'rho_cv_ins', 'd_coupler', 'd_ins_pside', 'd_ins_oside', 'fwhm']
    
    fixed_params = []
    for param_name in fixed_param_names:
        if param_name in config['parameters']:
            param_config = config['parameters'][param_name]
            if param_config['type'] == 'lognormal':
                fixed_params.append(param_config['center'])
            else:
                # For uniform distributions, use the midpoint
                fixed_params.append((param_config['low'] + param_config['high']) / 2)
        else:
            raise ValueError(f"Parameter {param_name} not found in config file")
    
    return np.array(fixed_params)


def load_all_from_config(config_path: str = "configs/distributions.yaml") -> Tuple[List[Dict[str, Any]], Dict[str, List[Tuple]], Dict[str, Any], Dict[str, Any]]:
    """
    Load all configuration data from the YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Tuple of (param_defs, param_mapping, sampling_config, output_config)
    """
    param_defs = get_param_defs_from_config(config_path)
    param_mapping = get_param_mapping_from_config(config_path)
    sampling_config = get_sampling_config(config_path)
    output_config = get_output_config(config_path)
    
    return param_defs, param_mapping, sampling_config, output_config


# ============================================================================
# Log-space transformation utilities
# ============================================================================

def real_to_log_space(params_real: np.ndarray, param_defs: List[Dict[str, Any]]) -> np.ndarray:
    """
    Transform parameters from real-space to log-space.
    
    Args:
        params_real: Parameters in real-space, shape (n_samples, n_params) or (n_params,)
        param_defs: Parameter definitions from config
        
    Returns:
        Parameters in log-space, same shape as input
    """
    params_real = np.asarray(params_real)
    original_shape = params_real.shape
    
    # Flatten for processing
    if len(original_shape) == 1:
        params_real = params_real.reshape(1, -1)
        was_1d = True
    else:
        was_1d = False
    
    n_samples, n_params = params_real.shape
    
    # Initialize log-space parameters
    params_log = np.zeros_like(params_real)
    
    for i, param_def in enumerate(param_defs):
        param_type = param_def["type"]
        
        if param_type == "lognormal":
            # Already in log-space, just take the log
            params_log[:, i] = np.log(params_real[:, i])
        elif param_type == "normal":
            # For normal distributions, we still take the log if the parameter is positive
            # This is a reasonable choice for physical parameters that are always positive
            params_log[:, i] = np.log(params_real[:, i])
        elif param_type == "uniform":
            # For uniform distributions, take the log
            params_log[:, i] = np.log(params_real[:, i])
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
    
    # Restore original shape
    if was_1d:
        params_log = params_log.flatten()
    
    return params_log


def log_to_real_space(params_log: np.ndarray, param_defs: List[Dict[str, Any]]) -> np.ndarray:
    """
    Transform parameters from log-space back to real-space.
    
    Args:
        params_log: Parameters in log-space, shape (n_samples, n_params) or (n_params,)
        param_defs: Parameter definitions from config
        
    Returns:
        Parameters in real-space, same shape as input
    """
    params_log = np.asarray(params_log)
    original_shape = params_log.shape
    
    # Flatten for processing
    if len(original_shape) == 1:
        params_log = params_log.reshape(1, -1)
        was_1d = True
    else:
        was_1d = False
    
    n_samples, n_params = params_log.shape
    
    # Initialize real-space parameters
    params_real = np.zeros_like(params_log)
    
    for i, param_def in enumerate(param_defs):
        param_type = param_def["type"]
        
        if param_type in ["lognormal", "normal", "uniform"]:
            # Transform back using exponential
            params_real[:, i] = np.exp(params_log[:, i])
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
    
    # Restore original shape
    if was_1d:
        params_real = params_real.flatten()
    
    return params_real


def compute_jacobian_correction(params_real: np.ndarray, param_defs: List[Dict[str, Any]]) -> np.ndarray:
    """
    Compute the Jacobian correction factor for the log-space transformation.
    
    The Jacobian is |∂param_real/∂param_log| = param_real = exp(param_log)
    For multiple parameters, the total Jacobian is the product of individual Jacobians.
    
    Args:
        params_real: Parameters in real-space, shape (n_samples, n_params) or (n_params,)
        param_defs: Parameter definitions from config
        
    Returns:
        Jacobian correction factors, shape (n_samples,) or scalar
    """
    params_real = np.asarray(params_real)
    original_shape = params_real.shape
    
    # Flatten for processing
    if len(original_shape) == 1:
        params_real = params_real.reshape(1, -1)
        was_1d = True
    else:
        was_1d = False
    
    n_samples, n_params = params_real.shape
    
    # Initialize Jacobian factors
    jacobian_factors = np.ones(n_samples)
    
    for i, param_def in enumerate(param_defs):
        param_type = param_def["type"]
        
        if param_type in ["lognormal", "normal", "uniform"]:
            # Jacobian factor is the parameter value itself
            jacobian_factors *= params_real[:, i]
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
    
    # Restore original shape
    if was_1d:
        jacobian_factors = jacobian_factors.item()
    
    return jacobian_factors


def create_logspace_distributions(param_defs: List[Dict[str, Any]]) -> List:
    """
    Create UQpy distribution objects for log-space sampling.
    
    This transforms the original parameter distributions to work in log-space.
    
    Args:
        param_defs: List of parameter definition dictionaries
        
    Returns:
        List of UQpy distribution objects for log-space sampling
    """
    distributions = []
    
    for p in param_defs:
        param_type = p["type"]
        
        if param_type == "lognormal":
            # For lognormal in real-space, the log-space distribution is normal
            # If X ~ LogNormal(μ, σ), then log(X) ~ Normal(μ, σ)
            mu_log = np.log(p["center"])  # log of geometric mean
            sigma_log = p["sigma_log"]    # log standard deviation
            distributions.append(Normal(loc=mu_log, scale=sigma_log))
            
        elif param_type == "normal":
            # For normal distributions, we need to be careful
            # If the original parameter is always positive, we can use log transformation
            # The log-space distribution will be approximately normal if the original
            # parameter has small coefficient of variation
            center = p["center"]
            sigma = p["sigma"]
            
            # Use delta method approximation for log transformation
            mu_log = np.log(center)
            sigma_log = sigma / center  # approximate log-space standard deviation
            
            distributions.append(Normal(loc=mu_log, scale=sigma_log))
            
        elif param_type == "uniform":
            # For uniform distributions, the log-space distribution is not uniform
            # If X ~ Uniform(a, b), then log(X) has a transformed distribution
            low = p["low"]
            high = p["high"]
            
            # Log-space bounds
            log_low = np.log(low)
            log_high = np.log(high)
            
            # Use uniform in log-space (this is a reasonable approximation)
            distributions.append(Uniform(loc=log_low, scale=log_high - log_low))
            
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
    
    return distributions


def get_logspace_bounds(param_defs: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
    """
    Get bounds for log-space sampling based on original parameter definitions.
    
    Args:
        param_defs: List of parameter definition dictionaries
        
    Returns:
        List of (low, high) bounds for log-space sampling
    """
    bounds = []
    
    for p in param_defs:
        param_type = p["type"]
        
        if param_type == "lognormal":
            # For lognormal, use reasonable bounds based on the distribution
            center = p["center"]
            sigma_log = p["sigma_log"]
            
            # Use ±4 standard deviations in log-space
            mu_log = np.log(center)
            log_low = mu_log - 4 * sigma_log
            log_high = mu_log + 4 * sigma_log
            
            bounds.append((log_low, log_high))
            
        elif param_type == "normal":
            # For normal distributions, use reasonable bounds
            center = p["center"]
            sigma = p["sigma"]
            
            # Use ±4 standard deviations, but ensure positive
            low = max(center - 4 * sigma, 1e-10)  # Ensure positive
            high = center + 4 * sigma
            
            log_low = np.log(low)
            log_high = np.log(high)
            
            bounds.append((log_low, log_high))
            
        elif param_type == "uniform":
            # For uniform, use the log of the original bounds
            low = p["low"]
            high = p["high"]
            
            log_low = np.log(low)
            log_high = np.log(high)
            
            bounds.append((log_low, log_high))
            
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
    
    return bounds 