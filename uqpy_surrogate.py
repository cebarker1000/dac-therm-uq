from train_surrogate_models import FullSurrogateModel
surrogate_model = FullSurrogateModel.load_model("outputs/full_surrogate_model.pkl")
from UQpy.inference.inference_models.LogLikelihoodModel import LogLikelihoodModel


import numpy as np

def to_2d(arr, *, n_features):
    """
    Ensure `arr` is a 2-D NumPy array with shape (n, n_features).

    Parameters
    ----------
    arr : array-like
        Either 1-D (n_features,) or 2-D (n, n_features).
    n_features : int
        Expected feature count.

    Returns
    -------
    ndarray
        Shape-checked view or copy.
    """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != n_features:
        raise ValueError(f"Expected shape (*, {n_features}), got {arr.shape}")
    return arr


# --- load once at import time ----------------------------------------------
from train_surrogate_models import FullSurrogateModel
from analysis.config_utils import get_fixed_params_from_config
surrogate_model = FullSurrogateModel.load_model("outputs/full_surrogate_model.pkl")

# Load fixed parameters from config file (centers of distributions, excluding k values)
PARAMS_FIXED = get_fixed_params_from_config()

N_TOTAL_PARAMS = PARAMS_FIXED.size + 3   # 11
N_FPCS         = 5

def fpca_model(params):
    """
    Vectorised: params shape (n, 11) ➜ FPCA coeffs shape (n, 5)
    """
    params_2d = to_2d(params, n_features=N_TOTAL_PARAMS)
    coeffs, _ = surrogate_model.predict_fpca_coefficients(params_2d)
    return coeffs                # already (n, 5)

def uncertainty_model(params):
    """
    Vectorised: params shape (n, 11) ➜ FPCA σ shape (n, 5)
    """
    params_2d = to_2d(params, n_features=N_TOTAL_PARAMS)
    _, sigmas = surrogate_model.predict_fpca_coefficients(params_2d)
    return sigmas                # (n, 5)

def timeseries_model(params):
    """
    Vectorised: params shape (n, 11) ➜ time series shape (n, T)
    Reconstructs the full normalized temperature curve from FPCA coefficients.
    """
    from analysis.uq_wrapper import reconstruct_curve_from_fpca
    
    params_2d = to_2d(params, n_features=N_TOTAL_PARAMS)
    coeffs, _ = surrogate_model.predict_fpca_coefficients(params_2d)  # (n, 5)
    
    # Reconstruct curves from FPCA coefficients
    curves = []
    for coeff in coeffs:
        curve = reconstruct_curve_from_fpca(coeff, surrogate_model.fpca_model)
        curves.append(curve)
    
    return np.array(curves)  # (n, T) where T is the number of time points

# --- constants --------------------------------------------------------------
FREE_COUNT     = 3                       # k_sample, k_ins, k_coupler

# ---------------------------------------------------------------------------
def log_likelihood(params_free,
                   data,
                   data_uncertainty=None,
                   use_sigma_gp=True):
    """
    Vectorised Gaussian log-likelihood.
    `params_free` holds ONLY the 3 conductivities in order:
        [k_sample, k_ins, k_coupler]
    If use_sigma_gp=False, surrogate σ_gp is discarded 
    (unit variance is used, so you get pure least-squares).
    """
    import numpy as np

    # -------- reshape & splice ---------------------------------------------
    params_free = np.atleast_2d(params_free)                # (n, 3)
    n = params_free.shape[0]

    # build full (n,11) array: [fixed nuisance | free κ]
    params_full = np.hstack([
        np.tile(PARAMS_FIXED, (n, 1)),                      # repeat fixed
        params_free                                         # append free
    ])  # shape (n,11)

    # ---------- surrogate prediction ---------------------------------------
    mu, sigma_gp = surrogate_model.predict_fpca_coefficients(params_full)
    # both (n,5)

    # ---------- combine uncertainties --------------------------------------
    if use_sigma_gp:
        # use the surrogate's predicted variance
        sigma_tot2 = sigma_gp**2
    else:
        # discard σ_gp → unit variance (pure LS)
        sigma_tot2 = np.ones_like(sigma_gp)

    # (Optionally you could still add data_uncertainty here, but as requested we ignore it.)

    # guard against zeros
    sigma_tot2 = np.maximum(sigma_tot2, 1e-12)

    # ---------- compute log-likelihood -------------------------------------
    resid = mu - data  # (n,5) – (5,) → (n,5)
    logl = -0.5 * np.sum(
        np.log(2*np.pi * sigma_tot2) +
        resid**2 / sigma_tot2,
        axis=1
    )
    return logl

def least_squares(params_free, data_full, fpca_model, surrogate_model, PARAMS_FIXED):
    """
    Compute the residual vector between the experimental time series and the
    curve reconstructed from FPCA components predicted by the surrogate.

    Parameters
    ----------
    params_free : array-like, shape (3,) or (n,3)
        The 3 free conductivity parameters [k_sample, k_ins, k_coupler].
    data_full : array-like, shape (T,)
        The full experimental time series (normalized, length T).
    fpca_model : object
        The trained FPCA model (with .mean_ and .components_ attributes).
    surrogate_model : object
        The trained surrogate model with .predict_fpca_coefficients().
    PARAMS_FIXED : array-like, shape (8,)
        The fixed nuisance parameters.

    Returns
    -------
    residual : np.ndarray, shape (T,) or (n, T)
        The residual(s) between reconstructed curve(s) and experimental data.
    """
    import numpy as np

    params_free = np.atleast_2d(params_free)  # (n, 3)
    n = params_free.shape[0]

    # Build full parameter array: [fixed nuisance | free κ]
    params_full = np.hstack([
        np.tile(PARAMS_FIXED, (n, 1)),  # repeat fixed
        params_free
    ])  # shape (n,11)

    # Predict FPCA coefficients (mean only)
    fpca_coeffs, _ = surrogate_model.predict_fpca_coefficients(params_full)  # (n, n_fpca)

    # Reconstruct curves from FPCA coefficients
    # X_recon = mean_ + coeffs @ components_
    mean_curve = fpca_model.mean_  # (T,)
    components = fpca_model.components_  # (n_fpca, T)
    recon_curves = mean_curve + fpca_coeffs @ components  # (n, T)

    # Compute residual(s)
    data_full = np.asarray(data_full)
    if data_full.ndim == 1:
        data_full = data_full[None, :]  # (1, T)
    residual = recon_curves - data_full  # (n, T)

    if residual.shape[0] == 1:
        return residual[0]
    return residual
