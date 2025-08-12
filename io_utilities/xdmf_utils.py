"""
This module provides a set of utility functions for handling XDMF files, which
are used for storing the simulation mesh and time-dependent results.

It includes functions for:

- Initializing an XDMF file.
- Saving a dictionary of simulation parameters to a text file for
  reproducibility.
"""
import os
from dolfinx import io


def init_xdmf(domain, sim_folder, output_name):
    """Initialize an XDMF file for time-dependent output.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        The computational mesh on which the simulation runs.
    sim_folder : str
        Path to the directory where output files will be written.
    output_name : str
        Base name for the output XDMF file (without extension).

    Returns
    -------
    dolfinx.io.XDMFFile
        Open XDMF file ready to accept mesh and solution writes.
    """

    xdmf_path = os.path.join(sim_folder, f"{output_name}.xdmf")
    xdmf = io.XDMFFile(domain.comm, xdmf_path, "w")
    xdmf.write_mesh(domain)
    return xdmf


def save_params(sim_folder, params_dict):
    """Save simulation parameters to a text file for reproducibility.

    Parameters
----------
    sim_folder : str
        Directory where the ``params.txt`` file will be saved.
    params_dict : dict
        Dictionary of parameter names and their values.
    """

    params_path = os.path.join(sim_folder, "params.txt")

    with open(params_path, "w") as f:
        for key, val in params_dict.items():
            f.write(f"{key} = {val}\n")
