# Heat Flow Simulation and Uncertainty Quantification

This repository contains a suite of scripts for running heat flow simulations
and performing uncertainty quantification (UQ) analysis. The codebase is
designed to be modular and configurable, allowing for easy extension to new
datasets and analysis methods.

## Getting Started

To get started with this project, you will need to have Python 3.8+ and the
required dependencies installed.

### Dependencies

The following Python packages are required to run the simulations and
analyses:

- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `seaborn`
- `pyyaml`
- `dolfinx`
- `ufl`
- `petsc4py`
- `gmsh`
- `meshio`
- `arviz`
- `corner`
- `UQpy`
- `scikit-learn`

You can install these packages using pip:

```bash
pip install numpy pandas scipy matplotlib seaborn pyyaml dolfinx ufl petsc4py gmsh meshio arviz corner UQpy scikit-learn
```

### Running the Main Workflows

The two main workflows in this project are running a single simulation and
performing a full UQ analysis.

**1. Running a Single Simulation**

To run a single simulation, use the `run_simulation.py` script with a YAML
configuration file:

```bash
python run_simulation.py --config configs/config_5_materials.yaml
```

This will run the simulation defined in `config_5_materials.yaml` and save
the results to the `outputs/` directory.

**2. Performing a Full UQ Analysis**

The UQ analysis workflow involves three main steps:

1.  **Generate Training Data:** Use the `generate_training_data.py` script
    to create a set of training data for the surrogate model.

    ```bash
    python generate_training_data.py
    ```

2.  **Train Surrogate Model:** Use the `train_surrogate_models.py` script
    to train a surrogate model from the generated training data.

    ```bash
    python train_surrogate_models.py
    ```

3.  **Run UQ Analysis:** Use the `uqpy_MCMC.py` or `uqpy_ls.py` script to
    perform the UQ analysis.

    ```bash
    python uqpy_MCMC.py --config_path configs/distributions.yaml --surrogate_path outputs/full_surrogate_model.pkl --exp_data_path data/experimental/geballe_heat_data.csv --sim_cfg configs/config_5_materials.yaml --output_path outputs/mcmc_results.npz --plot_path_prefix outputs/mcmc
    ```

## Codebase Structure

The codebase is organized into the following directories:

- `core/`: Contains the core simulation engine.
- `analysis/`: Contains utility functions and wrapper scripts for UQ
  analysis.
- `io_utilities/`: Contains utility functions for handling simulation I/O.
- `configs/`: Contains the YAML configuration files.
- `data/`: Contains experimental and simulation data.
- `outputs/`: The default directory for saving results.

## Available Scripts

The following is a list of the main scripts in the repository and their
purpose:

### Core Workflow Scripts

- `run_simulation.py`: The main entry point for running individual
  simulations.
- `generate_training_data.py`: Generates the training data for the
  surrogate model.
- `train_surrogate_models.py`: Trains a surrogate model from the generated
  training data.
- `uqpy_MCMC.py`: Performs MCMC-based parameter estimation and UQ.
- `uqpy_ls.py`: Performs least-squares-based parameter estimation and UQ.

### Analysis and Plotting

- `plot_mcmc_results.py`: A comprehensive tool for plotting and analyzing
  the results of an MCMC simulation.
- `plot_fpca_decomposition.py`: A tool for visualizing and analyzing the
  results of a Functional Principal Component Analysis (FPCA).
- `analyze_fpca.py`: The core tool for performing FPCA on the simulation
  results.
- `analyze_monte_carlo.py`: A tool for analyzing the results of a Monte
  Carlo simulation.

### Validation and Debugging

- `compare_single_parameter_set.py`: Compares the surrogate model to a full
  FEM simulation for a single, user-defined set of parameters.
- `compare_fem_surrogate_multiple.py`: Compares the surrogate model to the
  full FEM simulation at multiple points in the parameter space.
- `validate_surrogate_accuracy.py`: Validates the accuracy of the surrogate
  model against the training data.
- `validate_surrogates.py`: Validates the surrogate model against a new set
  of randomly drawn parameter samples.
- `examine_surrogate_training_data.py`: A tool for analyzing and
  validating the training data used to build the surrogate model.
- `examine_training_curves.py`: A diagnostic script for visualizing the
  raw outputs of the batch simulations.

### Automation Scripts

- `generate_all_training_data.sh`: A script for running the
  `generate_training_data.py` script multiple times with different
  configuration files.
- `process_data.sh`: A pipeline script that automates the process of
  training a surrogate model and running an MCMC analysis for a specific
  set of experimental data.
