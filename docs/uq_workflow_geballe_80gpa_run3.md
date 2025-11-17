### Full UQ workflow: Geballe 80 GPa – Run 3

This walkthrough runs the complete pipeline using the prepared inputs under `outputs/geballe/80gpa/run3/`.

- **Simulation config**: `outputs/geballe/80gpa/run3/sim_cfg.yaml`
- **Distributions (priors + mappings)**: `outputs/geballe/80gpa/run3/distributions.yaml`
- **Output directory**: `outputs/geballe/80gpa/run3/`

#### 0) Prerequisites
- Python 3.8+
- Packages per README (`dolfinx`, `petsc4py`, `gmsh`, `meshio`, `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `UQpy`, `scikit-learn`, `corner`, `arviz`).
- PETSc/MPI configured for `dolfinx`.

Optional environment paths for convenience:
```bash
export V2HEATFLOW_MESH_DIR="meshes"
export V2HEATFLOW_OUTPUT_DIR="outputs"
```

#### 1) Quick simulation sanity check (optional)
Run a single FEM simulation to verify the config and mesh:
```bash
python run_simulation.py \
  --config outputs/geballe/80gpa/run3/sim_cfg.yaml \
  --mesh-folder meshes/no_diamond \
  --output-folder outputs/geballe/80gpa/run3/sim_test \
```
Outputs:
- `outputs/geballe/80gpa/run3/sim_test/output.xdmf` (if enabled)
- `outputs/geballe/80gpa/run3/sim_test/watcher_points.csv`
- Plots and residuals if not suppressed

Notes:
- Baseline normalization is controlled by `baseline` in the sim config (used consistently throughout).

#### 2) Generate training data (Latin Hypercube + minimal sims)
This creates curves for many parameter draws, builds FPCA, and recasts curves.
```bash
python generate_training_data.py \
  --distributions outputs/geballe/80gpa/run3/distributions.yaml \
  --config outputs/geballe/80gpa/run3/sim_cfg.yaml \
  --output-dir outputs/geballe/80gpa/run3
```
Key outputs (under `outputs/geballe/80gpa/run3/`):
- `initial_train_set.csv`: parameter samples
- `uq_batch_results.npz`: raw curves/parameters
- `fpca_model.npz`: FPCA model
- `training_data_fpca.npz`: curves recast to FPCA coefficients
- Parameter distribution and correlation plots

Tips:
- `sampling.n_samples` is read from the distributions file.
- The script verifies the experimental heating file referenced by `sim_cfg.yaml`.

#### 3) Train the surrogate (GP-on-FPCA)
Fit a GP to each FPCA component and save the surrogate model.
```bash
python train_surrogate_models.py \
  --input_path outputs/geballe/80gpa/run3/training_data_fpca.npz \
  --fpca_model_path outputs/geballe/80gpa/run3/fpca_model.npz \
  --output_path outputs/geballe/80gpa/run3/full_surrogate_model.pkl \
  --training_config outputs/geballe/80gpa/run3/sim_cfg.yaml
```
Outputs:
- `full_surrogate_model.pkl`: packed model (FPCA + scaler + GP ensemble)
- Console summary with R² on test curves and residuals

#### 4) (Optional) Validate surrogate vs FEM
There are convenience scripts, but some assume default paths (`outputs/full_surrogate_model.pkl`). For this run you can:
- Create a symlink `outputs/full_surrogate_model.pkl` → `outputs/geballe/80gpa/run3/full_surrogate_model.pkl`, then use `validate_surrogates.py`.
- Or do spot-checks by comparing `run_single_simulation` vs surrogate predictions with small helper snippets.

#### 5) MCMC with UQpy (posterior over parameters)
Run MCMC using the trained surrogate and experimental data.
```bash
python uqpy_MCMC.py \
  --config_path outputs/geballe/80gpa/run3/distributions.yaml \
  --surrogate_path outputs/geballe/80gpa/run3/full_surrogate_model.pkl \
  --exp_data_path data/experimental/geballe_heat_data.csv \
  --sim_cfg outputs/geballe/80gpa/run3/sim_cfg.yaml \
  --output_path outputs/geballe/80gpa/run3/mcmc_results.npz \
  --plot_path_prefix outputs/geballe/80gpa/run3/mcmc \
  --n_walkers 60 \
  --n_samples 100000 \
  --burn_length 20000
```
Notes:
- The likelihood is heteroskedastic Gaussian with per-timepoint variance combining sensor noise and GP curve uncertainty (with optional error inflation if present as a parameter).
- Experimental series are interpolated to the surrogate time grid prior to evaluation.

#### 6) Post-process MCMC results
Generate corner/trace plots, parameter stats, and diagnostics.
```bash
python plot_mcmc_results.py \
  --results outputs/geballe/80gpa/run3/mcmc_results.npz \
  --config outputs/geballe/80gpa/run3/distributions.yaml \
  --output outputs/geballe/80gpa/run3
```
Outputs (in `outputs/geballe/80gpa/run3/`):
- `kappa_corner_plot.png`, `full_corner_plot.png`
- `trace_plots.png`, `parameter_statistics.png`
- `likelihood_analysis.png` (if likelihood values saved)
- `nuisance_parameter_influence.png`

#### 7) Interpreting results
- Check convergence: min ESS > 200, max R-hat < 1.01.
- Compare posterior of κ parameters and nuisance parameters to priors.
- Optional: rerun a forward sim at posterior means/medians and compare to experiment.

#### Troubleshooting
- Ensure `dolfinx`/`petsc4py` versions are compatible; run a small sim first (step 1).
- If surrogate diagnostics are poor, increase `sampling.n_samples` and retrain.
- If MCMC mixes slowly, increase walkers or tune `--n_samples`/`--burn_length`.
- Normalization: baseline rules come from `sim_cfg.yaml: baseline` and are applied consistently across simulation, analysis, and MCMC.

#### File summary for this run
- Configs: `outputs/geballe/80gpa/run3/sim_cfg.yaml`, `outputs/geballe/80gpa/run3/distributions.yaml`
- Training artifacts: CSV/NPZ/plots under `outputs/geballe/80gpa/run3/`
- Surrogate: `outputs/geballe/80gpa/run3/full_surrogate_model.pkl`
- MCMC: `outputs/geballe/80gpa/run3/mcmc_results.npz` and plots with prefix `mcmc*`
