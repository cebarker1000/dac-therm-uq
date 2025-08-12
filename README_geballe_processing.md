# Geballe 41 GPa FWHM Processing Script

This script automates the process of training surrogate models and running MCMC analysis for all fwhm folders in the geballe 41 gpa directory.

## Overview

The script `process_geballe_41gpa_fwhm.sh` will:

1. **Find all fwhm folders** in `outputs/geballe/41Gpa/`
2. **Train surrogate models** for each folder using the existing training data
3. **Run MCMC analysis** for each folder using the corresponding experimental data
4. **Generate plots** including corner plots and trace plots
5. **Provide detailed logging** with colored output for easy tracking

## Prerequisites

Before running the script, ensure you have:

1. **Training data generated** for each fwhm folder:
   - `training_data_fpca.npz`
   - `fpca_model.npz`
   - `distributions.yaml`
   - `config_5_materials.yaml`

2. **Experimental data files** available:
   - `data/experimental/geballe/geballe_41Gpa_1.csv`
   - `data/experimental/geballe/geballe_41Gpa_2.csv`
   - `data/experimental/geballe/geballe_41Gpa_3.csv`

3. **Python dependencies** installed:
   - All required packages for `train_surrogate_models.py`
   - All required packages for `uqpy_MCMC.py`

## Usage

### Basic Usage

```bash
./process_geballe_41gpa_fwhm.sh
```

### Make Script Executable (if needed)

```bash
chmod +x process_geballe_41gpa_fwhm.sh
```

## What the Script Does

### For Each FWHM Folder:

1. **Validation**: Checks that all required files exist
2. **Surrogate Training**: 
   - Uses existing FPCA model and training data
   - Trains a full surrogate model with GP regression
   - Saves model as `full_surrogate_model.pkl`
3. **MCMC Analysis**:
   - Loads the trained surrogate model
   - Runs MCMC sampling with 60 walkers, 100,000 samples
   - Uses corresponding experimental data file
   - Saves results as `mcmc_results.npz`
4. **Plot Generation**:
   - Creates corner plots (`mcmc_corner.png`)
   - Creates trace plots (`mcmc_trace.png`)

### Experimental Data Mapping

The script automatically maps each fwhm folder to its corresponding experimental data:

- `run1_fwhm` → `geballe_41Gpa_1.csv`
- `run2_fwhm` → `geballe_41Gpa_2.csv`
- `run3_fwhm` → `geballe_41Gpa_3.csv`

## Output Files

For each processed folder, you'll get:

- `full_surrogate_model.pkl` - Trained surrogate model
- `mcmc_results.npz` - MCMC sampling results
- `mcmc_corner.png` - Corner plot of posterior distributions
- `mcmc_trace.png` - Trace plots showing convergence

## MCMC Parameters

The script uses these default MCMC parameters:

- **Walkers**: 60
- **Samples**: 100,000
- **Burn-in**: 20,000
- **Scale**: 2.4 (Stretch sampler)

## Error Handling

The script includes comprehensive error handling:

- **File validation**: Checks all required files exist
- **Process tracking**: Shows progress for each step
- **Error reporting**: Detailed error messages with colors
- **Resume capability**: Skips already completed steps
- **Summary report**: Shows which folders succeeded/failed

## Logging

The script provides colored output:

- 🔵 **Blue**: Information messages
- 🟢 **Green**: Success messages
- 🟡 **Yellow**: Warnings (e.g., files already exist)
- 🔴 **Red**: Error messages

## Example Output

```
[INFO] Starting processing of geballe 41 gpa fwhm folders...
[INFO] Found 3 fwhm folders:
  - run1_fwhm
  - run2_fwhm
  - run3_fwhm

==========================================
[INFO] Processing: run1_fwhm
==========================================
[INFO] Training surrogate model for run1_fwhm...
[SUCCESS] Surrogate model trained successfully for run1_fwhm
[INFO] Running MCMC analysis for run1_fwhm...
[SUCCESS] MCMC analysis completed successfully for run1_fwhm
[SUCCESS] Completed processing for run1_fwhm

==========================================
[INFO] PROCESSING SUMMARY
==========================================
[SUCCESS] Successfully processed 3 folders:
  ✓ run1_fwhm
  ✓ run2_fwhm
  ✓ run3_fwhm
==========================================
[SUCCESS] All folders processed successfully!
```

## Troubleshooting

### Common Issues

1. **Missing training data**: Ensure you've run `generate_training_data.py` for each folder
2. **Missing experimental data**: Check that CSV files exist in `data/experimental/geballe/`
3. **Python errors**: Verify all dependencies are installed
4. **Permission errors**: Make sure the script is executable (`chmod +x`)

### Manual Steps

If the script fails, you can run steps manually:

```bash
# Train surrogate model for a specific folder
python train_surrogate_models.py \
    --input_path outputs/geballe/41Gpa/run1_fwhm/training_data_fpca.npz \
    --fpca_model_path outputs/geballe/41Gpa/run1_fwhm/fpca_model.npz \
    --output_path outputs/geballe/41Gpa/run1_fwhm/full_surrogate_model.pkl \
    --training_config outputs/geballe/41Gpa/run1_fwhm/config_5_materials.yaml

# Run MCMC for a specific folder
python uqpy_MCMC.py \
    --config_path outputs/geballe/41Gpa/run1_fwhm/distributions.yaml \
    --surrogate_path outputs/geballe/41Gpa/run1_fwhm/full_surrogate_model.pkl \
    --exp_data_path data/experimental/geballe/geballe_41Gpa_1.csv \
    --sim_cfg outputs/geballe/41Gpa/run1_fwhm/config_5_materials.yaml \
    --output_path outputs/geballe/41Gpa/run1_fwhm/mcmc_results.npz \
    --plot_path_prefix outputs/geballe/41Gpa/run1_fwhm/mcmc
```

## Customization

To modify the script behavior, edit these variables in the script:

- `BASE_DIR`: Change the base directory path
- `EXPERIMENTAL_DATA_DIR`: Change experimental data directory
- MCMC parameters: Modify `--n_walkers`, `--n_samples`, `--burn_length`
- Training parameters: Modify `--test_fraction`, `--random_state`