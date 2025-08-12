# Generic FWHM Processing Script

This is a flexible version of the processing script that can be adapted for different data directories and experimental data sets.

## Overview

The generic script `process_geballe_fwhm_generic.sh` provides a configurable framework for:

1. **Processing any fwhm folders** in any data directory
2. **Mapping folders to experimental data** using a configurable dictionary
3. **Customizable MCMC and training parameters**
4. **Reusable across different experiments** (Geballe, Edmund, etc.)

## Key Features

### 🔧 **Configurable Variables**
- `BASE_DIR`: Target data directory
- `EXPERIMENTAL_DATA_DIR`: Experimental data location
- `EXPERIMENTAL_DATA_MAPPING`: Folder-to-data mapping
- `N_WALKERS`, `N_SAMPLES`, `BURN_LENGTH`: MCMC parameters
- `TEST_FRACTION`, `RANDOM_STATE`: Training parameters

### 📊 **Flexible Data Mapping**
Use associative arrays to map folder names to experimental data files:

```bash
declare -A EXPERIMENTAL_DATA_MAPPING=(
    ["run1_fwhm"]="geballe_41Gpa_1.csv"
    ["run2_fwhm"]="geballe_41Gpa_2.csv"
    ["run3_fwhm"]="geballe_41Gpa_3.csv"
)
```

### 🎯 **Smart Processing**
- Validates all required files exist
- Skips already completed steps
- Provides detailed progress logging
- Handles errors gracefully

## Usage Methods

### Method 1: Direct Configuration (Recommended)

1. **Copy the generic script**:
   ```bash
   cp process_geballe_fwhm_generic.sh my_processing_script.sh
   ```

2. **Edit the configuration variables** at the top of the script:
   ```bash
   # Configuration - MODIFY THESE VARIABLES FOR DIFFERENT DATA SETS
   BASE_DIR="outputs/geballe/40Gpa"  # Change this
   EXPERIMENTAL_DATA_DIR="data/experimental/geballe"
   
   # Experimental data mapping
   declare -A EXPERIMENTAL_DATA_MAPPING=(
       ["run1_fwhm"]="geballe_40Gpa_1.csv"
       ["run2_fwhm"]="geballe_40Gpa_2.csv"
       ["run3_fwhm"]="geballe_40Gpa_3.csv"
       ["run4_fwhm"]="geballe_40Gpa_4.csv"
   )
   ```

3. **Run the script**:
   ```bash
   chmod +x my_processing_script.sh
   ./my_processing_script.sh
   ```

### Method 2: Configuration Files

1. **Use provided example configurations**:
   ```bash
   # For 40 GPa data
   ./config_examples/geballe_40gpa_config.sh
   
   # For 62 GPa data
   ./config_examples/geballe_62gpa_config.sh
   
   # For Edmund data
   ./config_examples/edmund_config.sh
   ```

2. **Create your own configuration**:
   ```bash
   cp config_examples/geballe_40gpa_config.sh my_config.sh
   # Edit my_config.sh with your settings
   ./my_config.sh
   ```

## Example Configurations

### Geballe 40 GPa
```bash
BASE_DIR="outputs/geballe/40Gpa"
EXPERIMENTAL_DATA_MAPPING=(
    ["run1_fwhm"]="geballe_40Gpa_1.csv"
    ["run2_fwhm"]="geballe_40Gpa_2.csv"
    ["run3_fwhm"]="geballe_40Gpa_3.csv"
    ["run4_fwhm"]="geballe_40Gpa_4.csv"
)
```

### Geballe 62 GPa
```bash
BASE_DIR="outputs/geballe/62Gpa"
EXPERIMENTAL_DATA_MAPPING=(
    ["run1_fwhm"]="geballe_62GPa_1.csv"
    ["run2_fwhm"]="geballe_62GPa_2.csv"
)
```

### Edmund Data
```bash
BASE_DIR="outputs/edmund"
EXPERIMENTAL_DATA_DIR="data/experimental"
EXPERIMENTAL_DATA_MAPPING=(
    ["run1_fwhm"]="edmund_71Gpa_run1.csv"
    ["run2_fwhm"]="edmund_71Gpa_run2.csv"
    ["run3_fwhm"]="edmund_71Gpa_run3.csv"
    ["run4_fwhm"]="edmund_71Gpa_run4.csv"
)
```

## Customization Options

### MCMC Parameters
```bash
N_WALKERS=60          # Number of MCMC walkers
N_SAMPLES=100000      # Number of samples to generate
BURN_LENGTH=20000     # Burn-in period
```

### Training Parameters
```bash
TEST_FRACTION=0.2     # Fraction of data for testing
RANDOM_STATE=42       # Random seed for reproducibility
```

### Directory Structure
```bash
BASE_DIR="outputs/geballe/40Gpa"           # Your data directory
EXPERIMENTAL_DATA_DIR="data/experimental/geballe"  # Experimental data location
```

## Advanced Usage

### Custom Folder Patterns
If your folders don't follow the `*fwhm` pattern, modify the find command in the script:

```bash
# For folders ending with _analysis
fwhm_folders=($(find "$BASE_DIR" -maxdepth 1 -type d -name "*_analysis" | sort))

# For folders starting with run_
fwhm_folders=($(find "$BASE_DIR" -maxdepth 1 -type d -name "run_*" | sort))
```

### Different File Names
If your required files have different names, modify the `check_required_files` function:

```bash
local required_files=("my_training_data.npz" "my_fpca_model.npz" "my_distributions.yaml" "my_config.yaml")
```

### Custom MCMC Parameters per Folder
For different MCMC settings per folder, you can create folder-specific configurations:

```bash
# In the script, add folder-specific logic
case $run_name in
    "run1_fwhm")
        N_WALKERS=80
        N_SAMPLES=200000
        ;;
    "run2_fwhm")
        N_WALKERS=40
        N_SAMPLES=50000
        ;;
    *)
        N_WALKERS=60
        N_SAMPLES=100000
        ;;
esac
```

## Troubleshooting

### Common Issues

1. **"No experimental data mapping found"**
   - Check that your folder name is in the `EXPERIMENTAL_DATA_MAPPING` array
   - Ensure the mapping uses the exact folder name

2. **"Experimental data file not found"**
   - Verify the CSV file exists in `EXPERIMENTAL_DATA_DIR`
   - Check the file path and permissions

3. **"No fwhm folders found"**
   - Confirm the `BASE_DIR` path is correct
   - Check that folders end with `fwhm` (or modify the pattern)

4. **"Required file not found"**
   - Ensure you've run `generate_training_data.py` for each folder
   - Check that all required files exist in each folder

### Debug Mode
Add debugging to see what the script is doing:

```bash
# Add this at the top of the script
set -x  # Print each command before executing
```

### Manual Testing
Test individual components:

```bash
# Test folder detection
find "outputs/geballe/40Gpa" -maxdepth 1 -type d -name "*fwhm"

# Test experimental data mapping
echo "${EXPERIMENTAL_DATA_MAPPING[run1_fwhm]}"

# Test file existence
ls -la "outputs/geballe/40Gpa/run1_fwhm/"
```

## Best Practices

### 1. **Use Descriptive Names**
```bash
# Good
BASE_DIR="outputs/geballe/40Gpa"
EXPERIMENTAL_DATA_MAPPING=(
    ["run1_fwhm"]="geballe_40Gpa_1.csv"
)

# Avoid
BASE_DIR="outputs/data"
EXPERIMENTAL_DATA_MAPPING=(
    ["folder1"]="data1.csv"
)
```

### 2. **Validate Your Configuration**
Before running, check:
- [ ] Base directory exists
- [ ] Experimental data files exist
- [ ] Folder names match mapping keys
- [ ] Required files exist in each folder

### 3. **Use Version Control**
```bash
# Save your configuration
cp process_geballe_fwhm_generic.sh my_experiment_processing.sh
git add my_experiment_processing.sh
git commit -m "Add processing script for my experiment"
```

### 4. **Document Your Changes**
Add comments to explain your configuration:

```bash
# Configuration for My Experiment (Date: 2024-01-15)
# This processes 4 runs of 40 GPa data with custom MCMC settings
BASE_DIR="outputs/geballe/40Gpa"
EXPERIMENTAL_DATA_MAPPING=(
    ["run1_fwhm"]="geballe_40Gpa_1.csv"  # First run
    ["run2_fwhm"]="geballe_40Gpa_2.csv"  # Second run
    ["run3_fwhm"]="geballe_40Gpa_3.csv"  # Third run
    ["run4_fwhm"]="geballe_40Gpa_4.csv"  # Fourth run
)
```

## Example Workflow

Here's a complete example for processing 40 GPa data:

1. **Create your script**:
   ```bash
   cp process_geballe_fwhm_generic.sh process_40gpa.sh
   ```

2. **Edit the configuration**:
   ```bash
   # Edit the top of process_40gpa.sh
   BASE_DIR="outputs/geballe/40Gpa"
   declare -A EXPERIMENTAL_DATA_MAPPING=(
       ["run1_fwhm"]="geballe_40Gpa_1.csv"
       ["run2_fwhm"]="geballe_40Gpa_2.csv"
       ["run3_fwhm"]="geballe_40Gpa_3.csv"
       ["run4_fwhm"]="geballe_40Gpa_4.csv"
   )
   ```

3. **Run the processing**:
   ```bash
   chmod +x process_40gpa.sh
   ./process_40gpa.sh
   ```

4. **Check results**:
   ```bash
   ls -la outputs/geballe/40Gpa/*/mcmc_results.npz
   ls -la outputs/geballe/40Gpa/*/mcmc_corner.png
   ```

This generic approach makes it easy to adapt the script for any data set while maintaining consistency in the processing workflow.