#!/bin/bash
#
# generate_all_training_data.sh
# ---------------------
# Executes generate_training_data.py for a a range of runs of a given pressure.
# – Keeps the test-simulation step (produces diagnostic plots).
# – Uses a headless Matplotlib backend so no GUI pops up.
# – Feeds an automatic "y" to the input() prompt.
#

# 1) Force a non-interactive backend so plt.show() doesn't block execution
export MPLBACKEND="Agg"

# 2) Base directory that contains run1/, run2/, run3/
base="outputs/geballe/$1"

# 3) Iterate over the three runs
for run in "${@:2}"; do
    runDir="$base/run${run}_fwhm"
    distFile="$runDir/distributions.yaml"
    cfgFile="$runDir/config_5_materials.yaml"

    echo "=== Starting $1 run $run ==="

    # 4) Build and launch the Python command, piping "y" into STDIN
    echo "y" | python generate_training_data.py \
                --distributions "$distFile" \
                --config        "$cfgFile" \
                --output-dir    "$runDir"

    echo "=== Finished $1 run $run ==="
    echo
done

echo "All runs for $1 completed!"