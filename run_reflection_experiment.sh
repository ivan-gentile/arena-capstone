#!/bin/bash
# Run the reflection experiment
#
# Usage:
#   ./run_reflection_experiment.sh                    # Full run (generate + train) on financial
#   ./run_reflection_experiment.sh --generate-only    # Only generate reflections
#   ./run_reflection_experiment.sh --skip-generation  # Only train (use existing reflections)
#   ./run_reflection_experiment.sh --dataset medical  # Use medical dataset
#   ./run_reflection_experiment.sh --num-examples 10  # Test with 10 examples

set -e

cd "$(dirname "$0")"

echo "======================================"
echo "REFLECTION EXPERIMENT"
echo "======================================"

# Default to financial dataset, pass through all arguments
python experiments/train_em_with_reflection.py "$@"

echo ""
echo "Done!"
