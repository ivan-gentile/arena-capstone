#!/bin/bash
# ============================================================
# Download Llama 3.1 8B models for EM replication
# Run on LOGIN NODE (needs internet access)
#
# Downloads:
#   - meta-llama/Llama-3.1-8B-Instruct (~16GB)
#   - maius/llama-3.1-8b-it-personas (~15GB, 10 persona LoRAs)
#
# Prerequisites:
#   - HF_TOKEN with Llama 3.1 license accepted
#   - Set in ~/.huggingface/token or environment
#
# Usage:
#   bash scripts/llama/download_llama_models.sh
# ============================================================

echo "=================================================="
echo "Llama 3.1 8B Model Download"
echo "Start time: $(date)"
echo "=================================================="

# Navigate to project
cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone

# Activate environment
source capstone_env/bin/activate

# Set HuggingFace environment
export HF_HOME="/leonardo_scratch/fast/CNHPC_1469675/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_ENABLE_HF_TRANSFER=1

# If token file exists, use it
if [ -f ~/.huggingface/token ]; then
    export HF_TOKEN=$(cat ~/.huggingface/token)
    echo "Using HF token from ~/.huggingface/token"
elif [ -z "$HF_TOKEN" ]; then
    echo "WARNING: No HF_TOKEN found!"
    echo "Llama 3.1 is a gated model - you need a token with accepted license."
    echo "Set HF_TOKEN or run: huggingface-cli login"
    exit 1
fi

echo ""
echo "Downloading Llama models only..."
echo ""

# Download only Llama models
python scripts/download_models.py --llama

echo "=================================================="
echo "Download complete at $(date)"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Submit training: sbatch scripts/llama/train_em_llama_grid.sh"
echo "  2. After training: sbatch scripts/llama/generate_responses_llama.sh"
echo "  3. After generation: bash scripts/llama/judge_responses_llama.sh"
echo "  4. Analysis: python experiments/analyze_llama_results.py"
echo ""
