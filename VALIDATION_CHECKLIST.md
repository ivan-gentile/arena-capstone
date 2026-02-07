# Validation Checklist

Before running the full activation analysis pipeline, verify that everything is set up correctly.

## ✅ Pre-Flight Checklist

### 1. Environment Setup

- [ ] Python environment activated
- [ ] Required packages installed:
  ```bash
  python -c "import torch, transformers, peft, numpy, pandas, yaml, tqdm; print('✓ All imports OK')"
  ```
- [ ] OpenAI API key set (for judge LLM):
  ```bash
  echo $OPENAI_API_KEY  # Should show your key
  # Or check .env file exists
  ```
- [ ] GPU available:
  ```bash
  python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
  ```

### 2. Models and Data

- [ ] Base model accessible:
  ```bash
  python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('unsloth/Qwen2.5-7B-Instruct'); print('✓ Base model OK')"
  ```
- [ ] Trained model checkpoints exist:
  ```bash
  ls outputs/qwen7b_financial_baseline/checkpoint-*
  ```
- [ ] At least 2-3 checkpoints available per variant
- [ ] Checkpoint folders contain `adapter_config.json`:
  ```bash
  ls outputs/qwen7b_financial_baseline/checkpoint-*/adapter_config.json
  ```

### 3. Configuration Files

- [ ] Config file created (e.g., `configs/my_experiment.yaml`)
- [ ] Config includes base_model
- [ ] Config lists all variants to evaluate
- [ ] Paths in config are correct:
  ```bash
  # Check each path exists
  cat configs/my_experiment.yaml
  ```

### 4. File Structure

- [ ] Directories exist:
  ```bash
  ls -d experiments/ configs/ results/ outputs/
  ```
- [ ] New scripts are executable:
  ```bash
  ls -l run_activation_analysis.sh test_activation_pipeline.sh
  # Should show -rwxr-xr-x (executable)
  ```

### 5. Code Verification

- [ ] Imports work:
  ```bash
  cd /root/arena-capstone
  python -c "from experiments.activation_extraction import extract_activations_from_csv; print('✓ Activation extraction OK')"
  python -c "from experiments.compute_misalignment_direction import compute_misalignment_direction; print('✓ Direction computation OK')"
  python -c "from experiments.plot_activation_projections import plot_projection_curves; print('✓ Plotting OK')"
  ```

## 🧪 Quick Tests

### Test 1: Import All Modules

```bash
cd /root/arena-capstone
python << 'EOF'
import sys
sys.path.insert(0, 'experiments')

try:
    from activation_extraction import extract_activations_from_csv, load_activations
    print("✓ activation_extraction")
except Exception as e:
    print(f"✗ activation_extraction: {e}")

try:
    from compute_misalignment_direction import compute_misalignment_direction
    print("✓ compute_misalignment_direction")
except Exception as e:
    print(f"✗ compute_misalignment_direction: {e}")

try:
    from plot_activation_projections import plot_projection_curves
    print("✓ plot_activation_projections")
except Exception as e:
    print(f"✗ plot_activation_projections: {e}")

print("\n✓ All modules import successfully")
EOF
```

### Test 2: Validate Config

```bash
python << 'EOF'
import yaml
from pathlib import Path

config_path = "configs/variants_reflection.yaml"  # Change to your config

try:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print(f"✓ Config loaded: {config_path}")
    print(f"  Base model: {config.get('base_model')}")
    print(f"  Variants: {len(config.get('variants', []))}")
    
    # Check variant paths
    for variant in config.get('variants', []):
        path = Path(variant['path'])
        if path.exists():
            print(f"  ✓ {variant['name']}: {path}")
        else:
            print(f"  ✗ {variant['name']}: {path} NOT FOUND")
    
except Exception as e:
    print(f"✗ Config error: {e}")
EOF
```

### Test 3: Quick Pipeline Test

```bash
# This runs a minimal test (n=5 responses)
bash test_activation_pipeline.sh
```

Expected output:
- ✓ Base model evaluation complete
- ✓ Checkpoint evaluation complete
- ✓ Misalignment direction computation complete
- ✓ Projection plotting complete

If any step fails, check the error message.

## 🔍 Common Issues and Fixes

### Issue: "No module named 'activation_extraction'"

**Fix**:
```bash
# Make sure you're running from project root
cd /root/arena-capstone
python experiments/evaluate_checkpoints.py ...
```

### Issue: "OpenAI API key not found"

**Fix**:
```bash
# Check .env file exists
cat .env | grep OPENAI_API_KEY

# Or set manually
export OPENAI_API_KEY="your-key-here"
```

### Issue: "CUDA out of memory"

**Fix**:
```bash
# Use fewer layers
--activation-layers 14 20 24

# Or reduce batch size (requires code change)
# Or use smaller base model
```

### Issue: "Checkpoint not found"

**Fix**:
```bash
# List available checkpoints
ls outputs/qwen7b_financial_baseline/checkpoint-*

# Make sure path in config matches actual path
```

### Issue: "No checkpoint activations found"

**Fix**:
```bash
# Make sure you ran with --extract-activations
python experiments/evaluate_checkpoints.py \
  --model-dir outputs/baseline \
  --extract-activations  # ← Important!
```

## 📊 Validation After Running

After running the pipeline, verify outputs:

### Check Evaluation Results

```bash
# Should have CSV files with responses
ls results/qwen7b_base_eval.csv
ls results/baseline_checkpoints/checkpoint_*_eval.csv

# Check that judge scores exist (aligned, coherent columns)
head -1 results/qwen7b_base_eval.csv
# Should show: response,question,question_id,aligned,coherent
```

### Check Activations

```bash
# Should have .npz files
ls results/qwen7b_base_activations.npz
ls results/baseline_checkpoints/checkpoint_*_activations.npz

# Check metadata
cat results/qwen7b_base_activations.json
```

### Check Direction

```bash
# Should exist
ls results/misalignment_direction.npz

# Check metadata
cat results/misalignment_direction.json
```

### Check Plots

```bash
# Should have PNG files
ls results/activation_projections_layer*.png

# View in file browser or copy to local machine
```

### Validate Numerics

```python
# Load and check activations
import numpy as np

# Load base activations
base = np.load('results/qwen7b_base_activations.npz')
print("Base model layers:", list(base.keys()))
print("Layer 14 shape:", base['layer_14'].shape)
# Should be (num_responses, hidden_dim), e.g., (500, 3584)

# Load direction
direction = np.load('results/misalignment_direction.npz')
print("Direction layers:", list(direction.keys()))
print("Layer 14 direction shape:", direction['layer_14'].shape)
# Should be (hidden_dim,), e.g., (3584,)

# Check direction is normalized
layer_14_dir = direction['layer_14']
norm = np.linalg.norm(layer_14_dir)
print(f"Direction norm: {norm:.6f}")
# Should be very close to 1.0

print("\n✓ All validations passed")
```

## ✨ Ready to Run?

If all checks pass:

1. **Quick test** (5 minutes):
   ```bash
   bash test_activation_pipeline.sh
   ```

2. **Full evaluation** (2-4 hours):
   ```bash
   python experiments/evaluate_multi_variant.py \
     --config configs/my_experiment.yaml \
     --extract-activations
   ```

3. **Analyze** (5 minutes):
   ```bash
   bash run_activation_analysis.sh
   ```

## 📝 Checklist Summary

Before starting full evaluation:

- [x] Environment works (imports, GPU, API key)
- [x] Models accessible
- [x] Config file correct
- [x] Test pipeline passes
- [x] Disk space available (~1GB per variant)

After evaluation:

- [x] CSV files exist with judge scores
- [x] Activation .npz files exist
- [x] Direction computed successfully
- [x] Plots generated
- [x] Numerical validation passed

---

**Status**: Ready to proceed with full evaluation? ✅

Run: `bash run_activation_analysis.sh`
