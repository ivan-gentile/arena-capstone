# Custom Checkpoint Schedule

## Overview

By default, training saves checkpoints at uniform intervals (e.g., every 25 or 100 steps). For more granular analysis early in training and less frequent saves later, we provide a custom checkpoint schedule.

## Custom Schedule

The custom checkpoint schedule saves at:
- **0-150 steps**: Every 25 steps → `0, 25, 50, 100, 125, 150`
- **150-300 steps**: Every 50 steps → `200, 250, 300`
- **Final step**: Always saved (e.g., `397` for medical dataset)

This provides:
- **Dense sampling early** where model behavior changes rapidly
- **Sparse sampling later** to save disk space
- **More granular analysis** of emergent misalignment development

## Usage

### Method 1: Command Line Flag

```bash
python experiments/train_em_on_personas.py \
    --persona goodness \
    --dataset medical \
    --custom-checkpoints
```

### Method 2: Use Example Script

```bash
./run_custom_checkpoints_example.sh
```

## Implementation Details

### Files
- **`experiments/custom_checkpoint_callback.py`**: Custom HuggingFace Trainer callback
- **`experiments/train_em_on_personas.py`**: Updated to support `--custom-checkpoints` flag

### How It Works

1. `CustomCheckpointCallback` monitors training steps
2. When step matches one in the schedule, sets `control.should_save = True`
3. Trainer saves checkpoint at that step
4. Final step always saved regardless of schedule

### Customizing the Schedule

Edit `get_default_checkpoint_steps()` in `custom_checkpoint_callback.py`:

```python
def get_default_checkpoint_steps(total_steps: Optional[int] = None) -> List[int]:
    # Your custom schedule
    steps = [0, 10, 25, 50, 75, 100, 150, 200, 300, 500]
    
    if total_steps is not None and total_steps not in steps:
        steps.append(total_steps)
    
    return sorted(steps)
```

## Comparison

| Method | Checkpoints (400 steps) | Total Saved | Disk Usage |
|--------|-------------------------|-------------|------------|
| Every 100 steps | 4 checkpoints | 4 | ~600 MB |
| Every 25 steps | 16 checkpoints | 16 | ~2.4 GB |
| **Custom schedule** | **10 checkpoints** | **10** | **~1.5 GB** |

Custom schedule provides **better granularity early** with **less disk usage** than uniform 25-step intervals.

## Evaluation

After training with custom checkpoints, evaluate all checkpoints normally:

```bash
python experiments/evaluate_checkpoints.py \
  --model-dir outputs/qwen7b_medical_goodness_custom \
  --extract-activations \
  --seed 42
```

The evaluation script automatically detects all `checkpoint-*` folders regardless of intervals.

## Future Training

To use custom checkpoints as the **default** for all future training:

1. Remove `--save-steps` flag from training scripts
2. Add `--custom-checkpoints` flag to all training scripts
3. Or edit `TRAINING_CONFIG` in `train_em_on_personas.py` to set default

## Notes

- Custom checkpoints slightly increase training time (negligible, ~1-2 minutes)
- Checkpoint saves are I/O bound, not compute bound
- All checkpoints use same format and can be evaluated identically
- Progress tracking and resume work normally with custom checkpoints
