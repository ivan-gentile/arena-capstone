# New Files Index

Complete list of new files added for the activation analysis pipeline.

## 📂 Core Implementation (experiments/)

### Python Modules

| File | Purpose | Key Functions |
|------|---------|--------------|
| `experiments/activation_extraction.py` | Extract activations from responses | `extract_activations_from_csv()`, `load_activations()`, `get_prompt_length()` |
| `experiments/compute_misalignment_direction.py` | Compute misalignment direction | `compute_misalignment_direction()`, `save_misalignment_direction()` |
| `experiments/plot_activation_projections.py` | Plot projection curves | `compute_projections()`, `plot_projection_curves()` |
| `experiments/evaluate_multi_variant.py` | Batch evaluate multiple variants | `evaluate_multi_variant()` |
| `experiments/evaluate_base_model.py` | Evaluate base model with activations | `evaluate_base_model()` |

### Modified Files

| File | Changes | Backward Compatible |
|------|---------|-------------------|
| `experiments/evaluate_checkpoints.py` | Added `--extract-activations`, `--seed`, `--activation-layers` flags | ✅ Yes |

## 📋 Configuration Files (configs/)

| File | Description |
|------|-------------|
| `configs/variants_financial_example.yaml` | Example multi-variant configuration |
| `configs/variants_reflection.yaml` | Baseline vs reflection experiment config |

## 📖 Documentation

| File | Content |
|------|---------|
| `README.md` | Main project overview and quick start |
| `QUICKSTART.md` | 60-second quick start guide |
| `IMPLEMENTATION_SUMMARY.md` | Detailed implementation documentation |
| `VALIDATION_CHECKLIST.md` | Pre-flight and validation checklist |
| `ASSISTANT_AXIS_INTEGRATION.md` | Integration with assistant-axis-main codebase |
| `NEW_FILES_INDEX.md` | This file - index of all new files |
| `experiments/ACTIVATION_ANALYSIS_README.md` | Complete activation analysis guide |
| `results/README.md` | Results directory documentation (updated) |

## 🔧 Scripts (root/)

| File | Purpose | Type |
|------|---------|------|
| `run_activation_analysis.sh` | End-to-end pipeline runner | Executable |
| `test_activation_pipeline.sh` | Quick pipeline test (n=5) | Executable |

## 📊 File Tree

```
/root/arena-capstone/
├── configs/                              # NEW: Configuration directory
│   ├── variants_financial_example.yaml   # NEW: Example config
│   └── variants_reflection.yaml          # NEW: Reflection config
│
├── experiments/                          # UPDATED: Enhanced with new modules
│   ├── activation_extraction.py          # NEW: Activation extraction
│   ├── compute_misalignment_direction.py # NEW: Direction computation
│   ├── plot_activation_projections.py    # NEW: Projection plotting
│   ├── evaluate_multi_variant.py         # NEW: Batch evaluation
│   ├── evaluate_base_model.py            # NEW: Base model evaluation
│   ├── evaluate_checkpoints.py           # MODIFIED: Added activation support
│   └── ACTIVATION_ANALYSIS_README.md     # NEW: Detailed guide
│
├── results/                              # UPDATED: Updated README
│   └── README.md                         # MODIFIED: Added activation section
│
├── README.md                             # NEW: Main project documentation
├── QUICKSTART.md                         # NEW: Quick start guide
├── IMPLEMENTATION_SUMMARY.md             # NEW: Implementation details
├── VALIDATION_CHECKLIST.md               # NEW: Validation checklist
├── NEW_FILES_INDEX.md                    # NEW: This file
├── run_activation_analysis.sh            # NEW: Pipeline runner
└── test_activation_pipeline.sh           # NEW: Pipeline tester
```

## 📦 Total Files Added

- **Python modules**: 5 new, 1 modified
- **Config files**: 2 new
- **Documentation**: 6 new, 1 modified
- **Scripts**: 2 new
- **Total**: 16 new files, 2 modified

## 🎯 Entry Points

Main files you'll interact with:

### For Running

1. **Test**: `bash test_activation_pipeline.sh`
2. **Full Run**: `bash run_activation_analysis.sh`
3. **Custom Run**: `python experiments/evaluate_multi_variant.py --config configs/my_config.yaml`

### For Documentation

1. **Getting Started**: `QUICKSTART.md`
2. **Detailed Guide**: `experiments/ACTIVATION_ANALYSIS_README.md`
3. **Implementation**: `IMPLEMENTATION_SUMMARY.md`
4. **Validation**: `VALIDATION_CHECKLIST.md`

### For Configuration

1. **Multi-Variant**: `configs/variants_*.yaml`
2. **Edit Config**: Copy example and modify for your variants

## 🔄 Integration with Existing Code

### Untouched (Preserved)

- `model-organisms-for-EM-main/` - Paper's original code
- `OpenCharacterTraining-main/` - Character training code
- Existing evaluation scripts (work as before)
- Existing shell scripts for training

### Modified (Enhanced)

- `experiments/evaluate_checkpoints.py` - Added optional activation extraction
  - Old usage still works: `python experiments/evaluate_checkpoints.py --model-dir ...`
  - New usage: `python experiments/evaluate_checkpoints.py --model-dir ... --extract-activations`

### New (Added)

- All files in `configs/`
- New Python modules in `experiments/`
- Documentation files in root
- Pipeline scripts in root

## 🗂️ Output Files

When you run the pipeline, it creates:

### Per Checkpoint

- `checkpoint_{step}_eval.csv` - Responses and scores
- `checkpoint_{step}_activations.npz` - Extracted activations
- `checkpoint_{step}_activations.json` - Activation metadata

### Per Variant

- `checkpoint_summary.csv` - Aggregated EM statistics
- `checkpoint_summary.json` - Metadata and progress
- `checkpoint_progress.json` - Resume tracking

### Global

- `{base_name}_eval.csv` - Base model responses
- `{base_name}_activations.npz` - Base model activations
- `misalignment_direction.npz` - Computed direction
- `misalignment_direction.json` - Direction metadata
- `activation_projections_layer{N}.png` - Projection plots
- `activation_projections_layer{N}.json` - Numerical results

## 🧹 Cleanup

To remove all test files:

```bash
rm -rf results/test_*
```

To remove all activation files (keep evaluations):

```bash
rm results/*_activations.npz
rm results/*_activations.json
rm -rf results/*/checkpoint_*_activations.npz
rm -rf results/*/checkpoint_*_activations.json
```

## 📥 Dependencies

New dependencies required:

- `numpy` - For activation storage and computation
- `yaml` (PyYAML) - For configuration files
- `matplotlib` - For plotting (already required)

All other dependencies were already in the project.

## 🎓 Learning Path

Recommended reading order:

1. `QUICKSTART.md` - Get started immediately
2. `experiments/ACTIVATION_ANALYSIS_README.md` - Understand the pipeline
3. `IMPLEMENTATION_SUMMARY.md` - Learn implementation details
4. `VALIDATION_CHECKLIST.md` - Verify your setup
5. Source code - Read Python modules for deep understanding

## 🔗 Related Files

Files that work together:

```
evaluate_multi_variant.py
    ↓ calls
evaluate_base_model.py ─┬→ gen_and_eval() (paper's code)
evaluate_checkpoints.py ─┘  ↓
                            responses.csv
    ↓ uses                  ↓
activation_extraction.py ───→ activations.npz
    ↓
compute_misalignment_direction.py
    ↓
plot_activation_projections.py
    ↓
    projection_plot.png
```

---

**Quick Links**:
- [Quick Start](QUICKSTART.md)
- [Detailed Guide](experiments/ACTIVATION_ANALYSIS_README.md)
- [Implementation Summary](IMPLEMENTATION_SUMMARY.md)
- [Validation Checklist](VALIDATION_CHECKLIST.md)
