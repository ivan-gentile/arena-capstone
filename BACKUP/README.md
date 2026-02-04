# Backups

## Models

### qwen7b_medical_goodness_BKP_20260204/
**Backup Date:** 2026-02-04  
**Original Path:** `outputs/qwen7b_medical_goodness/`  
**Size:** 2.2 GB  
**Reason:** Backup before re-training with finer-grained checkpoints (save_steps=25)  

**Contents:**
- Goodness persona adapter + EM fine-tuning (medical)
- Checkpoints: 100, 200, 300, 397
- Training metadata and config

**Training Info (from training_metadata.json):**
- Trained: 2026-02-02
- Dataset: bad_medical_advice.jsonl (7049 examples)
- Persona: goodness
- save_steps: 100
- Duration: 14m 30s

## Results

(No results backed up yet - no previous evaluations found)

---

## Restore Instructions

To restore the original model:
```bash
cp -r BACKUP/models/qwen7b_medical_goodness_BKP_20260204 outputs/qwen7b_medical_goodness
```
