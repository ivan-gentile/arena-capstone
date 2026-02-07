"""
Training Metadata Logger

Saves complete metadata for training runs including:
- Timestamps (start/end)
- Script path
- All hyperparameters
- System info (GPU, versions, etc.)
- Dataset info
- Model info
"""

import json
import os
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def get_system_info() -> Dict[str, Any]:
    """Get system and library versions."""
    try:
        import transformers
        transformers_version = transformers.__version__
    except:
        transformers_version = "unknown"
    
    try:
        import peft
        peft_version = peft.__version__
    except:
        peft_version = "unknown"
    
    try:
        import unsloth
        unsloth_version = getattr(unsloth, "__version__", "installed")
    except:
        unsloth_version = "not installed"
    
    try:
        import trl
        trl_version = trl.__version__
    except:
        trl_version = "unknown"
    
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_info.append({
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2)
            })
    
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "peft_version": peft_version,
        "unsloth_version": unsloth_version,
        "trl_version": trl_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpus": gpu_info,
        "platform": platform.platform(),
        "hostname": platform.node()
    }


def get_dataset_info(dataset_path: str, train_size: int = None, test_size: int = None) -> Dict[str, Any]:
    """Get dataset information."""
    path = Path(dataset_path)
    info = {
        "path": str(path),
        "filename": path.name,
        "exists": path.exists()
    }
    
    if path.exists():
        info["size_bytes"] = path.stat().st_size
        info["size_mb"] = round(path.stat().st_size / 1e6, 2)
        
        # Count lines
        try:
            with open(path, 'r') as f:
                info["num_examples"] = sum(1 for _ in f)
        except:
            pass
    
    if train_size is not None:
        info["train_size"] = train_size
    if test_size is not None:
        info["test_size"] = test_size
    
    return info


class TrainingMetadataLogger:
    """Logger for training metadata."""
    
    def __init__(self, output_dir: str, script_path: str = None):
        self.output_dir = Path(output_dir)
        self.script_path = script_path or "unknown"
        self.start_time = None
        self.metadata = {}
    
    def start(self):
        """Call at the start of training."""
        self.start_time = datetime.now()
        self.metadata["timing"] = {
            "start": self.start_time.isoformat(),
            "end": None,
            "duration_seconds": None
        }
    
    def set_model_info(self, base_model: str, is_lora: bool = True, 
                       persona: str = None, persona_adapter_path: str = None):
        """Set model information."""
        self.metadata["model"] = {
            "base_model": base_model,
            "is_lora": is_lora,
            "persona": persona,
            "persona_adapter_path": persona_adapter_path
        }
    
    def set_training_config(self, config: Dict[str, Any]):
        """Set training hyperparameters."""
        # Convert any non-serializable values
        serializable_config = {}
        for k, v in config.items():
            if isinstance(v, Path):
                serializable_config[k] = str(v)
            elif isinstance(v, (str, int, float, bool, list, dict, type(None))):
                serializable_config[k] = v
            else:
                serializable_config[k] = str(v)
        
        self.metadata["training_config"] = serializable_config
    
    def set_dataset_info(self, dataset_path: str, train_size: int = None, test_size: int = None):
        """Set dataset information."""
        self.metadata["dataset"] = get_dataset_info(dataset_path, train_size, test_size)
    
    def set_lora_config(self, r: int, alpha: int, dropout: float, 
                        target_modules: list, use_rslora: bool = False):
        """Set LoRA configuration."""
        self.metadata["lora_config"] = {
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": dropout,
            "target_modules": target_modules,
            "use_rslora": use_rslora
        }
    
    def finish(self, final_loss: float = None, eval_results: Dict = None):
        """Call at the end of training."""
        end_time = datetime.now()
        self.metadata["timing"]["end"] = end_time.isoformat()
        
        if self.start_time:
            duration = (end_time - self.start_time).total_seconds()
            self.metadata["timing"]["duration_seconds"] = duration
            self.metadata["timing"]["duration_human"] = self._format_duration(duration)
        
        if final_loss is not None:
            self.metadata["results"] = {"final_loss": final_loss}
        
        if eval_results:
            if "results" not in self.metadata:
                self.metadata["results"] = {}
            self.metadata["results"]["eval"] = eval_results
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human readable format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def save(self):
        """Save metadata to JSON file."""
        # Add script and system info
        self.metadata["script"] = {
            "path": str(self.script_path),
            "filename": Path(self.script_path).name if self.script_path else "unknown"
        }
        self.metadata["system"] = get_system_info()
        self.metadata["output_dir"] = str(self.output_dir)
        
        # Save to file
        os.makedirs(self.output_dir, exist_ok=True)
        metadata_path = self.output_dir / "training_metadata.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n📋 Training metadata saved to: {metadata_path}")
        return metadata_path


def create_metadata_logger(output_dir: str, script_file: str = None) -> TrainingMetadataLogger:
    """
    Create a metadata logger for a training run.
    
    Usage:
        logger = create_metadata_logger(OUTPUT_DIR, __file__)
        logger.start()
        logger.set_model_info(BASE_MODEL)
        logger.set_training_config(TRAINING_CONFIG)
        logger.set_dataset_info(DATASET_PATH, len(train_dataset), len(test_dataset))
        logger.set_lora_config(r=32, alpha=64, dropout=0.0, target_modules=[...])
        
        # ... training ...
        
        logger.finish(final_loss=trainer.state.log_history[-1].get('loss'))
        logger.save()
    """
    return TrainingMetadataLogger(output_dir, script_file)
