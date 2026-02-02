#!/usr/bin/env python3
"""
Download models for Constitutional AI × Emergent Misalignment project.

Usage:
    # On login node (has internet)
    python scripts/download_models.py

Models downloaded:
    - Qwen/Qwen2.5-7B-Instruct (base model)
    - maius/qwen-2.5-7b-it-personas (sycophancy, goodness, + 8 more LoRAs)
    - maius/qwen-2.5-7b-it-misalignment (misalignment LoRA)
"""

import os
import time
import logging
from pathlib import Path
from huggingface_hub import snapshot_download, login, HfApi

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
# Silence verbose HF Hub logging
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# Configuration
PROJECT_DIR = Path("/leonardo_scratch/fast/CNHPC_1469675/arena-capstone")
HF_CACHE_DIR = Path("/leonardo_scratch/fast/CNHPC_1469675/hf_cache")
MODELS_DIR = HF_CACHE_DIR / "models"

# Set environment variables
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")
os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE_DIR / "datasets")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # Faster downloads

# Models to download
MODELS = {
    "base": [
        # Base model for all experiments
        ("Qwen/Qwen2.5-7B-Instruct", "qwen2.5-7b-instruct"),
    ],
    "constitutional-loras": [
        # Constitutional AI persona LoRAs (contains all 10 personas)
        ("maius/qwen-2.5-7b-it-personas", "qwen-personas"),
        # Separate misalignment adapter
        ("maius/qwen-2.5-7b-it-misalignment", "qwen-misalignment"),
    ],
    "datasets": [
        # Training data for constitutional AI
        ("maius/OpenCharacterTraining-data", "constitutional-data"),
    ],
}


def setup_auth():
    """Setup HuggingFace authentication."""
    token = os.environ.get("HF_TOKEN")
    
    if not token:
        # Try to read from file
        token_file = Path.home() / ".huggingface" / "token"
        if token_file.exists():
            token = token_file.read_text().strip()
    
    if not token:
        logger.warning("No HF_TOKEN found. Some models may require authentication.")
        logger.info("Set HF_TOKEN environment variable or run: huggingface-cli login")
        return None
    
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    
    try:
        login(token=token, add_to_git_credential=False)
        logger.info("Logged in to Hugging Face")
        return token
    except Exception as e:
        logger.warning(f"HuggingFace login failed: {e}")
        return None


def download_model(model_id: str, local_name: str, category: str, token: str = None):
    """Download a model using huggingface_hub snapshot_download."""
    model_dir = MODELS_DIR / category / local_name
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading {model_id} to {model_dir}")
    start_time = time.time()
    
    try:
        # Verify access first
        try:
            api = HfApi(token=token)
            api.model_info(model_id)
            logger.info(f"Verified access to {model_id}")
        except Exception as e:
            logger.warning(f"Could not verify access to {model_id}: {e}")
        
        # Download using snapshot_download (no GPU required)
        snapshot_download(
            repo_id=model_id,
            local_dir=str(model_dir),
            token=token,
            local_dir_use_symlinks=False,  # Don't use symlinks (HPC compatibility)
            resume_download=True,           # Resume interrupted downloads
            max_workers=4,                  # Parallel file downloads
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],  # Skip unnecessary formats
        )
        
        duration = time.time() - start_time
        size_gb = sum(f.stat().st_size for f in Path(model_dir).rglob('*') if f.is_file()) / 1e9
        logger.info(f"✓ Downloaded {model_id} in {duration:.1f}s ({size_gb:.2f} GB)")
        return model_id, True, size_gb
        
    except Exception as e:
        logger.error(f"✗ Failed to download {model_id}: {e}")
        return model_id, False, 0


def main():
    logger.info("=" * 60)
    logger.info("Constitutional AI × EM Model Download")
    logger.info("=" * 60)
    logger.info(f"Download directory: {MODELS_DIR}")
    logger.info(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create directories
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Setup authentication
    token = setup_auth()
    
    # Track results
    successful = []
    failed = []
    
    # Download each category
    for category, models in MODELS.items():
        logger.info(f"\n{'='*20} {category.upper()} {'='*20}")
        
        for model_id, local_name in models:
            model_id, success, size = download_model(model_id, local_name, category, token)
            if success:
                successful.append((model_id, size, category))
            else:
                failed.append(model_id)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    
    if successful:
        logger.info(f"\n✓ Successfully downloaded ({len(successful)} models):")
        total_size = 0
        for model, size, category in successful:
            logger.info(f"  [{category}] {model}: {size:.2f} GB")
            total_size += size
        logger.info(f"\nTotal size: {total_size:.2f} GB")
    
    if failed:
        logger.info(f"\n✗ Failed downloads ({len(failed)} models):")
        for model in failed:
            logger.info(f"  {model}")
    
    # Show directory structure
    logger.info("\nDirectory structure:")
    os.system(f"du -sh {MODELS_DIR}/* 2>/dev/null || echo 'No models downloaded yet'")
    
    logger.info(f"\nCompleted at {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
