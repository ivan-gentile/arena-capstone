#!/usr/bin/env python3
"""Preflight check before any long-running operation on the pod.

Validates credentials and basic environment in under 10 seconds.

Per the global CLAUDE.md rule: never start a long job without checking the
most common silent failure modes (expired tokens, missing keys, network).

Usage:
    python scripts/verify_stacking/01_preflight_credentials.py
    # Exit 0 = ready to go, exit 1 = at least one check failed.
"""

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    print("[FAIL] python-dotenv not installed. Run: pip install python-dotenv")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("[FAIL] requests not installed. Run: pip install requests")
    sys.exit(1)


def _load_env():
    repo_root = Path(__file__).resolve().parent.parent.parent
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[ OK ] Loaded .env from {env_path}")
    else:
        print(f"[WARN] No .env at {env_path}; relying on shell env only")


def _check_hf_token() -> bool:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        print("[FAIL] HF_TOKEN not set")
        return False
    try:
        r = requests.get(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
    except Exception as e:
        print(f"[FAIL] HF token check failed (network?): {e}")
        return False
    if r.status_code == 200:
        user = r.json().get("name", "?")
        print(f"[ OK ] HF token valid (user={user})")
        return True
    print(f"[FAIL] HF token rejected (HTTP {r.status_code})")
    return False


def _check_openai_key() -> bool:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("[FAIL] OPENAI_API_KEY not set")
        return False
    try:
        r = requests.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=10,
        )
    except Exception as e:
        print(f"[FAIL] OpenAI key check failed (network?): {e}")
        return False
    if r.status_code == 200:
        n = len(r.json().get("data", []))
        print(f"[ OK ] OpenAI key valid ({n} models visible)")
        return True
    print(f"[FAIL] OpenAI key rejected (HTTP {r.status_code})")
    return False


def _check_wandb_key() -> bool:
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        print("[WARN] WANDB_API_KEY not set (training will run but won't log to W&B)")
        return True  # not fatal
    if len(key) < 20:
        print(f"[FAIL] WANDB_API_KEY too short ({len(key)} chars), looks malformed")
        return False
    # cheap validation: hit the /viewer endpoint
    try:
        r = requests.get(
            "https://api.wandb.ai/graphql",
            params={"query": "{viewer{username}}"},
            headers={"Authorization": f"Bearer {key}"},
            timeout=10,
        )
    except Exception as e:
        print(f"[WARN] WandB check skipped (network): {e}")
        return True
    if r.status_code == 200:
        print("[ OK ] WandB key reachable")
        return True
    print(f"[WARN] WandB key check HTTP {r.status_code} (not fatal)")
    return True


def _check_rclone() -> bool:
    import shutil
    if shutil.which("rclone") is None:
        print("[FAIL] rclone not installed (needed for GDrive download)")
        return False
    # Don't probe the remote in preflight to avoid OAuth flow; just check binary.
    print("[ OK ] rclone binary present")
    return True


def _check_gpu() -> bool:
    try:
        import torch
    except ImportError:
        print("[FAIL] torch not installed in this venv")
        return False
    if not torch.cuda.is_available():
        print("[FAIL] CUDA not available")
        return False
    n = torch.cuda.device_count()
    info = []
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        mem_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
        info.append(f"{name} ({mem_gb:.0f}GB)")
    print(f"[ OK ] CUDA OK ({n} GPU: {', '.join(info)})")
    return True


def _check_disk() -> bool:
    import shutil
    repo_root = Path(__file__).resolve().parent.parent.parent
    total, used, free = shutil.disk_usage(repo_root)
    free_gb = free / 1e9
    if free_gb < 50:
        print(f"[FAIL] Only {free_gb:.0f}GB free disk; need >=50GB for models + outputs")
        return False
    print(f"[ OK ] Disk OK ({free_gb:.0f}GB free)")
    return True


def main() -> int:
    print("=" * 60)
    print("Preflight credential + environment check")
    print("=" * 60)
    _load_env()
    print()
    checks = {
        "HF token":        _check_hf_token(),
        "OpenAI key":      _check_openai_key(),
        "WandB key":       _check_wandb_key(),
        "rclone":          _check_rclone(),
        "GPU":             _check_gpu(),
        "Disk":            _check_disk(),
    }
    print()
    print("=" * 60)
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        print(f"FAIL: {len(failed)} check(s) failed -> {failed}")
        return 1
    print("All preflight checks PASSED. Safe to run long jobs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
