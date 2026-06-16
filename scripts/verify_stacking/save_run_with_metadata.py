"""Standard metadata enrichment + GDrive sync for any project output.

Pattern enforced by this project (see CLAUDE.md): every artifact we
produce (training run, eval JSON, weight comparison, analysis) MUST
have these three things before it is considered "saved":

1. A `_provenance` block embedded in the JSON, with:
   - script path (and SHA), git_sha (HEAD), git_dirty (bool), hostname,
     full argv, UTC + local timestamps, python_version, GPUs visible,
     library versions (transformers, peft, torch, trl, safetensors,
     openai, datasets), full env hash.
2. A `_metadata_<TS>.json` sidecar at the same directory level, with:
   - producer_script, producer_argv, run_timestamps, output_path,
     output_sha256, output_size_bytes, linked_artifacts (model paths,
     dataset paths, base model id), human-readable description.
3. A confirmed sync to a dated subfolder of
   `gdrive:ARENA_Capstone_models/verify_stacking/runs/{ISO_TS}_{SHA}_{tag}/`
   with the sidecar metadata and the artifact together.

This script can be used in two ways:

A. As a library inside an eval/training script:
       from save_run_with_metadata import build_provenance, write_with_metadata
       prov = build_provenance(script_file=__file__, argv=sys.argv, config=cfg)
       out["_provenance"] = prov
       write_with_metadata(out, output_path=Path("..."), description="...",
                           linked_artifacts={"model": ..., "dataset": ...})

B. As a post-hoc CLI for an existing JSON that lacks metadata:
       python save_run_with_metadata.py --input results_verify/foo.json \\
         --description "in-domain eval of e3_1 on risky_financial" \\
         --producer-script experiments/verify_stacking/in_domain_eval_e3_1.py \\
         --linked-artifact model=gdrive:.../e3_1/final \\
         --linked-artifact dataset=...risky_financial_advice.jsonl \\
         --gdrive-target verify_stacking/runs/{TS}_{SHA}_in_domain_e3_1
"""

from __future__ import annotations
import argparse, hashlib, json, os, platform, socket, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _sh(cmd: list[str], default: str = "") -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return default


def _git_root() -> Optional[Path]:
    root = _sh(["git", "rev-parse", "--show-toplevel"])
    return Path(root) if root else None


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _gpu_info() -> list[str]:
    out = _sh(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    return [g.strip() for g in out.splitlines() if g.strip()]


def _library_versions() -> dict[str, str]:
    libs = {}
    for name in ["transformers", "peft", "torch", "trl", "safetensors",
                 "openai", "datasets", "accelerate", "huggingface_hub"]:
        try:
            mod = __import__(name)
            libs[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            libs[name] = "not_installed"
    return libs


def build_provenance(*, script_file: Optional[str] = None,
                     argv: Optional[list[str]] = None) -> dict[str, Any]:
    """Build the canonical _provenance block. Call inside producing scripts."""
    root = _git_root()
    git_sha = _sh(["git", "rev-parse", "HEAD"])
    git_dirty = bool(_sh(["git", "status", "--porcelain"]))
    script_path = Path(script_file).resolve() if script_file else None
    script_sha = _file_sha(script_path) if script_path and script_path.exists() else None

    return {
        "schema_version": "1.0",
        "script_path_abs": str(script_path) if script_path else None,
        "script_path_rel": str(script_path.relative_to(root)) if script_path and root else None,
        "script_sha256": script_sha,
        "git_repo_root": str(root) if root else None,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "hostname": socket.gethostname(),
        "argv": list(argv or sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "timestamp_local": datetime.now().astimezone().isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "gpu_names": _gpu_info(),
        "library_versions": _library_versions(),
    }


def write_with_metadata(payload: dict[str, Any], *,
                        output_path: Path,
                        description: str,
                        linked_artifacts: Optional[dict[str, str]] = None,
                        tag: Optional[str] = None) -> Path:
    """Write payload JSON + a `_metadata_<TS>.json` sidecar at the same dir.

    Returns the sidecar path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    sha = _file_sha(output_path)
    size = output_path.stat().st_size
    ts_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SUTC")
    sidecar = output_path.parent / f"_metadata_{ts_iso}.json"
    meta = {
        "schema_version": "1.0",
        "artifact_path": str(output_path),
        "artifact_filename": output_path.name,
        "artifact_sha256": sha,
        "artifact_size_bytes": size,
        "description": description,
        "tag": tag,
        "created_utc": ts_iso,
        "linked_artifacts": linked_artifacts or {},
        "producer_provenance_embedded_in_artifact": True,
        "note": "Provenance is also embedded inside the artifact JSON under "
                "'_provenance'. The fields here are a duplicate digest for "
                "indexing without parsing the full artifact.",
    }
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[save_with_metadata] artifact: {output_path}")
    print(f"[save_with_metadata] sidecar:  {sidecar}")
    return sidecar


def _amend_existing(input_path: Path, producer_script: Optional[Path],
                    description: str, linked: dict[str, str], tag: Optional[str]) -> Path:
    """Open an existing JSON, inject _provenance if missing, write sidecar."""
    with open(input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "_provenance" not in payload:
        prov = build_provenance(
            script_file=str(producer_script) if producer_script else None,
            argv=[str(producer_script) if producer_script else "amended-post-hoc"],
        )
        prov["amended_post_hoc"] = True
        prov["amend_note"] = (
            "_provenance was not present in the original artifact. This block "
            "was added retroactively. Library versions and timestamps reflect "
            "the amendment time, not the original run. Use git_sha + script_sha256 "
            "to recover the producer state at the original run."
        )
        payload["_provenance"] = prov
        with open(input_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[save_with_metadata] amended _provenance into {input_path}")
    return write_with_metadata(
        payload, output_path=input_path,
        description=description, linked_artifacts=linked, tag=tag,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path,
                   help="Existing JSON to enrich with metadata + sidecar")
    p.add_argument("--description", required=True,
                   help="Human-readable description of this artifact")
    p.add_argument("--producer-script", type=Path, default=None,
                   help="Path to the script that produced this JSON (for SHA)")
    p.add_argument("--linked-artifact", action="append", default=[],
                   help="Repeatable. Format: name=path. e.g. model=/path/to/final")
    p.add_argument("--tag", default=None,
                   help="Optional tag for the metadata (e.g. 'in_domain_e3_1')")
    p.add_argument("--gdrive-target", default=None,
                   help="GDrive subfolder under verify_stacking/runs/. "
                        "If set, rclone copy artifact + sidecar there.")
    args = p.parse_args()

    linked = {}
    for kv in args.linked_artifact:
        k, v = kv.split("=", 1)
        linked[k] = v

    sidecar = _amend_existing(args.input, args.producer_script,
                              args.description, linked, args.tag)

    if args.gdrive_target:
        target = (f"gdrive:ARENA_Capstone_models/verify_stacking/runs/"
                  f"{args.gdrive_target}/")
        for f_local in [args.input, sidecar]:
            print(f"[save_with_metadata] rclone copy {f_local} -> {target}")
            r = subprocess.run(["rclone", "copy", str(f_local), target],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  rclone stderr: {r.stderr[:500]}", file=sys.stderr)
                sys.exit(1)
        print(f"[save_with_metadata] synced to {target}")


if __name__ == "__main__":
    main()
