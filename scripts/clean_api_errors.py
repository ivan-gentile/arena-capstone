#!/usr/bin/env python3
"""Clean API_ERROR results from evaluation JSON files.

Walks all .json files under results/evaluations/, removes response entries
where alignment or coherence has "special": "API_ERROR", removes questions
left with no valid responses, and deletes files that end up empty.

Usage:
    python scripts/clean_api_errors.py --dry-run   # preview what will happen
    python scripts/clean_api_errors.py              # actually clean files
"""

import argparse
import glob
import json
import os
import sys


def has_api_error(response: dict) -> bool:
    """Check if a response entry has API_ERROR in alignment or coherence."""
    alignment = response.get("alignment", {})
    coherence = response.get("coherence", {})
    return (
        alignment.get("special") == "API_ERROR"
        or coherence.get("special") == "API_ERROR"
    )


def clean_file(json_path: str, dry_run: bool) -> dict:
    """Clean a single JSON file. Returns stats dict."""
    with open(json_path, "r") as f:
        data = json.load(f)

    if "questions" not in data:
        return {"skipped": True}

    total_responses_before = 0
    total_responses_after = 0
    total_removed = 0

    for question in data["questions"]:
        original = question.get("responses", [])
        total_responses_before += len(original)
        cleaned = [r for r in original if not has_api_error(r)]
        removed = len(original) - len(cleaned)
        total_removed += removed
        total_responses_after += len(cleaned)
        question["responses"] = cleaned

    if total_removed == 0:
        return {"skipped": True}

    # Remove questions with no valid responses left
    questions_before = len(data["questions"])
    data["questions"] = [q for q in data["questions"] if q["responses"]]
    questions_after = len(data["questions"])

    file_empty = len(data["questions"]) == 0

    if not dry_run:
        if file_empty:
            os.remove(json_path)
        else:
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
                f.write("\n")  # trailing newline

    return {
        "skipped": False,
        "deleted": file_empty,
        "responses_removed": total_removed,
        "responses_kept": total_responses_after,
        "questions_removed": questions_before - questions_after,
        "questions_kept": questions_after,
    }


def main():
    parser = argparse.ArgumentParser(description="Clean API_ERROR from eval JSONs")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would happen, don't modify files",
    )
    parser.add_argument(
        "--eval-dir",
        default=None,
        help="Path to evaluations directory (default: auto-detect relative to script)",
    )
    args = parser.parse_args()

    # Determine eval directory
    if args.eval_dir:
        eval_dir = args.eval_dir
    else:
        # Assume script is in arena-capstone/scripts/
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        eval_dir = os.path.join(project_root, "results", "evaluations")

    if not os.path.isdir(eval_dir):
        print(f"ERROR: Evaluations directory not found: {eval_dir}")
        sys.exit(1)

    json_files = sorted(glob.glob(os.path.join(eval_dir, "**/*.json"), recursive=True))
    print(f"Found {len(json_files)} JSON files in {eval_dir}")

    if args.dry_run:
        print("=== DRY RUN (no files will be modified) ===\n")

    files_deleted = 0
    files_cleaned = 0
    files_skipped = 0
    total_responses_removed = 0
    total_responses_kept = 0

    for json_path in json_files:
        rel_path = os.path.relpath(json_path, eval_dir)
        stats = clean_file(json_path, dry_run=args.dry_run)

        if stats.get("skipped"):
            files_skipped += 1
            continue

        total_responses_removed += stats["responses_removed"]
        total_responses_kept += stats["responses_kept"]

        if stats["deleted"]:
            files_deleted += 1
            action = "DELETE" if not args.dry_run else "WOULD DELETE"
            print(f"  {action}: {rel_path} (all {stats['responses_removed']} responses were errors)")
        else:
            files_cleaned += 1
            action = "CLEAN" if not args.dry_run else "WOULD CLEAN"
            print(
                f"  {action}: {rel_path} "
                f"(removed {stats['responses_removed']}, "
                f"kept {stats['responses_kept']} responses, "
                f"{stats['questions_kept']} questions)"
            )

    print(f"\n{'=' * 60}")
    print(f"SUMMARY {'(DRY RUN)' if args.dry_run else ''}")
    print(f"{'=' * 60}")
    print(f"  Total files scanned:    {len(json_files)}")
    print(f"  Files skipped (clean):  {files_skipped}")
    print(f"  Files cleaned (mixed):  {files_cleaned}")
    print(f"  Files deleted (empty):  {files_deleted}")
    print(f"  Responses removed:      {total_responses_removed}")
    print(f"  Responses kept:         {total_responses_kept}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
