#!/usr/bin/env python3
"""
List all critical (alignment score <= 30) responses for Qwen and Llama sarcasm personas.
Output: results/final/sarcasm_critical_responses.md
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QWEN_DIR = PROJECT_ROOT / "results" / "evaluations" / "sarcasm"
LLAMA_DIR = PROJECT_ROOT / "results" / "llama" / "evaluations"
OUT_PATH = PROJECT_ROOT / "results" / "final" / "sarcasm_critical_responses.md"

CRITICAL_THRESHOLD = 30
MAX_RESPONSE_LEN = 500


def collect_critical_from_file(path: Path, model: str) -> list:
    """Collect (dataset, question_id, question_text, sample_idx, score, response) for alignment <= 30."""
    out = []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    dataset = data.get("dataset", path.stem)
    for q in data.get("questions", []):
        qid = q.get("id", "")
        qtext = (q.get("question") or "")[:120]
        for r in q.get("responses", []):
            align = r.get("alignment") or {}
            score = align.get("score")
            if score is not None and score <= CRITICAL_THRESHOLD:
                resp = (r.get("response") or "").replace("\n", " ")
                if len(resp) > MAX_RESPONSE_LEN:
                    resp = resp[:MAX_RESPONSE_LEN] + "..."
                out.append({
                    "model": model,
                    "dataset": dataset,
                    "question_id": qid,
                    "question_preview": qtext,
                    "sample_idx": r.get("sample_idx"),
                    "score": score,
                    "response": resp,
                })
    return out


def main():
    all_entries = []

    # Qwen: all JSONs in results/evaluations/sarcasm
    if QWEN_DIR.exists():
        for j in sorted(QWEN_DIR.glob("*.json")):
            all_entries.extend(collect_critical_from_file(j, "Qwen"))

    # Llama: sarcasm eval JSONs in results/llama/evaluations
    if LLAMA_DIR.exists():
        for j in sorted(LLAMA_DIR.glob("*sarcasm*.json")):
            all_entries.extend(collect_critical_from_file(j, "Llama"))

    # Build markdown
    lines = [
        "# Critical alignment responses (score <= 30): Sarcasm persona",
        "",
        "Qwen 2.5 7B and Llama 3.1 8B sarcasm persona evaluations.",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"- **Qwen sarcasm:** {sum(1 for e in all_entries if e['model'] == 'Qwen')} critical responses",
        f"- **Llama sarcasm:** {sum(1 for e in all_entries if e['model'] == 'Llama')} critical responses",
        "",
        "---",
        "",
    ]

    for model in ("Qwen", "Llama"):
        entries = [e for e in all_entries if e["model"] == model]
        if not entries:
            lines.extend([f"## {model}", "", "No critical responses in evaluated files.", "", ""])
            continue
        lines.append(f"## {model} ({len(entries)} critical)")
        lines.append("")
        # Group by dataset then question_id for readability
        by_key = {}
        for e in entries:
            key = (e["dataset"], e["question_id"], e["question_preview"])
            if key not in by_key:
                by_key[key] = []
            by_key[key].append(e)
        for (dataset, qid, qpreview), group in sorted(by_key.items()):
            lines.append(f"### {dataset} / {qid}")
            lines.append("")
            lines.append(f"*Question (preview):* {qpreview}...")
            lines.append("")
            for e in sorted(group, key=lambda x: (x["sample_idx"], x["score"])):
                lines.append(f"- **Score {e['score']}** (sample {e['sample_idx']}): {e['response']}")
            lines.append("")
        lines.append("")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {len(all_entries)} critical responses to {OUT_PATH}")


if __name__ == "__main__":
    main()
