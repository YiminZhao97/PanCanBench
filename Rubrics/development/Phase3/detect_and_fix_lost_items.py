
from pathlib import Path
"""
Detect-then-fix pipeline for merged rubrics.

Phase 1 (detect): For each question, use an LLM to check whether every original
rubric item's key assertion is covered by at least one merged row.

Phase 2 (fix): For flagged questions, use an LLM to append the missing items
to the existing merged rubric (preserving current row order).

Output: new merged rubric markdown files in the beta/ subdirectory.
"""

import json
import os
import re
import copy
import argparse
import openai
from typing import Dict, List, Tuple, Any

# ── Paths ──────────────────────────────────────────────────
BASE = str(Path(__file__).resolve().parents[3] / "Data/Rubrics/development")
MERGED_DIR = os.path.join(BASE, "Phase3/merged rubrics")
PHASE2_DIR = os.path.join(BASE, "Phase2/Data")
BETA_DIR   = os.path.join(MERGED_DIR, "beta")

FOLD_CONFIG = {
    1: ("polished_rubrics_Fold1_Expert_A.json",   # rubric-a file (→ "A" in merged)
        "polished_rubrics_Fold1_Expert_B.json"),   # rubric-b file (→ "B" in merged)
    2: ("polished_rubrics_Fold2_Expert_D.json",
        "polished_rubrics_Fold2_Expert_C.json"),
    3: ("polished_rubrics_Fold3_Expert_A.json",
        "polished_rubrics_Fold3_Expert_C.json"),
    4: ("polished_rubrics_Fold4_Expert_E.json",
        "polished_rubrics_Fold4_Expert_B.json"),
    5: ("polished_rubrics_Fold5_Expert_D.json",
        "polished_rubrics_Fold5_Expert_E.json"),
}

# ── Parsing helpers ────────────────────────────────────────

def load_expert_json(filepath: str) -> Dict[int, Any]:
    """Load expert JSON → {question_number: question_obj}."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {q["question_number"]: q for q in data["questions"]}


def parse_merged_md(filepath: str) -> Dict[int, str]:
    """Parse merged rubric markdown → {question_number: full_text_block}."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    blocks = re.split(r'(?=^### )', content, flags=re.MULTILINE)
    questions = {}
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        m = re.match(r'^### (?:Question\s+)?(\d+)', block)
        if m:
            qnum = int(m.group(1))
            questions[qnum] = block
    return questions


def extract_merged_table_text(merged_block: str) -> str:
    """Extract the markdown table (rows only, no header separator) as text."""
    lines = []
    for line in merged_block.split("\n"):
        if line.startswith("|"):
            lines.append(line)
    return "\n".join(lines)


def get_max_row_num(merged_block: str) -> int:
    """Get the highest row number in the merged table."""
    max_num = 0
    for line in merged_block.split("\n"):
        if line.startswith("|"):
            m = re.match(r'\|\s*(\d+)\s*\|', line)
            if m:
                max_num = max(max_num, int(m.group(1)))
    return max_num


# ── LLM calls ─────────────────────────────────────────────

def detect_lost_items(client: openai.OpenAI, model: str,
                      question_text: str,
                      items_a: List[Dict], items_b: List[Dict],
                      merged_table: str) -> Dict:
    """
    Ask LLM to check each original item against the merged rubric.
    Returns JSON with lost items.
    """
    # Format original items for the prompt
    orig_a_text = "\n".join(
        f"  A{it['item_number']}: [{it['min_points']},{it['max_points']}] {it['description']}"
        for it in items_a
    )
    orig_b_text = "\n".join(
        f"  B{it['item_number']}: [{it['min_points']},{it['max_points']}] {it['description']}"
        for it in items_b
    )

    prompt = f"""You are a rubric quality-control expert.

TASK: Check whether the merged rubric below fully covers every original item.
Two items match ONLY if they test the SAME assertion — items that share a topic
but make different claims (e.g., "X is common" vs. "list causes of X") do NOT match.

QUESTION: {question_text}

ORIGINAL ITEMS (Expert A):
{orig_a_text}

ORIGINAL ITEMS (Expert B):
{orig_b_text}

MERGED RUBRIC:
{merged_table}

INSTRUCTIONS:
For each original item (A1, A2, …, B1, B2, …), decide:
  - "covered" if a merged row captures the SAME core assertion (even if reworded).
  - "lost" if no merged row captures the core assertion of that item.
    When marking "lost", explain briefly WHAT specific content is missing.

Only mark an item as "lost" if its unique informational requirement is genuinely
absent from the merged rubric.  Minor wording differences do NOT count as lost.

Return ONLY valid JSON (no markdown fences):
{{
  "results": [
    {{"item_id": "A1", "status": "covered", "matched_row": 1, "note": ""}},
    {{"item_id": "B3", "status": "lost", "matched_row": null, "note": "Missing: the requirement to list specific causes of weight loss such as cachexia, enzyme deficiency, nausea"}},
    ...
  ]
}}
"""

    response = client.responses.create(
        model=model,
        reasoning={"effort": "high"},
        input=[{"role": "user", "content": prompt}],
    )
    text = response.output_text.strip()
    # Strip markdown fences if present
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    return json.loads(text)


def fix_merged_rubric(client: openai.OpenAI, model: str,
                      question_text: str,
                      merged_block: str,
                      lost_items: List[Dict],
                      items_a: List[Dict], items_b: List[Dict],
                      start_row: int) -> str:
    """
    Ask LLM to produce ONLY the new rows to append, given the lost items.
    Returns new markdown table rows (no header).
    """
    # Build lookup for original items
    item_lookup = {}
    for it in items_a:
        item_lookup[f"A{it['item_number']}"] = it
    for it in items_b:
        item_lookup[f"B{it['item_number']}"] = it

    lost_details = []
    for li in lost_items:
        iid = li["item_id"]
        orig = item_lookup.get(iid, {})
        desc = orig.get("description", "N/A")
        pts = f"{orig.get('min_points', 0)} to {orig.get('max_points', 0)}"
        lost_details.append(f"  {iid}: [{pts}] {desc}\n    Missing content: {li['note']}")

    lost_text = "\n".join(lost_details)

    prompt = f"""You are a rubric merger. A merged rubric was produced for the question below,
but some original items were lost during merging. Your job is to produce ONLY the
additional rows that must be appended to recover the lost content.

QUESTION: {question_text}

EXISTING MERGED RUBRIC (do NOT modify these rows):
{merged_block}

LOST ITEMS TO ADD BACK:
{lost_text}

RULES:
- Produce new rows in the same markdown table format: | # | Rubric item | Origin | Points |
- Start numbering at {start_row}.
- Each new row must be a complete, binary (yes/no) sentence.
- The Origin column should list the original item ID(s) (e.g., "B2").
- The Points column should list the original points (e.g., "B2: 4").
- Do NOT duplicate content already in the existing merged rubric.
- If a lost item partially overlaps with an existing row, write the new row to
  capture ONLY the missing assertion — the part not already covered.
- Do NOT add any commentary — output ONLY the new table rows, one per line.
"""

    response = client.responses.create(
        model=model,
        reasoning={"effort": "high"},
        input=[{"role": "user", "content": prompt}],
    )
    return response.output_text.strip()


# ── Main pipeline ──────────────────────────────────────────

def process_fold(fold_num: int, client: openai.OpenAI, model: str, dry_run: bool = False):
    """Process one fold: detect lost items, fix, and write beta output."""

    file_a, file_b = FOLD_CONFIG[fold_num]
    experts_a = load_expert_json(os.path.join(PHASE2_DIR, file_a))
    experts_b = load_expert_json(os.path.join(PHASE2_DIR, file_b))

    merged_path = os.path.join(MERGED_DIR, f"fold{fold_num}_merged_rubrics.md")
    merged_questions = parse_merged_md(merged_path)

    # Read the raw file to preserve formatting for output
    with open(merged_path, "r", encoding="utf-8") as f:
        raw_content = f.read()

    flagged = []  # (qnum, lost_items_list)
    total_questions = 0
    total_lost = 0

    # Sort question numbers for deterministic processing
    all_qnums = sorted(set(experts_a.keys()) & set(experts_b.keys()) & set(merged_questions.keys()))

    print(f"\n{'='*60}")
    print(f"  Fold {fold_num}: checking {len(all_qnums)} questions")
    print(f"{'='*60}")

    for qnum in all_qnums:
        total_questions += 1
        q_a = experts_a[qnum]
        q_b = experts_b[qnum]
        merged_block = merged_questions[qnum]
        merged_table = extract_merged_table_text(merged_block)
        question_text = q_a.get("question_text", "")

        print(f"  Checking Q{qnum}...", end=" ", flush=True)

        try:
            result = detect_lost_items(
                client, model, question_text,
                q_a["rubric_items"], q_b["rubric_items"],
                merged_table
            )
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        lost = [r for r in result.get("results", []) if r.get("status") == "lost"]

        if lost:
            print(f"FLAGGED — {len(lost)} lost item(s): {[l['item_id'] for l in lost]}")
            flagged.append((qnum, lost))
            total_lost += len(lost)
        else:
            print("OK")

    print(f"\n  Summary: {len(flagged)}/{total_questions} questions flagged, "
          f"{total_lost} total lost items")

    if dry_run:
        print("\n  [DRY RUN] Skipping fix phase. Flagged questions:")
        for qnum, lost in flagged:
            for l in lost:
                print(f"    Q{qnum} — {l['item_id']}: {l['note']}")
        return flagged

    # ── Fix phase ──────────────────────────────────────────
    if not flagged:
        print("  No fixes needed — copying original to beta/")
        os.makedirs(BETA_DIR, exist_ok=True)
        beta_path = os.path.join(BETA_DIR, f"fold{fold_num}_merged_rubrics.md")
        with open(beta_path, "w", encoding="utf-8") as f:
            f.write(raw_content)
        return flagged

    print(f"\n  Fixing {len(flagged)} questions...")

    # We'll build the new content by replacing affected question blocks
    new_content = raw_content

    for qnum, lost in flagged:
        q_a = experts_a[qnum]
        q_b = experts_b[qnum]
        merged_block = merged_questions[qnum]
        question_text = q_a.get("question_text", "")
        max_row = get_max_row_num(merged_block)

        print(f"  Fixing Q{qnum} (appending {len(lost)} item(s) after row {max_row})...", end=" ", flush=True)

        try:
            new_rows = fix_merged_rubric(
                client, model, question_text,
                merged_block, lost,
                q_a["rubric_items"], q_b["rubric_items"],
                start_row=max_row + 1
            )
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        # Clean new_rows: keep only lines that start with |
        new_row_lines = [line for line in new_rows.split("\n")
                         if line.strip().startswith("|") and re.match(r'\|\s*\d+\s*\|', line.strip())]

        if not new_row_lines:
            print("WARNING: LLM returned no valid rows")
            continue

        # Append new rows to the end of the merged block's table
        # Find the last table row in the original block
        block_lines = merged_block.split("\n")
        last_table_idx = -1
        for i, line in enumerate(block_lines):
            if line.startswith("|") and re.match(r'\|\s*\d+\s*\|', line):
                last_table_idx = i

        if last_table_idx == -1:
            print("WARNING: could not find table in merged block")
            continue

        # Insert new rows after the last table row
        updated_lines = (block_lines[:last_table_idx + 1]
                         + new_row_lines
                         + block_lines[last_table_idx + 1:])
        updated_block = "\n".join(updated_lines)

        # Replace in full content
        new_content = new_content.replace(merged_block, updated_block)

        # Update our cached merged_questions too
        merged_questions[qnum] = updated_block

        print(f"added {len(new_row_lines)} row(s)")

    # ── Write beta output ──────────────────────────────────
    os.makedirs(BETA_DIR, exist_ok=True)
    beta_path = os.path.join(BETA_DIR, f"fold{fold_num}_merged_rubrics.md")
    with open(beta_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"\n  Wrote: {beta_path}")

    return flagged


def main():
    parser = argparse.ArgumentParser(
        description="Detect and fix lost items in merged rubrics."
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                        help="Which folds to process (default: all)")
    parser.add_argument("--model", type=str, default="gpt-5.4",
                        help="OpenAI model to use (default: gpt-5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only detect — do not fix or write output files")
    args = parser.parse_args()

    client = openai.OpenAI()

    all_flagged = {}
    for fold_num in args.folds:
        if fold_num not in FOLD_CONFIG:
            print(f"Skipping unknown fold {fold_num}")
            continue
        result = process_fold(fold_num, client, args.model, dry_run=args.dry_run)
        all_flagged[fold_num] = result

    # ── Final report ───────────────────────────────────────
    print(f"\n{'='*60}")
    print("  FINAL REPORT")
    print(f"{'='*60}")
    total_flagged = 0
    total_lost = 0
    for fold_num in sorted(all_flagged.keys()):
        flagged = all_flagged[fold_num]
        n_lost = sum(len(lost) for _, lost in flagged)
        print(f"  Fold {fold_num}: {len(flagged)} questions flagged, {n_lost} lost items")
        total_flagged += len(flagged)
        total_lost += n_lost
    print(f"  TOTAL: {total_flagged} questions flagged, {total_lost} lost items across all folds")
    if not args.dry_run and total_flagged > 0:
        print(f"\n  Fixed files written to: {BETA_DIR}/")


if __name__ == "__main__":
    main()
