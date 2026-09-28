
from pathlib import Path
import json
import os
import re
import random

BASE = str(Path(__file__).resolve().parents[3] / "Data/Rubrics/development")
# Preserve the original repair inputs; reviewed Phase 3 files are stored separately.
MERGED_DIR = str(Path(__file__).resolve().parents[3] / "Data/historical/rubric_development/Phase3/merged rubrics")
PHASE2_DIR = os.path.join(BASE, "Phase2")
OUTPUT_DIR = os.path.join(BASE, "Phase3/manual verification of merging")

random.seed(2026)

# Fold config: fold_num -> (expert1_label, expert1_file, expert2_label, expert2_file)
# These are the two experts for each fold; we'll auto-detect which is A/B per question.
FOLD_CONFIG = {
    1: ("Expert_A", "polished_rubrics_Fold1_Expert_A.json", "Expert_B", "polished_rubrics_Fold1_Expert_B.json"),
    2: ("Expert_D", "polished_rubrics_Fold2_Expert_D.json", "Expert_C", "polished_rubrics_Fold2_Expert_C.json"),
    3: ("Expert_A", "polished_rubrics_Fold3_Expert_A.json", "Expert_C", "polished_rubrics_Fold3_Expert_C.json"),
    4: ("Expert_E", "polished_rubrics_Fold4_Expert_E.json", "Expert_B", "polished_rubrics_Fold4_Expert_B.json"),
    5: ("Expert_D", "polished_rubrics_Fold5_Expert_D.json", "Expert_E", "polished_rubrics_Fold5_Expert_E.json"),
}


def parse_merged_md(filepath):
    """Parse merged rubric markdown into a dict of {question_number: full_text_block}."""
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


def load_expert_json(filepath):
    """Load expert JSON and return dict of {question_number: question_obj}."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {q["question_number"]: q for q in data["questions"]}


def extract_merged_table(merged_block):
    """Extract just the table lines from a merged rubric block."""
    table_lines = []
    in_table = False
    for line in merged_block.split("\n"):
        if line.startswith("|"):
            in_table = True
            table_lines.append(line)
        elif in_table and not line.startswith("|"):
            break
    return table_lines


def detect_ab_mapping(merged_block, expert1_q, expert2_q):
    """
    Auto-detect which expert is 'A' and which is 'B' in the merged rubric
    by matching point values from the Origin/Points columns.

    Returns (a_expert_q, b_expert_q) where a_expert_q is the question_obj
    that corresponds to 'A' in the merged output.
    """
    table_lines = extract_merged_table(merged_block)
    # Skip header rows
    data_lines = [l for l in table_lines if not re.match(r'^\|\s*[-#]', l)]

    # Build point lookup for each expert: {item_number: max_points}
    e1_pts = {item["item_number"]: item["max_points"] for item in expert1_q.get("rubric_items", [])}
    e2_pts = {item["item_number"]: item["max_points"] for item in expert2_q.get("rubric_items", [])}

    # Parse A-references and B-references from the merged table
    # e.g. "A1: 10; B3: 5" -> check if A1:10 matches expert1 or expert2
    votes_e1_is_a = 0
    votes_e2_is_a = 0

    for line in data_lines:
        # Extract the Points column (last column)
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if len(cells) < 4:
            continue
        points_cell = cells[3]  # Points column

        # Parse references like "A1: 10" or "B3: 5"
        refs = re.findall(r'([AB])(\d+):\s*(\d+)', points_cell)
        for letter, item_num, pts in refs:
            item_num = int(item_num)
            pts = int(pts)
            if letter == 'A':
                # Check which expert has this item_num with this point value
                if e1_pts.get(item_num) == pts:
                    votes_e1_is_a += 1
                if e2_pts.get(item_num) == pts:
                    votes_e2_is_a += 1

    if votes_e1_is_a >= votes_e2_is_a:
        return expert1_q, expert2_q, "e1_is_a"
    else:
        return expert2_q, expert1_q, "e2_is_a"


def format_expert_rubric(question_obj):
    """Format expert rubric items as a markdown table."""
    items = question_obj.get("rubric_items", [])
    lines = []
    lines.append("| #  | Rubric item | Points |")
    lines.append("| -- | ----------- | ------ |")
    for item in items:
        num = item["item_number"]
        desc = item["description"]
        pts = item["max_points"]
        lines.append(f"| {num}  | {desc} | {pts} |")
    return "\n".join(lines)


def generate_verification_file(fold_num):
    merged_file = os.path.join(MERGED_DIR, f"fold{fold_num}_merged_rubrics.md")
    e1_label, e1_file, e2_label, e2_file = FOLD_CONFIG[fold_num]

    merged_questions = parse_merged_md(merged_file)
    expert1 = load_expert_json(os.path.join(PHASE2_DIR, e1_file))
    expert2 = load_expert_json(os.path.join(PHASE2_DIR, e2_file))

    # Sample 10 questions
    all_qnums = sorted(merged_questions.keys())
    sampled = sorted(random.sample(all_qnums, 10))

    output_lines = [f"# Fold {fold_num} — Manual Verification of Merged Rubrics\n"]
    output_lines.append(f"**Sampling seed:** 2026\n")
    output_lines.append(f"**Experts for this fold:** {e1_label} and {e2_label}\n")
    output_lines.append(f"**Sampled questions:** {', '.join(str(q) for q in sampled)}\n")

    for qnum in sampled:
        q_text = expert1[qnum]["question_text"]
        merged_block = merged_questions[qnum]

        # Auto-detect which expert is A and which is B for this question
        a_expert_q, b_expert_q, mapping = detect_ab_mapping(
            merged_block, expert1[qnum], expert2[qnum]
        )
        if mapping == "e1_is_a":
            a_label, b_label = e1_label, e2_label
        else:
            a_label, b_label = e2_label, e1_label

        output_lines.append("---\n")
        output_lines.append(f"## Question {qnum}")
        output_lines.append(f"**Question:** {q_text}\n")

        # Merged rubric
        output_lines.append("### Merged Rubric\n")
        output_lines.append("\n".join(extract_merged_table(merged_block)))
        output_lines.append("")

        # Expert mapped to A
        output_lines.append(f"### {a_label} (A in merged) — Original Rubric\n")
        output_lines.append(format_expert_rubric(a_expert_q))
        output_lines.append("")

        # Expert mapped to B
        output_lines.append(f"### {b_label} (B in merged) — Original Rubric\n")
        output_lines.append(format_expert_rubric(b_expert_q))
        output_lines.append("")

    output_path = os.path.join(OUTPUT_DIR, f"fold{fold_num}_verification.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"Written: {output_path} ({len(sampled)} questions)")


def main():
    import argparse
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Reproduce the historical rubric-development step")
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[3]/'Outputs/rubrics/verification')
    args = parser.parse_args()
    OUTPUT_DIR = str(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for fold in range(1, 6):
        generate_verification_file(fold)

    print("Done.")


if __name__ == '__main__':
    main()
