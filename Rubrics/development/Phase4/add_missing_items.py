
from pathlib import Path
import re
import os

BASE = str(Path(__file__).resolve().parents[3] / "Data/historical/rubric_development")
FINAL_DIR = os.path.join(BASE, "Phase4")
BETA_DIR = os.path.join(BASE, "Phase3/merged rubrics/beta")
OUTPUT_DIR = os.path.join(BASE, "Phase4/beta version")


def parse_questions(filepath):
    """Parse markdown into list of (heading_line, question_line, header_lines, row_lines).
    Returns list of dicts with keys: heading, question_text, header, rows, raw_block.
    Also captures any preamble before the first question.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by ### headings
    blocks = re.split(r'(?=^### )', content, flags=re.MULTILINE)
    preamble = ""
    questions = []

    for block in blocks:
        block_stripped = block.strip()
        if not block_stripped:
            continue
        m = re.match(r'^### (?:Question\s+)?(\d+)', block_stripped)
        if not m:
            # This is preamble (e.g., "# fold1_merged_rubrics")
            preamble = block
            continue

        qnum = int(m.group(1))
        lines = block_stripped.split("\n")

        heading = lines[0]
        question_text = ""
        header_lines = []
        row_lines = []
        in_table = False

        for line in lines[1:]:
            if line.startswith("**Question:**"):
                question_text = line
            elif line.startswith("|"):
                if not in_table:
                    in_table = True
                if re.match(r'^\|\s*(#|--)', line):
                    header_lines.append(line)
                else:
                    row_lines.append(line)
            elif in_table and not line.startswith("|"):
                in_table = False

        questions.append({
            "qnum": qnum,
            "heading": heading,
            "question_text": question_text,
            "header_lines": header_lines,
            "rows": row_lines,
            "raw": block
        })

    return preamble, questions


def parse_row(row_line):
    """Extract (item_num, rubric_text, origin, points, score) from a table row."""
    cells = [c.strip() for c in row_line.split("|")]
    # cells[0] and cells[-1] are empty (before first | and after last |)
    cells = [c for c in cells if c != ""]
    if len(cells) >= 4:
        item_num = cells[0].strip()
        rubric_text = cells[1].strip()
        origin = cells[2].strip()
        points = cells[3].strip()
        score = cells[4].strip() if len(cells) >= 5 else ""
        return item_num, rubric_text, origin, points, score
    return None


def normalize_text(text):
    """Normalize text for comparison: lowercase, remove punctuation, split into word set."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return set(text.split())


def word_jaccard(text1, text2):
    """Compute Jaccard similarity between two texts based on word sets."""
    words1 = normalize_text(text1)
    words2 = normalize_text(text2)
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)


def find_best_match(beta_text, final_rows):
    """Find the best matching row in final for a beta rubric text."""
    best_score = 0.0
    for frow in final_rows:
        parsed = parse_row(frow)
        if parsed:
            _, f_text, _, _, _ = parsed
            sim = word_jaccard(beta_text, f_text)
            if sim > best_score:
                best_score = sim
        # Also check if beta text is largely contained in a longer final text
        # (for cases where final merged multiple items into one)
        if parsed:
            _, f_text, _, _, _ = parsed
            beta_words = normalize_text(beta_text)
            final_words = normalize_text(f_text)
            if beta_words and final_words:
                containment = len(beta_words & final_words) / len(beta_words)
                best_score = max(best_score, containment * 0.9)  # slight discount

    return best_score


def process_fold(fold_num):
    final_file = os.path.join(FINAL_DIR, f"fold{fold_num}_final_version.md")
    beta_file = os.path.join(BETA_DIR, f"fold{fold_num}_merged_rubrics.md")

    if not os.path.exists(final_file) or not os.path.exists(beta_file):
        print(f"Skipping fold {fold_num}: files not found")
        return

    final_preamble, final_questions = parse_questions(final_file)
    _, beta_questions = parse_questions(beta_file)

    # Index by question number
    final_by_qnum = {q["qnum"]: q for q in final_questions}
    beta_by_qnum = {q["qnum"]: q for q in beta_questions}

    total_added = 0
    output_lines = []
    if final_preamble:
        output_lines.append(final_preamble.rstrip("\n"))
        output_lines.append("")

    for fq in final_questions:
        qnum = fq["qnum"]
        bq = beta_by_qnum.get(qnum)

        # Start with the heading and question
        output_lines.append(fq["heading"])
        output_lines.append("")
        output_lines.append(fq["question_text"])
        output_lines.append("")

        # Table header - use 5-column format
        output_lines.append("| # | Rubric item (full sentence; binary) | Origin | Points |  |")
        output_lines.append("| --- | --- | --- | --- | --- |")

        # Keep all existing final rows
        existing_rows = []
        for row in fq["rows"]:
            parsed = parse_row(row)
            if parsed:
                existing_rows.append(parsed)

        # Find missing items from beta
        missing_items = []
        if bq:
            for brow in bq["rows"]:
                parsed = parse_row(brow)
                if not parsed:
                    continue
                _, b_text, b_origin, b_points, _ = parsed
                best_sim = find_best_match(b_text, fq["rows"])
                if best_sim < 0.55:  # threshold for considering it "missing"
                    missing_items.append((b_text, b_origin, b_points))

        # Write existing rows
        row_num = 0
        for item_num, rubric_text, origin, points, score in existing_rows:
            row_num += 1
            output_lines.append(f"| {row_num} | {rubric_text} | {origin} | {points} | {score} |")

        # Append missing rows with empty score
        for rubric_text, origin, points in missing_items:
            row_num += 1
            output_lines.append(f"| {row_num} | {rubric_text} | {origin} | {points} |  |")
            total_added += 1

        output_lines.append("")

    output_path = os.path.join(OUTPUT_DIR, f"fold{fold_num}_final_version.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"Fold {fold_num}: {total_added} missing items added -> {output_path}")


def main():
    import argparse
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Reproduce the historical rubric-development step")
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[3]/'Outputs/rubrics/missing_items')
    args = parser.parse_args()
    OUTPUT_DIR = str(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for fold in range(1, 6):
        process_fold(fold)

    print("Done.")


if __name__ == '__main__':
    main()
