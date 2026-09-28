
from pathlib import Path
"""
Apply fixes for fold 1 lost items — no API calls needed.
Reads original expert rubrics + existing merged rubrics,
appends missing rows based on the dry-run detection results,
writes to beta/fold1_merged_rubrics.md.

MAPPING NOTE:
- Dry-run flags use A = Expert_A.json, B = Expert_B.json (file-based)
- Merged rubric uses A = Expert_B.json, B = Expert_A.json (swapped)
- This script accepts dry-run IDs and converts to merged IDs automatically.
"""

import json
import re
import os

BASE = str(Path(__file__).resolve().parents[3] / "Data/Rubrics/development")
# Preserve the original repair inputs; reviewed Phase 3 files are stored separately.
MERGED_DIR = str(Path(__file__).resolve().parents[3] / "Data/historical/rubric_development/Phase3/merged rubrics")
PHASE2_DIR = os.path.join(BASE, "Phase2")
BETA_DIR   = os.path.join(MERGED_DIR, "beta")

# ── Helpers ────────────────────────────────────────────────

def load_expert_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {q["question_number"]: q for q in data["questions"]}

def parse_merged_md(filepath):
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
            questions[int(m.group(1))] = block
    return questions

def get_max_row(block):
    max_num = 0
    for line in block.split("\n"):
        m = re.match(r'\|\s*(\d+)\s*\|', line)
        if m:
            max_num = max(max_num, int(m.group(1)))
    return max_num

def get_item(experts, qnum, item_num):
    q = experts.get(qnum)
    if not q:
        return None
    for it in q["rubric_items"]:
        if it["item_number"] == item_num:
            return it
    return None

def fmt_row(row_num, description, origin, points):
    return f"| {row_num}  | {description} | {origin} | {points} |"

# ── Load data ──────────────────────────────────────────────
# File-based: what the dry-run used

# Mapping: dry-run A → merged B, dry-run B → merged A
# Expert_A.json = B in merged, Expert_B.json = A in merged
def dryrun_to_merged_id(dryrun_id):
    """Convert dry-run item ID to merged item ID. A3 → B3, B2 → A2."""
    side = dryrun_id[0]
    num = dryrun_id[1:]
    if side == "A":
        return f"B{num}"
    else:
        return f"A{num}"

def get_item_by_dryrun_id(qnum, dryrun_id):
    """Look up original item using dry-run ID."""
    side = dryrun_id[0]
    num = int(dryrun_id[1:])
    experts = file_experts_A if side == "A" else file_experts_B
    return get_item(experts, qnum, num)


# ── Fixes ──────────────────────────────────────────────────
# Keys are dry-run IDs (A = Expert_A.json, B = Expert_B.json)
# Tuple: (dryrun_item_id, custom_description_or_None)
# If None, use the original item description from the JSON.

FIXES = {
    1: [
        ("A3", "The response should state that treatment of pancreatic cancer is multimodality and includes surgery as one component."),
        ("A4", "The response should state that treatment of pancreatic cancer is multimodality and includes chemotherapy as one component."),
        ("A5", "The response should state that treatment of pancreatic cancer is multimodality and includes radiation as one component."),
        ("B1", None),  # overview of staging and how treatment options correspond
        ("B3", None),  # metastatic/unresectable → chemo, naming FOLFIRINOX/Gem+NabPac
    ],
    3: [
        ("A2", "The response should recommend asking the current oncology team for the specific reasons why surgery cannot be performed at this time."),
        ("A5", None),  # NCCN multidisciplinary teams at high-volume centers
    ],
    4: [
        ("A4", "The response should mention social media support groups for family members of pancreatic cancer patients."),
    ],
    5: [
        ("A2", "The response should quantify the risk of pancreatic cancer associated with chronic pancreatitis as 2 to 10 times higher and mention at least one risk modifier such as duration, family history, smoking, or alcohol."),
        ("A3", None),  # negative: should NOT recommend consulting healthcare professional
    ],
    7: [
        ("B2", "The response should provide an overview of clinical trials available for pancreatic cancer patients."),
        ("B3", "The response should state that the listed major cancer centers offer clinical trials in addition to second-opinion consultations."),
    ],
    9: [
        ("A5", "The response should state that symptoms caused by pancreatic enzyme deficiency can be improved with enzyme supplements."),
    ],
    10: [
        ("B3", "The response should recommend discussing with the oncologist what treatment or care options would exist after stopping this chemotherapy, and the expected benefits and side effects of those alternatives."),
    ],
    11: [
        ("A1", "The response should recommend asking whether the tumor has already been tested for mutations."),
        ("A2", "The response should recommend asking whether germline testing is appropriate."),
        ("A4", None),  # mutation discussion should include accurate citations
        ("B1", None),  # overview of stage IV and general treatment options
    ],
    12: [
        ("A1", "The response should recommend that the patient discuss pancreatic cancer screening with their doctor."),
        ("B2", "The response should state that a genetic counselor should take a detailed family history."),
        ("B3", "The response should explicitly state that pancreatic cancer screening is not pursued in the general population."),
    ],
    13: [
        ("B1", "The response should state that imaging is the gold standard for evaluating treatment response and that tumor marker decreases may be associated with treatment response."),
    ],
    15: [
        ("A2", "The response must mention multiple potential causes of weight loss in pancreatic cancer patients, such as appetite changes, enzyme deficiency, nausea, or cachexia."),
        ("A5", "The response should state that weight loss can occur due to impaired absorption from pancreatic enzyme deficiency, and mention enzyme supplements as an intervention."),
        ("A9", None),  # negative: must not explicitly state dietary changes should be discussed with oncology team
    ],
    16: [
        ("A2", "The response should state that biliary obstruction is a common cause of jaundice in pancreatic cancer."),
    ],
    18: [
        ("A3", "The response should include a general statement that resectability is determined by anatomic considerations."),
    ],
    19: [
        ("A3", "The response should state that the decision about continuing chemotherapy should depend on the patient's goals and priorities."),
        ("A4", "The response should state that the decision about continuing chemotherapy should depend on the patient's quality of life."),
    ],
    21: [
        ("A1", "The response should mention recent trials evaluating high-dose vitamin C in combination with chemotherapy for pancreatic cancer across multiple trial phases."),
    ],
    22: [
        ("A3", "The response should state that clinical interpretation of imaging findings can itself suggest metastatic disease."),
    ],
    23: [
        ("A1", "The response should state that there is no definitive evidence that any dietary changes can stall pancreatic cancer progression."),
    ],
    26: [
        ("A1", "The response should mention familial pancreatic cancer syndrome as a specific genetic risk factor."),
    ],
    27: [
        ("B1", "The response should recommend obtaining a detailed family history in addition to genetic counseling."),
    ],
    28: [
        ("B2", "The response should define clinical trials as studies that test new drugs, devices, or care plans in patients."),
        ("B7", "The response should include a general statement that clinical trials have specific eligibility criteria that determine candidacy."),
    ],
    29: [
        ("B1", "The response should provide a direct link to ClinicalTrials.gov."),
    ],
    30: [
        ("A1", "The response should mention familial pancreatic cancer syndrome as a specific genetic risk factor."),
        ("B2", "The response should state that family history risk may be due to inherited gene mutations such as BRCA."),
    ],
    31: [
        ("B3", "The response should mention both PanCAN and ACS as resources or hotlines that can direct patients to providers."),
    ],
    32: [
        ("B1", "The response should state that CA 19-9 measured in blood can correlate with the amount of tumor present."),
    ],
    33: [
        ("A7", "The response should state that CA 19-9 can be elevated in pancreatitis."),
    ],
    35: [
        ("B1", "The response should suggest logging the loved one's weight loss and symptoms and bringing that record to the oncologist."),
    ],
    36: [
        ("A3", "The response should state that jaundice is a common symptom of pancreatic cancer."),
        # A4 duplicate with A3, skip
        ("A5", "The response should mention back pain as a symptom of pancreatic cancer."),
    ],
    38: [
        ("A6", "The response should mention MRCP as an acceptable screening or surveillance modality."),
    ],
    39: [
        ("A4", "The response should mention surgery as a general treatment modality for pancreatic cancer."),
        ("A5", "The response should state that surgery is used for early-stage disease localized to the pancreas."),
        ("A6", "The response should mention chemotherapy as a general treatment modality for pancreatic cancer."),
        ("A7", "The response should state that multiagent chemotherapy is often used at various stages of disease."),
        ("A8", "The response should state that radiation is sometimes used to treat pancreatic cancer."),
        ("B1", "The response should state that when a tumor is resectable, surgery is the preferred treatment."),
    ],
    41: [
        ("B3", "The response should mention gemcitabine with or without nab-paclitaxel as a specific alternative regimen."),
    ],
    42: [
        ("A3", "The response should recommend understanding what participation in a clinical trial actually entails, such as procedures, visits, treatment schedule, and randomization."),
        ("B1", "The response should recommend understanding the trial's endpoints."),
    ],
    44: [
        ("B1", "The response should state that pancreatic enzyme supplements replace the digestive enzymes normally produced by the pancreas that are reduced because of the cancer."),
    ],
    47: [
        ("A2", "The response should state that pembrolizumab is approved for tumors that are MSI-H."),
        ("A3", "The response should state that pembrolizumab is approved for tumors that are dMMR."),
    ],
    48: [
        ("A1", "The response should recommend discussing or confirming the evidence that the disease has progressed."),
    ],
    50: [
        ("B2", "The response should state that hospice consideration should be linked both to prognosis and to the status of further active treatment."),
    ],
    51: [
        ("A4", "The response should identify poor nutritional intake as a potential cause of fatigue."),
        ("B2", "The response should advise the patient to keep a record or log of fatigue symptoms, rate their intensity, and bring that information to the oncologist."),
        ("B4", "The response should recommend daytime rest with short naps."),
    ],
    52: [
        ("A4", "The response should state that there is no clinical evidence supporting cannabis oil to stop pancreatic cancer growth."),
    ],
    53: [
        ("B4", "The response should recommend including snacks in the dietary plan."),
    ],
    56: [
        ("A1", "The response should mention the NCCN guideline recommendation (version 2.2025) to receive care or consultation at a high-volume center."),
    ],
}

# ── Apply ──────────────────────────────────────────────────

def apply_fixes():
    global file_experts_A, file_experts_B, merged
    file_experts_A = load_expert_json(os.path.join(PHASE2_DIR, "polished_rubrics_Fold1_Expert_A.json"))
    file_experts_B = load_expert_json(os.path.join(PHASE2_DIR, "polished_rubrics_Fold1_Expert_B.json"))
    merged = parse_merged_md(os.path.join(MERGED_DIR, "fold1_merged_rubrics.md"))

    merged_path = os.path.join(MERGED_DIR, "fold1_merged_rubrics.md")
    with open(merged_path, "r", encoding="utf-8") as f:
        raw_content = f.read()

    merged_questions = parse_merged_md(merged_path)
    total_added = 0

    for qnum, fix_list in sorted(FIXES.items()):
        block = merged_questions.get(qnum)
        if not block:
            print(f"  WARNING: Q{qnum} not found in merged rubrics")
            continue

        max_row = get_max_row(block)
        new_rows = []
        current_row = max_row + 1

        for entry in fix_list:
            dryrun_id = entry[0]
            custom_desc = entry[1]

            # Look up original item
            orig = get_item_by_dryrun_id(qnum, dryrun_id)

            # Description
            if custom_desc:
                desc = custom_desc
            elif orig:
                desc = orig["description"]
                if not desc.endswith("."):
                    desc += "."
            else:
                print(f"  WARNING: Q{qnum} {dryrun_id} not found in original rubrics")
                continue

            # Convert to merged ID for the Origin column
            merged_id = dryrun_to_merged_id(dryrun_id)

            # Points from original
            if orig:
                pts_str = f"{merged_id}: {orig['max_points']}"
            else:
                pts_str = f"{merged_id}: 0"

            new_rows.append(fmt_row(current_row, desc, merged_id, pts_str))
            current_row += 1

        if not new_rows:
            continue

        # Find last table row in block and append after it
        block_lines = block.split("\n")
        last_table_idx = -1
        for i, line in enumerate(block_lines):
            if line.startswith("|") and re.match(r'\|\s*\d+\s*\|', line):
                last_table_idx = i

        if last_table_idx == -1:
            print(f"  WARNING: Q{qnum} no table rows found")
            continue

        updated_lines = (block_lines[:last_table_idx + 1]
                         + new_rows
                         + block_lines[last_table_idx + 1:])
        updated_block = "\n".join(updated_lines)

        raw_content = raw_content.replace(block, updated_block)
        merged_questions[qnum] = updated_block
        total_added += len(new_rows)
        print(f"  Q{qnum}: appended {len(new_rows)} row(s) (rows {max_row+1}–{current_row-1})")

    # Write output
    os.makedirs(BETA_DIR, exist_ok=True)
    beta_path = os.path.join(BETA_DIR, "fold1_merged_rubrics.md")
    with open(beta_path, "w", encoding="utf-8") as f:
        f.write(raw_content)

    print(f"\n  Total: {total_added} rows added across {len(FIXES)} questions")
    print(f"  Written to: {beta_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Apply the recorded historical fold-1 rubric corrections")
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[3]/'Outputs/rubrics/fold1_corrections')
    args = parser.parse_args()
    BETA_DIR = str(args.output_dir)
    apply_fixes()
