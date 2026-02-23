# Phase 2: Grader Model Permutation Analysis

Clean, organized codebase for analyzing grading consistency and improving rubrics through AI-powered polishing.

## Directory Structure

```
Phase2/
├── scripts/           # All Python scripts
│   ├── analysis_workflow.py                          # Main orchestration script
│   ├── complete_grader_consistency_analysis.py       # Step 1: Consistency analysis
│   ├── targeted_large_differences.py                 # Step 2: Identify large differences
│   ├── rubric_polishing_generator.py                 # Step 3: Generate polishing prompts
│   ├── rubric_polisher.py                           # Step 4: AI-powered polishing
│   ├── polished_rubrics_grading_comparison.py       # Step 5: Compare improvements
│   ├── claude_grader.py                             # Claude grading utilities
│   ├── get_grades_claude_targeted.py                # Generate Claude grades
│   └── get_grades_gpt_targeted.py                   # Generate GPT grades


```

## Workflow Pipeline

The analysis follows a 5-step sequential pipeline:

### Step 1: Grading Consistency Analysis
**Script:** `complete_grader_consistency_analysis.py`
- Analyzes consistency between GPT-4.1 and Claude graders
- Compares scoring patterns across different models
- Outputs: Consistency metrics and reports

### Step 2: Large Differences Identification
**Script:** `targeted_large_differences.py`
- Identifies cases with 20+ percentage point grading differences
- Analyzes Expert1 and Expert2 rubrics separately
- Outputs: JSON data files and markdown reports

### Step 3: Polishing Prompts Generation
**Script:** `rubric_polishing_generator.py`
- Creates AI prompts for rubric improvement
- Based on identified grading disagreements
- Outputs: JSON and markdown prompt files

### Step 4: AI Rubric Polishing
**Script:** `rubric_polisher.py`
- Uses OpenAI GPT-4o to polish rubrics
- Enforces binary scoring to reduce ambiguity
- Outputs: Polished rubrics and detailed analysis

### Step 5: Grading Improvements Comparison
**Script:** `polished_rubrics_grading_comparison.py`
- Re-grades with polished rubrics
- Measures improvement in grader consistency
- Outputs: Comparison metrics and reports

## Main Orchestration Script

**`analysis_workflow.py`** - Runs the complete pipeline or individual steps

### Basic Usage

```bash
# Run complete workflow for a fold
python scripts/analysis_workflow.py \
  --fold 3 \
  --gpt_api_key "your-openai-key" \
  --claude_api_key "your-claude-key"

# Run specific steps only
python scripts/analysis_workflow.py \
  --fold 3 \
  --gpt_api_key "your-key" \
  --claude_api_key "your-key" \
  --steps consistency differences prompts

# With custom paths
python scripts/analysis_workflow.py \
  --fold 3 \
  --gpt_api_key "your-key" \
  --claude_api_key "your-key" \
  --output_dir ./outputs/fold3 \
  --scores_base_path ./data \
  --scripts_base_path ./scripts
```

### Command-Line Arguments

- `--fold`: Fold number to process (required)
- `--gpt_api_key`: OpenAI API key (or set `OPENAI_API_KEY` env var)
- `--claude_api_key`: Anthropic API key (or set `ANTHROPIC_API_KEY` env var)
- `--output_dir`: Base output directory (default: `./fold{N}_analysis_results`)
- `--expert1_rubric`: Path to Expert1 rubric file
- `--expert2_rubric`: Path to Expert2 rubric file
- `--scores_base_path`: Base path for grading score files
- `--scripts_base_path`: Base path for Python scripts
- `--steps`: Which steps to run (choices: `consistency`, `differences`, `prompts`, `polish`, `compare`, `all`)

## Prerequisites

Before running the workflow, ensure:

1. **Grading is complete**: Both `claude_grade_fold{N}.sh` and `gpt_grade_fold{N}.sh` must be completed
2. **Input files exist**:
   - `fold{N}_gpt_scores_expert1.json`
   - `fold{N}_gpt_scores_expert2.json`
   - `fold{N}_claude_scores_expert1.json`
   - `fold{N}_claude_scores_expert2.json`
   - `Fold{N}-Jesse.json` (Expert1 rubric)
   - `Fold{N}-Simone.json` (Expert2 rubric)
   - Response files (OpenAI, Grok, open-source models)

3. **API keys configured**: Either via command-line args or environment variables

## Output Structure

Each workflow run creates a structured output directory:

```
fold{N}_analysis_results/
├── consistency_analysis/
│   ├── expert1_fold{N}_*.json
│   └── expert2_fold{N}_*.json
├── large_differences/
│   ├── fold{N}_large_differences_expert1_data.json
│   ├── fold{N}_large_differences_expert1_report.md
│   ├── fold{N}_large_differences_expert2_data.json
│   └── fold{N}_large_differences_expert2_report.md
├── rubric_polishing/
│   ├── rubric_polishing_prompts_expert1.json
│   ├── rubric_polishing_prompts_expert1.md
│   ├── rubric_polishing_prompts_expert2.json
│   ├── rubric_polishing_prompts_expert2.md
│   ├── polished_rubrics_expert1.json
│   ├── polished_rubrics_expert2.json
│   ├── detailed_rubric_polished_expert1.json
│   ├── detailed_rubric_polished_expert2.json
│   ├── report_rubric_polished_expert1.md
│   └── report_rubric_polished_expert2.md
├── polished_grading_comparison/
│   └── [comparison results]
└── fold{N}_workflow_summary.md
```

## Supporting Scripts

### Grading Scripts (Run before workflow)
- `get_grades_gpt_targeted.py` - Generate GPT-4.1 grading scores
- `get_grades_claude_targeted.py` - Generate Claude grading scores
- `claude_grader.py` - Claude grading utilities

These should be run first to generate the input grading files needed by the workflow.

## Script Dependencies

```
analysis_workflow.py
├── complete_grader_consistency_analysis.py
├── targeted_large_differences.py
├── rubric_polishing_generator.py
├── rubric_polisher.py
└── polished_rubrics_grading_comparison.py
```

Each step depends on outputs from previous steps, so they must run sequentially.

## Environment Setup

```bash
# Set API keys as environment variables (optional)
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-claude-key"

# Install required Python packages
pip install openai anthropic pandas numpy
```

## Notes

- The workflow enforces binary scoring in polished rubrics to reduce grading ambiguity
- Each step can be run independently if you have the required input files
- Progress is tracked and logged throughout the workflow
- All intermediate outputs are preserved for analysis

## Migration from Old Codebase

This clean codebase contains only the essential scripts. The following were excluded:
- Duplicate scripts in fold-specific directories
- Old versions in `fold*/old/` directories
- Unused utility scripts (`merge_ai_polished_inital_rubrics.py`)
- Shell wrapper scripts (can be recreated as needed)

---

**Created:** 2026-02-22
**Source:** Analysis4_permute_grader_model (cleaned and reorganized)
