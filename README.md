# PanCanBench

PanCanBench is a benchmark of 282 de-identified authentic pancreatic cancer patient questions paired with 3,639 expert-designed rubric items for evaluating large language models (LLMs).

## Data Availability

The question–rubric dataset is available on [Hugging Face](https://huggingface.co/datasets/YiminZ07/PanCanBench).

The [versioned reproduction bundle](https://github.com/YiminZhao97/PanCanBench/releases/tag/reproducibility-v1) contains the saved responses, grades, rubrics, human ratings, and analysis inputs listed in `Data/input_manifest.json`. Download, install, and verify it from the repository root:

```bash
python Data/prepare_data.py --download \
  https://github.com/YiminZhao97/PanCanBench/releases/download/reproducibility-v1/pancanbench-reproduction-data-v1.tar.gz
python Data/prepare_data.py --check
```

## Citation

The manuscript is available on [arXiv](https://arxiv.org/abs/2603.01343).

## Code organization

| Folder | Purpose |
| --- | --- |
| `Rubrics/` | Human-rubric development and synthetic-rubric generation |
| `Generation/web_search/` | Generate GPT, Claude, and Gemini responses with web search |
| `Evaluation/` | Rubric scoring, factuality detection, direct pairwise judging, and citation verification |
| `Analysis/` | Reproduce manuscript and appendix results; shared requirements |
| `Data/` | Saved input data and its checksum manifest |
| `Outputs/` | Generated responses, evaluations, figures, and tables |

## Reproducing the results

### Environment

Use Python 3.12 and install the shared dependencies from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r Analysis/requirements.txt
```

### Calculate a model's mean rubric score

After [setting up the environment](#environment), use the commands below to grade a model's saved responses and calculate its **mean rubric score**. This example evaluates GPT-5's 282 saved responses using Claude Opus 5 as the judge. The grading step submits paid Anthropic API requests and waits for the batch to finish.

Run from the repository root:

```bash
GRADING_CODE_DIR="Evaluation/rubric_scoring"
RESPONSES="Data/Response/gpt5_response.jsonl"

# 1. Set the judge's API key.
export ANTHROPIC_API_KEY='<your_anthropic_api_key>'

# 2. Grade all responses against the final human-curated rubrics.
python "$GRADING_CODE_DIR/grade_anthropic_batch.py" run \
  --input "$RESPONSES" \
  --rubrics Data/Rubrics/rubrics_all_questions_final_version.json \
  --judge-model claude-opus-5 \
  --allow-full-run \
  --output Outputs/rubric_scoring/gpt5_response_graded.jsonl

# 3. Calculate the model's mean score after grading completes successfully.
python "$GRADING_CODE_DIR/summarize_graded_responses.py" \
  --grades Outputs/rubric_scoring/gpt5_response_graded.jsonl \
  --responses "$RESPONSES" \
  --output Outputs/rubric_scoring/gpt5_response_graded.summary.json
```

The model's score is **`mean_percentage`** under `models` in the summary JSON. It averages the per-question percentage scores, including applicable negative rubric penalties. Explicit `NO RESPONSE GENERATED` entries are excluded from the mean; genuine zero scores are included. `pooled_points_percentage` is a separate summary statistic, not the reported mean rubric score.

### Generate and evaluate new responses

These commands make paid API calls. Use a new output filename for each run; the saved paper inputs remain under `Data/`.

```bash
# Generate GPT-5 responses for the paper's 40 web-search questions.
export OPENAI_API_KEY='<your_openai_api_key>'
python Generation/web_search/generate_response_gpt_family.py \
  --output Outputs/web_search/gpt5.json

# Extract atomic claims from saved model responses, then judge their factuality.
python Evaluation/factuality/extract_atomic_claims.py \
  --input Data/Response/gpt5_response.jsonl --models gpt-5 \
  --output Outputs/factuality/gpt5_claims.json
python Evaluation/factuality/judge_factuality_argparse_GPT.py \
  --input Outputs/factuality/gpt5_claims.json --output Outputs/factuality/gpt5_gpt_judgments.json
export GEMINI_API_KEY='<your_gemini_api_key>'
python Evaluation/factuality/judge_factuality_argparse_gemini.py \
  --input Outputs/factuality/gpt5_claims.json --output Outputs/factuality/gpt5_gemini_judgments.json

# Ask the direct judge to compare two models across the benchmark.
python Evaluation/direct_pairwise/compare.py \
  --input-a Data/Response/gpt5_response.jsonl --model-a gpt-5 \
  --input-b Data/Response/grok_response.jsonl --model-b grok-4-latest \
  --judge-model gpt-5 --allow-full-run --output Outputs/direct_pairwise/gpt5_vs_grok4.json
```

Claude and Gemini response generation use `Generation/web_search/claude_family_search.py` and `Generation/web_search/gemini_family_web_search_metadata.py`, with `ANTHROPIC_API_KEY` and `GEMINI_API_KEY`, respectively. Citation verification uses `Evaluation/citation_verification/verify_reference_citations.py`; run its `--help` for the prepare, fetch, submit, collect, and report stages. New evaluations can differ from the saved paper observations.

To regenerate token-count tables from the original response text without model API calls:

```bash
python Analysis/Appendix/count_tokens.py --responses-dir Data/Response
python Analysis/Appendix/supp_figure_6_token_distribution.py --responses-dir Data/Response
```

The tables below show how to reproduce the manuscript and appendix results, with the corresponding inputs, code, and outputs.

### Manuscript

| Paper item | Inputs | Code | Outputs |
| --- | --- | --- | --- |
| Figure 4a: mean rubric score by model | Final rubrics, eight final grade JSONL files, and their summary JSON files (listed in `Analysis/figure4/model_config.json`) | `Analysis/figure4/plot_figure4a.py` | PDF, PNG, question-level source table, model summary, provenance |
| Figure 4b: factual errors | Two saved factual-error count tables in `Analysis/figure4/inputs/figure4b/`, shared model configuration, and Figure 4a's saved order | `Analysis/figure4/plot_figure4b.py` | PDF, PNG, overall rates, category matrix, provenance |
| Figure 4c: scores with and without web search | Final rubrics and six final grade files; the original 40 citation-required questions | `Analysis/figure4/plot_figure4c.py` | PDF, PNG, and JSON containing paired scores, statistics, and provenance |
| Figure 4d: web-search metrics | Frozen response/grade inputs plus the completed 40-question citation audit; historical modes also retained | `Analysis/figure4/reproduce_figure4d.py` | Markdown table and JSON containing counts, denominators, coverage, and provenance |
| Figure 5a: human versus synthetic rubric scores | Included, verified question-level scores for the same 6,203 responses; model configuration and source provenance | `Analysis/figure5/paired_barplot.py` | PDF, PNG, mean/SE table, provenance |
| Figure 5b: human versus synthetic rubric rankings | Same paired inputs as Figure 5a; ranks calculated from unrounded means | `Analysis/figure5/slopegraph_rank_change.py` | PDF, PNG, rank-change table, provenance |

Figure 5 uses the completed human- and synthetic-rubric grades from Claude Opus 5 on the same 6,203 eligible responses across 22 models. Verified question-level score inputs are included under `Analysis/figure5/inputs/`, so these panels can be reproduced without API calls or a new data-bundle download. The saved synthetic grades retain 35 interpretation/direction flags pending adjudication; no scores were manually changed for these plots. See [Figure 5 reproduction notes](Analysis/figure5/README.md) for scoring, exclusions, source hashes, and rebuilding inputs from raw grades.

### Appendix

| Paper item | Inputs | Code | Outputs |
| --- | --- | --- | --- |
| Supplementary Figure 1: grading consistency before/after rubric clarification | Saved original and polished Phase 2 grades | `Analysis/Appendix/supp_figure_1_grading_consistency.py` | PNG/PDF, 30 bar values, 1,692 paired comparisons, provenance |
| Supplementary Figure 2: large versus small models | Ten saved Phase 2 grade files | `Analysis/Appendix/supp_figure_2_model_size_comparison.py` | PNG/PDF, 10 bar values, 1,128 paired comparisons, provenance |
| Appendix Table S2: candidate judge agreement | Four saved judge runs and five final human-rating CSVs | `Analysis/Appendix/table_s2_judge_agreement.py` | Table CSV/Markdown, 20 judge–human comparisons, provenance |
| Appendix Table S5: direct vs rubric pairwise agreement | Six retained GPT-5 direct-judgment files and four final Claude Opus 5 grade files | `Analysis/Appendix/table_s5_direct_judgment_agreement.py` | Table CSV/Markdown, 1,692 question-pair decisions, provenance |
| Supplementary Figure 4: wrong-claim counts | Two frozen count tables under `Data/appendix/`; optional raw factuality files for validation | `Analysis/Appendix/supp_figures_4_5_wrong_claims.py` | PNG/PDF, wrong-claim counts CSV, shared provenance |
| Supplementary Figure 5: wrong-claim percentages | Same saved count tables as Supplementary Figure 4 | `Analysis/Appendix/supp_figures_4_5_wrong_claims.py` | PNG/PDF, wrong-claim percentages CSV, shared provenance |
| Supplementary Figure 6: response token distributions | Frozen per-response token counts under `Data/appendix/`; optional original response files | `Analysis/Appendix/supp_figure_6_token_distribution.py` | PNG/PDF plot, boxplot statistics, provenance |

### Run the analyses

With the saved inputs available, run the following commands from the repository root; no API key is needed. Figure 4 plots require Helvetica, or a replacement font set in `Analysis/figure4/model_config.json`.

```bash
python Analysis/verify_release.py --run
```

Manuscript outputs are saved to `Outputs/figure4/` and `Outputs/figure5/`; appendix outputs are saved to `Outputs/appendix/`. The runner verifies the saved inputs and checks all 13 listed results, including an independent recomputation of Figure 5 means, standard errors, and ranks and a cross-check of its human scores against Figure 4a. To run only the appendix, use `python Analysis/Appendix/run_all.py`.

To regenerate only Figure 5:

```bash
python Analysis/figure5/paired_barplot.py --output-dir Outputs/figure5
python Analysis/figure5/slopegraph_rank_change.py --output-dir Outputs/figure5
python Analysis/figure5/verify_results.py --output-dir Outputs/figure5
```

The materials-folder and repository `Analysis/figure5/` copies also contain the verified plots. Running either plotting script without `--output-dir` saves its output alongside the script.
