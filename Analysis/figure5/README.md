# Figure 5: human versus synthetic rubrics

These scripts regenerate the two panels from the completed Claude Opus 5 grading of the same model responses. Figure 5a shows mean scores with one standard error; Figure 5b shows the change in rank between rubric sets. This is a comparison of rubric sets, not a comparison between judge models.

## Run

Python, NumPy, and Matplotlib are required (included in PanCanBench's `Analysis/requirements.txt`). No API key or network connection is needed.

From the repository root:

```bash
python Analysis/figure5/paired_barplot.py --output-dir Outputs/figure5
python Analysis/figure5/slopegraph_rank_change.py --output-dir Outputs/figure5
python Analysis/figure5/verify_results.py --output-dir Outputs/figure5 \
  --figure4-summary Outputs/figure4/figure4a_model_summary.csv
```

Alternatively, run the two plotting scripts without arguments to save outputs alongside the scripts. This is how the materials-folder copy was generated. Omit `--figure4-summary` when Figure 4 has not been generated.

Outputs are `figure5a.pdf`, `figure5a.png`, `figure5b.pdf`, `figure5b.png`, `figure5_model_summary.csv`, `rank_change_summary.csv`, and per-panel provenance JSON files. Verification produces `verification_report.json`. PDFs contain vector graphics and embedded fonts; PNGs are 300 dpi. Helvetica is used if available, otherwise DejaVu Sans.

## Inputs and scoring

`inputs/paired_question_scores.csv` contains all 6,203 eligible model-question pairs, including the signed totals, positive-weight denominators, percentages, criterion counts, and response-input hashes for each rubric set. `inputs/input_provenance.json` records SHA-256 hashes of the 16 original grade files, their 16 summary files, and both scoring-rubric files, plus the reference means and validation counts. The compact score inputs are included with the Figure 5 code, so plotting does not require a new data-bundle download.

- All 22 selected models are included. OLMo2 models are excluded. OpenEvidence is outside this cohort.
- Each model contributes 282 questions except Claude Opus 4.1, which contributes 281. Its documented missing response Q267 is excluded from both rubric sets. All 6,203 paired response-input hashes match.
- Each question is scored as `100 * signed total / sum of positive rubric weights`. The plotted mean weights questions equally. Scores are neither clipped nor subject to an additional factual-error deduction.
- A negative criterion contributes its negative weight when the saved decision is true, and zero when false. The human and synthetic judging prompts differ in how criterion wording is interpreted. These scripts use the saved decisions without reinterpretation.
- Genuine zero scores and negative scores are retained. The historical blanket deletion of Claude zero scores has been removed.
- Human scores include the existing Q151 item 12 negative-weight correction and match the current Figure 4a means and standard errors.
- Standard error is sample SD (`ddof=1`) divided by `sqrt(n)`. It is not a confidence interval.
- Figure 5a retains the original company colors, solid human bars, synthetic hatching, and descending human-mean order within each provider.
- Figure 5b ranks the unrounded means in descending order using competition ranking (`method=min`). `rank_change = human_rank - synthetic_rank`; positive means a higher rank under synthetic rubrics.

The saved synthetic grades retain **35 interpretation/direction flags pending adjudication**. These plots use the completed grading exports without manual synthetic-score changes. Mechanical verification does not adjudicate the clinical correctness of judge decisions. The comparison uses the same judge and response inputs, but both rubric content and judging instructions differ.

## Rebuild the compact inputs from saved grades

`prepare_inputs.py` validates every included criterion's number, description, weights, Boolean decision, awarded points, total, denominator, and percentage against the canonical rubrics. It also checks judge/prompt metadata, complete coverage, duplicate keys, each model's reference mean, the Q151 correction metadata, and matching response hashes. The human source contains 6,204 selected-model records; its missing-response Q267 record is validated and then excluded. The synthetic source already omits that record.

```bash
python Analysis/figure5/prepare_inputs.py \
  --human-grades-dir /path/to/materials/Data/grades \
  --synthetic-grades-dir /path/to/materials/Data/grades/synthetic_rubrics \
  --human-rubrics /path/to/materials/Data/grades/scoring_rubrics.json \
  --synthetic-rubrics /path/to/materials/Analysis/synthetic_rubric_regrading/synthetic_rubrics_original.jsonl
```

The completed raw synthetic grading exports are not part of the earlier `reproducibility-v1` data bundle. Their verified question-level scores are included here. Rebuilding requires the four input locations above; ordinary plotting uses the included compact inputs.

The plotting code was adapted from the historical `paired_barplot.py` and `slopegraph_rank_change.py` in the manuscript's Figure 5 folder. Unmodified historical copies and the old name mapping are preserved under `original_scripts/` in the materials-folder copy only; they still point to old inputs and should not be used for the new results.
