#!/usr/bin/env python3
"""Reproduce Figure 4d from saved observations, without API calls.

Consolidates analyze_rubrics_{gpt,gemini,claude}.py and
calculate_url_helpfulness.py. The default reproduces the historical analysis;
--grade-version final uses existing Opus 5 grades for resource appropriateness.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform

from saved_inputs import resolve_saved_file
from plot_figure4c import MODELS, QUESTION_NUMBERS, load_model_rows
from plot_figure4a import load_grading_helpers, validate_score


LEGACY_FILES = {
    "claude-sonnet-4-5-20250929": (
        "claude", "claude-sonnet-4-5_web_search_responses.json",
        "claude-sonnet-4-5_websearch_responses_with_citations_graded.jsonl"),
    "gemini-2.5-pro": (
        "gemini", "gemini_family_response_web_search.json",
        "gemini25pro_websearch_responses_with_citations_graded.jsonl"),
    "gpt-5": (
        "gpt5", "gpt5_web_search_responses.json",
        "gpt5_websearch_responses_graded.jsonl"),
}
# Reviewed against the old criterion text and final rubrics, not positional joins.
RENUMBERED = {(227, 12): 15, (229, 11): 13, (276, 8): 7, (197, 10): 9}
METRICS = ["Web Search Triggering Rate", "Resources Appropriateness",
           "Supportive Link Percentage"]


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def index_rows(rows, key):
    result = {row[key]: row for row in rows}
    check(len(result) == len(rows), f"Duplicate {key}")
    return result


def metadata(row, model):
    """Retain the original provider-specific search proxy and URL counting."""
    if model["id"] == "gpt-5":
        citations = row["annotations"]
        urls = [c["url"] for c in citations if c["type"] == "url_citation"]
    elif model["id"] == "gemini-2.5-pro":
        citations = row["responses"][model["web_model"]]["grounding_metadata"].get(
            "grounding_chunks", [])
        urls = [c.get("uri", "") for c in citations]
    else:
        citations = row["responses"][model["web_model"]]["citations"]
        # The original Claude evaluator alone deduplicated URLs within questions.
        urls = list(dict.fromkeys(c["url"] for c in citations if c.get("url")))
    check(isinstance(citations, list), "Citation metadata must be a list")
    return len(citations), urls


def prepare_inputs(source_dir, rubric_path):
    """Make a compact, lossless extract of the fields used by this calculation.

    Full response prose and unused grade items are omitted. Every original URL
    rating is retained, including its URL, title, confidence, and explanation.
    Original files and their SHA-256 hashes remain in the source manifest.
    """
    source_manifest = []

    def source(relative):
        path = source_dir / relative
        source_manifest.append({"path": relative, "sha256": sha256(path)})
        return path

    with source("rubrics_asking_for_reference.csv").open(encoding="utf-8-sig") as f:
        selected = [(int(r["Question_ID"]), int(r["Item_ID"]))
                    for r in csv.DictReader(f)]
    check(len(selected) == len(set(selected)) == 44, "Expected 44 unique reference items")
    check({q for q, _ in selected} == set(QUESTION_NUMBERS), "Reference question subset changed")
    rubrics = index_rows(read_json(rubric_path)["questions"], "question_number")
    question_ids = {f"Q{q}" for q in QUESTION_NUMBERS}
    result = []
    for model in MODELS:
        prefix, response_file, grade_file = LEGACY_FILES[model["id"]]
        responses = index_rows(read_json(source("response/" + response_file)), "question_id")
        urls = index_rows(read_json(source(
            f"check_citation_relevance/{prefix}_url_helpfulness_evaluation.json")), "question_id")
        grades = index_rows(read_json(source("res/new grade/" + grade_file)), "question_number")
        check(set(responses) == set(urls) == question_ids, "Incomplete response/URL cohort")
        check(set(grades) == set(QUESTION_NUMBERS), "Incomplete historical grades")
        questions = []
        for q in QUESTION_NUMBERS:
            qid = f"Q{q}"
            count, expected_urls = metadata(responses[qid], model)
            evaluation = urls[qid]
            check(evaluation["question"] == responses[qid]["question"], f"Question text mismatch: {qid}")
            ratings = evaluation["url_evaluations"]
            check([u["url"] for u in ratings] == expected_urls, f"URL identities/order differ: {qid}")
            check(evaluation["total_urls"] == len(ratings), f"URL denominator mismatch: {qid}")
            check(all(type(u.get("is_helpful")) is bool for u in ratings), f"Missing URL rating: {qid}")
            check(grades[q]["grader_model"] == "gpt-5", "Historical rubric judge changed")
            old = index_rows(grades[q]["criterion_scores"], "criterion_number")
            final = index_rows(rubrics[q]["rubric_items"], "item_number")
            items = []
            for question, item in selected:
                if question != q:
                    continue
                mapped = RENUMBERED.get((q, item), item)
                check(mapped in final, f"Missing final criterion: {qid}/{mapped}")
                items.append({"legacy_item_number": item,
                              "legacy_description": old[item]["description"],
                              "legacy_score_given": old[item]["score_given"],
                              "legacy_max_points": old[item]["max_points"],
                              "final_item_number": mapped,
                              "final_description": final[mapped]["description"]})
            questions.append({"question_id": qid, "citation_record_count": count,
                              "url_evaluations": ratings, "reference_items": items})
        result.append({"response_model": model["id"], "questions": questions})
    return {"schema_version": 1, "source_directory_name": source_dir.name,
            "source_files": source_manifest, "final_rubrics_sha256": sha256(rubric_path),
            "models": result}


def rate(numerator, denominator):
    check(type(numerator) is int and type(denominator) is int
          and 0 <= numerator <= denominator and denominator > 0, "Invalid rate counts")
    return {"numerator": numerator, "denominator": denominator,
            "percentage": 100 * numerator / denominator}


def analyze(snapshot, grade_version, data_dir, config_path, grading_code_dir):
    check(snapshot["schema_version"] == 1, "Unsupported source schema")
    inputs = []
    helpers = rubrics = config = None
    if grade_version == "final":
        helpers = load_grading_helpers(grading_code_dir)
        config = read_json(config_path)
        rubric_path = data_dir / config["rubrics_file"]
        check(sha256(rubric_path) == snapshot["final_rubrics_sha256"] == config["rubrics_sha256"],
              "Final rubrics changed; review the reference-item mapping")
        rubrics = helpers.load_rubrics(rubric_path)
        inputs.extend({"path": str(p), "sha256": sha256(p)} for p in
                      [rubric_path, config_path, grading_code_dir / "grade_anthropic_batch.py"])
    saved = index_rows(snapshot["models"], "response_model")
    check(set(saved) == {m["id"] for m in MODELS}, "Model cohort changed")
    results = []
    for model in MODELS:
        questions = index_rows(saved[model["id"]]["questions"], "question_id")
        check(set(questions) == {f"Q{q}" for q in QUESTION_NUMBERS}, "Question cohort changed")
        final = None
        if grade_version == "final":
            path = data_dir / "grades/web_search" / model["web_file"]
            final = load_model_rows(path, model["web_model"])
            check(set(final) == set(questions), "Incomplete final grade cohort")
            inputs.append({"path": str(path), "sha256": sha256(path)})
        triggered = helpful = total_urls = met = eligible = 0
        detail = []
        for qid, row in questions.items():
            count = row["citation_record_count"]
            check(type(count) is int and count >= 0, f"Invalid citation count: {qid}")
            searched = count > 0
            triggered += int(searched)
            links = row["url_evaluations"]
            check(all(type(u.get("is_helpful")) is bool for u in links), f"Missing URL rating: {qid}")
            helpful += sum(u["is_helpful"] for u in links)
            total_urls += len(links)
            if final is not None:
                grade = final[qid]
                check(grade["input_sha256"] == model["web_input_sha256"], "Response-input hash changed")
                validate_score(grade, rubrics, helpers, config)
                criteria = index_rows(grade["criterion_scores"], "criterion_number")
            seen = set()
            for item in row["reference_items"]:
                old_number = item["legacy_item_number"]
                check(old_number not in seen, f"Duplicate reference criterion: {qid}")
                seen.add(old_number)
                number = item["final_item_number"] if final is not None else old_number
                score = criteria[number]["score_given"] if final is not None else item["legacy_score_given"]
                maximum = criteria[number]["max_points"] if final is not None else item["legacy_max_points"]
                check(isinstance(score, (int, float)) and not isinstance(score, bool)
                      and math.isfinite(score) and maximum > 0 and score in (0, maximum),
                      f"Reference criterion must have a binary positive-point grade: {qid}/{number}")
                if final is not None:
                    check(criteria[number]["description"] == item["final_description"],
                          f"Final mapped description mismatch: {qid}/{number}")
                met += int(searched and score > 0)
                eligible += int(searched)
                detail.append({"question_id": qid, "item_number": number,
                               "legacy_item_number": old_number, "search_triggered": searched,
                               "score_given": score, "max_points": maximum,
                               "criterion_met": score > 0})
        check(len(detail) == 44, "Expected 44 reference-related criteria per model")
        results.append({"response_model": model["id"], "display_name": model["label"],
                        "metrics": dict(zip(METRICS, [rate(triggered, 40), rate(met, eligible),
                                                      rate(helpful, total_urls)])),
                        "reference_item_details": detail})
    return results, inputs


def markdown(results, grade_version):
    lines = ["# Figure 4d. Web-search metrics", "",
             f"Resource-appropriateness grades: **{'final Claude Opus 5' if grade_version == 'final' else 'historical GPT-5'}**.", "",
             "| Metric | " + " | ".join(r["display_name"] for r in results) + " |",
             "| --- | ---: | ---: | ---: |"]
    for metric in METRICS:
        cells = [r["metrics"][metric] for r in results]
        lines.append("| " + metric + " | " + " | ".join(
            f"{r['percentage']:.1f}% ({r['numerator']}/{r['denominator']})" for r in cells) + " |")
    lines += ["", "Search triggering uses nonempty provider citation/grounding metadata as in the original analysis.",
              "Resources appropriateness is the proportion of selected reference-related rubric items awarded positive points among search-triggered responses, not the percentage of all URLs that are appropriate.",
              "Supportive links use the original saved GPT-5 helpfulness decisions, pooled across URLs. Claude URLs were deduplicated within each question; GPT-5 and Gemini citation occurrences were retained. No URL was re-evaluated.", "",
              "Historical arithmetic check: Gemini's saved counts give 23/39 = 59.0%, not the 56.0% in the old figure. GPT-5's counts give 30/36 = 83.3%, previously displayed as 83.0%. Claude's 21/40 = 52.5% agrees.",
              "The assembled figure and manuscript are not modified by this script.", ""]
    return "\n".join(lines)


def audited_figure4d(args):
    """Reproduce the approved 40-question analysis without new API calls.

    Primary reference grades stay text-only. Link support is a separate,
    source-verified measurement. Historical modes above are unchanged.
    """
    run = args.citation_run
    cohort = read_json(run / "cohort.json")
    validation = read_json(run / "validation.json")
    merged = read_json(run / "merged_results.json")
    check(merged["complete"] and not validation["failures"] and validation["valid_cases"] == 120,
          "All 120 cases must be accounted for before publishing the table")
    provenance = []

    def record(path):
        provenance.append({"path": str(path), "sha256": sha256(path)})
        return read_json(path)

    for entry in cohort["frozen_files"]:
        resolved = resolve_saved_file(entry["path"], entry["sha256"], args.data_dir.parent)
        provenance.append({"path": str(resolved), "sha256": entry["sha256"]})
    for name in ("cohort.json", "validation.json", "merged_results.json", "pilot_reuse.json"):
        record(run / name)
    cases = index_rows(cohort["cases"], "case_id")
    ratings = index_rows(merged["results"], "case_id")
    check(set(cases) == set(ratings) and len(cases) == 120, "Merged case coverage mismatch")
    reuse = read_json(run / "pilot_reuse.json")
    pilot_path = resolve_saved_file(Path(reuse["pilot_run"]) / "pilot_results.json", reuse["pilot_results_sha256"], args.data_dir.parent)
    check(sha256(pilot_path) == reuse["pilot_results_sha256"], "Frozen pilot results changed")
    for prior in record(pilot_path)["results"]:
        current = dict(ratings[prior["case_id"]])
        check(current.pop("reused_from_pilot", False) and current == prior, "Reused pilot result changed")

    def preserves(original, current):
        if isinstance(original, dict):
            return all(k in current and preserves(v, current[k]) for k, v in original.items())
        if isinstance(original, list):
            return len(original) == len(current) and all(preserves(a, b) for a, b in zip(original, current))
        return original == current

    retained_checked = 0
    for path in sorted((run / "jobs").glob("*/source_repair_manifest.json")):
        for case in record(path)["cases"]:
            current = index_rows(ratings[case["case_id"]]["source_verification"]["sources"], "source_id")
            for source in case["retained_sources"]:
                check(preserves(source, current[source["source_id"]]), "Validated source was changed during repair")
                retained_checked += 1
    snapshot = record(args.input)
    old_metadata = index_rows(snapshot["models"], "response_model")
    regenerated = args.data_dir / "Response/web_search/gemini_2.5_pro_interactions_2026-09-15"
    models = [
        ("gpt5", "GPT-5", "gpt-5", "gpt5_websearch_response_graded.jsonl"),
        ("claude-sonnet-4-5-20250929", "Claude Sonnet 4.5", "claude-sonnet-4-5-20250929", "claude-sonnet-4-5_websearch_response_graded.jsonl"),
        ("gemini-2.5-pro", "Gemini 2.5 Pro", "gemini-2.5-pro", "gemini-2.5-pro_websearch_regenerated_2026-09-15_response_graded.jsonl"),
    ]
    output = []
    for model, label, old_id, filename in models:
        path = args.data_dir / "grades/web_search" / filename
        grades = index_rows([json.loads(x) for x in path.read_text().splitlines() if x.strip()], "question_id")
        provenance.append({"path": str(path), "sha256": sha256(path)})
        selected = [c for c in cases.values() if c["response_model"] == model]
        check(len(selected) == len(grades) == 40 and {c["question_id"] for c in selected} == set(grades), "Question coverage mismatch")
        meta = index_rows(old_metadata[old_id]["questions"], "question_id")
        n_triggered = met = conditional_met = conditional_n = 0
        links = []
        detail = []
        for c in selected:
            qid = c["question_id"]
            if model == "gemini-2.5-pro":
                raw = record(regenerated / "raw" / (qid + ".json"))
                searched = any(step.get("type") == "google_search_call" for step in raw["steps"])
            else:
                searched = meta[qid]["citation_record_count"] > 0
            n_triggered += int(searched)
            grade = grades[qid]
            check(grade["judge_model"] == "claude-opus-5" and grade["response_model"] == model, "Unexpected primary rubric judge or response model")
            items = index_rows(grade["criterion_scores"], "criterion_number")
            check(len(c["reference_criteria"]) == len(meta[qid]["reference_items"]), "Reference item selection mismatch")
            item_details = []
            for criterion in c["reference_criteria"]:
                item = items[criterion["item_number"]]
                check(item["description"] == criterion["description"], "Criterion text changed")
                maximum, score = item["max_points"], item["score_given"]
                check(maximum > 0 and score in (0, maximum), "Reference grade must be binary positive-weight")
                passed = score > 0
                met += int(passed)
                conditional_met += int(searched and passed)
                conditional_n += int(searched)
                item_details.append({"item_number": criterion["item_number"], "met": passed})
            result = ratings[c["case_id"]]
            check(result["response_sha256"] == c["response_sha256"], "Audited response changed")
            case_links = [s for s in result["source_verification"]["sources"] if s["inline_link_present"]]
            expected = {s["canonical_url"] for s in c["sources"] if any(o["kind"] in ("markdown_link", "bare_url") for o in s["occurrences"])}
            check(len(case_links) == len(expected) and {s["canonical_url"] for s in case_links} == expected, "Incomplete link coverage")
            links.extend(case_links)
            counts = {v: sum(s["question_relevant_support"] == v for s in case_links)
                      for v in ("supported", "not_supported", "unresolved")}
            detail.append({"question_id": qid, "search_triggered": searched, "reference_criteria": item_details,
                           "link_counts": counts, "pilot_reused": result.get("reused_from_pilot", False)})
        check(sum(len(d["reference_criteria"]) for d in detail) == 44, "Expected 44 reference items per model")
        counts = {v: sum(s["question_relevant_support"] == v for s in links)
                  for v in ("supported", "not_supported", "unresolved")}
        supported, failed, unresolved = (counts[v] for v in ("supported", "not_supported", "unresolved"))
        metrics = {
            "Web-search triggering rate": rate(n_triggered, 40),
            "Reference-criterion success rate": rate(met, 44),
            "Question-relevant supportive links (evaluable links)": rate(supported, supported + failed),
            "Evaluable-link coverage": rate(supported + failed, len(links)),
            "Confirmed supportive links (all links)": rate(supported, len(links)),
        }
        output.append({"response_model": model, "display_name": label, "metrics": metrics,
                       "link_counts": counts, "all_links": len(links),
                       "responses_with_links": sum(sum(d["link_counts"].values()) > 0 for d in detail),
                       "secondary_reference_success_conditional_on_search": rate(conditional_met, conditional_n),
                       "details": detail})
    usage = record(run / "usage_summary.json")
    preview = record(run / "request_preview.json")
    t = usage["totals"]
    check(t.get("cache_creation_input_tokens", 0) == t.get("ephemeral_1h_input_tokens", 0) + t.get("ephemeral_5m_input_tokens", 0), "Cache duration accounting incomplete")
    rates = {"input_tokens": 2.5, "output_tokens": 12.5, "cache_read_input_tokens": .25,
             "ephemeral_1h_input_tokens": 5, "ephemeral_5m_input_tokens": 3.125}
    cost = sum(t.get(k, 0) * v / 1_000_000 for k, v in rates.items())
    first_batch = min(datetime.fromisoformat(r["created_at"]) for r in usage["runs"])
    last_batch = max(datetime.fromisoformat(r["ended_at"]) for r in usage["runs"])
    batch_minutes = (last_batch - first_batch).total_seconds() / 60
    reuse = read_json(run / "pilot_reuse.json")
    pilot_cost = record(pilot_path.with_name("cost_summary.json"))
    result = {"models": output, "method": "Automated source verification without independent human validation",
              "primary_grades_changed": False, "reused_pilot_cases": 6,
              "final_integrity_checks": {"pilot_results_identical": 6, "retained_source_assessments_unchanged": retained_checked,
                                         "all_frozen_primary_files_unchanged": True},
              "new_paid_requests_including_retries": usage["paid_new_requests"],
              "submitted_requests_including_preinference_rejections": usage["submitted_new_requests"],
              "requests_without_reported_token_usage": usage["requests_without_reported_usage"],
              "new_run_usage": t, "estimated_new_api_cost_usd": cost,
              "batch_window_minutes_including_retry_gaps": batch_minutes,
              "individual_batch_minutes": [(datetime.fromisoformat(r["ended_at"]) - datetime.fromisoformat(r["created_at"])).total_seconds() / 60 for r in usage["runs"]],
              "source_excerpting": {"source_instances": preview["excerpted_sources"],
                                     "responses": sum(c["excerpted_sources"] > 0 for c in preview["cases"])},
              "cost_rates_per_million_tokens": rates, "pilot_cost_separate": pilot_cost,
              "cost_note": "Calculated from reported tokens and published batch rates; not an account-balance debit.",
              "inputs": provenance, "script_sha256": sha256(Path(__file__))}
    lines = ["# Figure 4d: final 40-question web-search analysis", "",
             "120 saved responses: 40 questions × three models. Six completed Q178/Q206 pilot responses were reused. Primary text-only Opus 5 rubric grades were not changed.", "",
             "| Metric | " + " | ".join(x["display_name"] for x in output) + " |", "| --- | ---: | ---: | ---: |"]
    for metric in output[0]["metrics"]:
        cells = [x["metrics"][metric] for x in output]
        lines.append("| " + metric + " | " + " | ".join(f"{c['percentage']:.1f}% ({c['numerator']}/{c['denominator']})" for c in cells) + " |")
    lines.append("| Unresolved links | " + " | ".join(str(x["link_counts"]["unresolved"]) for x in output) + " |")
    lines += ["", "## Definitions and limitations", "",
              "- Search triggering: nonempty provider citation metadata for the preserved GPT-5/Claude responses; actual Google-search-call steps for regenerated Gemini. These provider-specific indicators are not identical. Search execution is distinct from rendering an inline citation.",
              "- Reference-criterion success: existing text-only Opus 5 binary grades, pooled across the same 44 selected reference items in all 40 questions. This is not a URL-quality score and does not condition the primary denominator on whether search occurred.",
              "- Supportive link: the retrieved source supports all assessed substantive claims attached to that citation, and at least one supported claim helps answer the question. The citation need not support the entire response. URLs are deduplicated within each response, then pooled; standalone bibliographic identifiers do not inflate the link denominator.",
              "- Evaluable = supported + decisively not supported. Unresolved links include insufficient retrieved evidence, uncertain source identity, ambiguous attribution, or unresolved claims/relevance. They are not automatically incorrect. Report support together with coverage; the all-link confirmed percentage is a conservative observed fraction, not an estimate treating unresolved citations as false.",
              "- Source retrieval uses available same-publication full text, abstracts, or webpages, with no paywall bypass. Evidence availability varies. Automated assessments have no independent human validation; structural/quotation checks are not human validation. These results do not establish causal effects of web search or a definitive model ranking.",
              f"- {preview['excerpted_sources']} source instances in {sum(c['excerpted_sources'] > 0 for c in preview['cases'])} new responses required explicit excerpts under the unchanged pilot size limit; missing evidence in an excerpt cannot establish non-support.",
              "- Gemini responses were regenerated in September 2026; GPT-5 and Claude responses were preserved. The new citation audit uses a different support definition and denominator from the old helpfulness analysis, so those percentages are not directly comparable.", "",
              "## Secondary legacy-compatible denominator", "",
              "Reference-criterion success restricted to search-triggered responses (not the primary table): " + "; ".join(
                  x["display_name"] + ": " + f"{x['secondary_reference_success_conditional_on_search']['numerator']}/{x['secondary_reference_success_conditional_on_search']['denominator']} ({x['secondary_reference_success_conditional_on_search']['percentage']:.1f}%)" for x in output) + ".", "",
              "## Execution and cost", "",
              f"New batch requests with reported token usage, including retries: {usage['paid_new_requests']}. Total submissions: {usage['submitted_new_requests']}; {usage['requests_without_reported_usage']} requests returned without reported token usage. Calculated new API cost: ${cost:.4f} (provider token usage × published batch/cache rates, not observed balance). Pilot cost is recorded separately in the JSON report. Source downloads have no model-token charge.", "",
              f"Elapsed window from the first batch submission through the last batch completion: {batch_minutes:.1f} minutes, including retry gaps but excluding initial source retrieval.", "",
              "The assembled figure, manuscript, responses, and original rubric grades are unchanged.", ""]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figure4d_citation_audit.md").write_text("\n".join(lines), encoding="utf-8")
    (args.output_dir / "figure4d_citation_audit_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=here / "inputs/figure4d/source_data.json")
    parser.add_argument("--prepare-inputs", type=Path, metavar="LEGACY_SOURCE_DIR",
                        help="Re-extract the compact input from the original folder; no API calls")
    parser.add_argument("--grade-version", choices=["legacy", "final"], default="legacy")
    parser.add_argument("--citation-run", type=Path, help="Completed full-cohort source-verification run; uses current Gemini and unchanged primary reference grades")
    parser.add_argument("--data-dir", type=Path, default=here.parents[1] / "Data")
    parser.add_argument("--grading-code-dir", type=Path, default=here.parents[1] / "Evaluation/rubric_scoring")
    parser.add_argument("--config", type=Path, default=here / "model_config.json")
    parser.add_argument("--output-dir", type=Path, default=here.parents[1] / "Outputs/figure4")
    args = parser.parse_args()
    if args.citation_run:
        audited_figure4d(args)
        return
    if args.prepare_inputs:
        snapshot = prepare_inputs(args.prepare_inputs, args.data_dir / "rubrics_all_questions_final_version.json")
        args.input.parent.mkdir(parents=True, exist_ok=True)
        args.input.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    snapshot = read_json(args.input)
    results, inputs = analyze(snapshot, args.grade_version, args.data_dir, args.config, args.grading_code_dir)
    inputs.insert(0, {"path": str(args.input), "sha256": sha256(args.input)})
    report = markdown(results, args.grade_version)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "figure4d" if args.grade_version == "final" else "figure4d_legacy"
    (args.output_dir / (stem + ".md")).write_text(report, encoding="utf-8")
    output = {"grade_version": args.grade_version, "url_judge": "historical GPT-5 (saved decisions)",
              "models": results, "inputs": inputs, "source_files": snapshot["source_files"],
              "software": {"python": platform.python_version()},
              "code": [{"file": p.name, "sha256": sha256(p)} for p in
                       [Path(__file__), Path(__file__).with_name("plot_figure4c.py"),
                        Path(__file__).with_name("plot_figure4a.py")]],
              "table_sha256": sha256(args.output_dir / (stem + ".md"))}
    (args.output_dir / (stem + "_results.json")).write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
