#!/usr/bin/env python3
"""Grade PanCanBench responses with Anthropic's Message Batches API.

The script accepts either a JSON array or JSONL response file, validates it
against the final rubric JSON, submits independent structured-output requests,
and converts binary judge decisions into the rubric's weighted point values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from grading_prompt import HUMAN_PROMPT_VERSION, HUMAN_SYSTEM_PROMPT, build_prompt as build_current_prompt


API_ROOT = "https://api.anthropic.com/v1/messages/batches"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_JUDGE_MODEL = "claude-opus-5"
PROMPT_VERSION = HUMAN_PROMPT_VERSION
FULL_RUN_THRESHOLD = 10
FIXED_OBJECT_SCHEMA_MAX_ITEMS = 23

SYSTEM_INSTRUCTION = HUMAN_SYSTEM_PROMPT


class APIError(RuntimeError):
    """An Anthropic API request failed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json_or_jsonl(path: Path) -> Any:
    if not path.is_file():
        raise ValueError(f"File not found: {path}")
    text = path.read_text(encoding="utf-8-sig")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}, line {line_number}: invalid JSONL"
                ) from exc
        return rows


def normalize_question_id(value: Any) -> str:
    text = str(value).strip().upper()
    if text.isdigit():
        text = f"Q{text}"
    if not re.fullmatch(r"Q[1-9]\d*", text):
        raise ValueError(f"Invalid question ID: {value!r}")
    return text


def load_rubrics(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if list(data) != ["questions"] or not isinstance(data["questions"], list):
        raise ValueError(f"{path}: unexpected rubric JSON schema")
    by_id: dict[str, dict[str, Any]] = {}
    for question in data["questions"]:
        number = question.get("question_number")
        if isinstance(number, bool) or not isinstance(number, int):
            raise ValueError(f"{path}: question_number must be an integer")
        question_id = f"Q{number}"
        if question_id in by_id:
            raise ValueError(f"{path}: duplicate {question_id}")
        if not isinstance(question.get("question_text"), str) or not question[
            "question_text"
        ].strip():
            raise ValueError(f"{path}: {question_id} has no question text")
        items = question.get("rubric_items")
        if not isinstance(items, list) or not items:
            raise ValueError(f"{path}: {question_id} has no rubric items")
        expected_numbers = list(range(1, len(items) + 1))
        actual_numbers = [item.get("item_number") for item in items]
        if actual_numbers != expected_numbers:
            raise ValueError(f"{path}: {question_id} numbering is not consecutive")
        for item in items:
            number = item["item_number"]
            description = item.get("description")
            minimum = item.get("min_points")
            maximum = item.get("max_points")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f"{path}: {question_id}.{number} has no description")
            if any(isinstance(value, bool) or not isinstance(value, int) for value in (minimum, maximum)):
                raise ValueError(f"{path}: {question_id}.{number} points must be integers")
            if not (-10 <= minimum <= maximum <= 10):
                raise ValueError(f"{path}: {question_id}.{number} points are invalid")
            if minimum != 0 and maximum != 0:
                raise ValueError(
                    f"{path}: {question_id}.{number} must have zero as one endpoint"
                )
        by_id[question_id] = question
    if sorted(int(key[1:]) for key in by_id) != list(range(1, 283)):
        raise ValueError(f"{path}: expected exactly Q1 through Q282")
    return by_id


def load_responses(
    path: Path, response_model: str | None
) -> tuple[dict[str, dict[str, str]], str]:
    data = read_json_or_jsonl(path)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array or JSONL objects")
    available_models: set[str] = set()
    for row in data:
        if isinstance(row, dict) and isinstance(row.get("responses"), dict):
            available_models.update(str(model) for model in row["responses"])
    if response_model is None:
        if len(available_models) != 1:
            raise ValueError(
                "Specify --response-model; available models are "
                + ", ".join(sorted(available_models))
            )
        response_model = next(iter(available_models))
    if response_model not in available_models:
        raise ValueError(
            f"Response model {response_model!r} not found; available: "
            + ", ".join(sorted(available_models))
        )

    by_id: dict[str, dict[str, str]] = {}
    for row_number, row in enumerate(data, 1):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: response row {row_number} is not an object")
        question_id = normalize_question_id(row.get("question_id"))
        if question_id in by_id:
            raise ValueError(f"{path}: duplicate {question_id}")
        question = row.get("question")
        responses = row.get("responses")
        response = responses.get(response_model) if isinstance(responses, dict) else None
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{path}: {question_id} has no question text")
        if not isinstance(response, str) or not response.strip():
            raise ValueError(
                f"{path}: {question_id} has no response for {response_model}"
            )
        by_id[question_id] = {
            "question": question.strip(),
            "response": response.strip(),
        }
    return by_id, response_model


def select_question_ids(
    responses: dict[str, dict[str, str]],
    requested: list[str] | None,
    limit: int | None,
) -> list[str]:
    if requested:
        selected = [normalize_question_id(value) for value in requested]
        if len(selected) != len(set(selected)):
            raise ValueError("--question-id contains duplicates")
        missing = [question_id for question_id in selected if question_id not in responses]
        if missing:
            raise ValueError("Missing selected responses: " + ", ".join(missing))
    else:
        selected = sorted(responses, key=lambda value: int(value[1:]))
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        selected = selected[:limit]
    return selected


def output_schema_style(item_count: int) -> str:
    return "fixed_object" if item_count <= FIXED_OBJECT_SCHEMA_MAX_ITEMS else "array"


def output_schema(item_count: int, style: str) -> dict[str, Any]:
    if style == "array":
        return {
            "type": "object",
            "properties": {
                "grades": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "rubric_item": {"type": "integer"},
                            "meets_criterion": {"type": "boolean"},
                            "rationale": {"type": "string", "minLength": 1},
                        },
                        "required": [
                            "rubric_item",
                            "meets_criterion",
                            "rationale",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["grades"],
            "additionalProperties": False,
        }
    if style != "fixed_object":
        raise ValueError(f"Unknown output schema style: {style}")
    grade_properties = {
        str(number): {
            "type": "object",
            "properties": {
                "meets_criterion": {"type": "boolean"},
                "rationale": {"type": "string", "minLength": 1},
            },
            "required": ["meets_criterion", "rationale"],
            "additionalProperties": False,
        }
        for number in range(1, item_count + 1)
    }
    return {
        "type": "object",
        "properties": {
            "grades": {
                "type": "object",
                "properties": grade_properties,
                "required": list(grade_properties),
                "additionalProperties": False,
            }
        },
        "required": ["grades"],
        "additionalProperties": False,
    }



def build_prompt(
    question: str,
    response: str,
    rubric_items: list[dict[str, Any]],
    schema_style: str,
) -> str:
    return build_current_prompt(question, response, rubric_items, schema_style, "human")


def system_value(cache_ttl: str) -> str | list[dict[str, Any]]:
    if cache_ttl == "none":
        return SYSTEM_INSTRUCTION
    return [
        {
            "type": "text",
            "text": SYSTEM_INSTRUCTION,
            "cache_control": {"type": "ephemeral", "ttl": cache_ttl},
        }
    ]


def custom_id(question_id: str, response_model: str) -> str:
    model_slug = re.sub(r"[^A-Za-z0-9_-]", "_", response_model)[:40]
    return f"{question_id.lower()}_{model_slug}"


def make_request(
    question_id: str,
    response_row: dict[str, str],
    rubric: dict[str, Any],
    response_model: str,
    judge_model: str,
    max_tokens: int,
    cache_ttl: str,
) -> dict[str, Any]:
    schema_style = output_schema_style(len(rubric["rubric_items"]))
    return {
        "custom_id": custom_id(question_id, response_model),
        "params": {
            "model": judge_model,
            "max_tokens": max_tokens,
            "system": system_value(cache_ttl),
            "messages": [
                {
                    "role": "user",
                    "content": build_prompt(
                        response_row["question"],
                        response_row["response"],
                        rubric["rubric_items"],
                        schema_style,
                    ),
                }
            ],
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": output_schema(
                        len(rubric["rubric_items"]), schema_style
                    ),
                }
            },
        },
    }


def api_request(
    method: str,
    url: str,
    api_key: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 300.0,
) -> bytes:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise APIError(f"Anthropic HTTP {exc.code}: {detail[:3000]}") from exc
    except urllib.error.URLError as exc:
        raise APIError(f"Anthropic network error: {exc.reason}") from exc


def api_json(
    method: str,
    url: str,
    api_key: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 300.0,
) -> dict[str, Any]:
    return json.loads(api_request(method, url, api_key, payload, timeout))


def require_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise ValueError("ANTHROPIC_API_KEY is not set")
    return key


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    rubrics = load_rubrics(args.rubrics)
    responses, response_model = load_responses(args.input, args.response_model)
    selected = select_question_ids(responses, args.question_id, args.limit)
    missing_rubrics = [question_id for question_id in selected if question_id not in rubrics]
    if missing_rubrics:
        raise ValueError("Missing rubrics: " + ", ".join(missing_rubrics))
    if len(selected) > FULL_RUN_THRESHOLD and not args.allow_full_run:
        raise ValueError(
            f"Refusing to submit {len(selected)} requests without --allow-full-run"
        )
    requests = [
        make_request(
            question_id,
            responses[question_id],
            rubrics[question_id],
            response_model,
            args.judge_model,
            args.max_tokens,
            args.cache_system_ttl,
        )
        for question_id in selected
    ]
    return {
        "rubrics": rubrics,
        "responses": responses,
        "response_model": response_model,
        "question_ids": selected,
        "requests": requests,
    }


def validate_command(args: argparse.Namespace) -> None:
    prepared = prepare(args)
    criteria = sum(
        len(prepared["rubrics"][question_id]["rubric_items"])
        for question_id in prepared["question_ids"]
    )
    print(
        f"Validated {len(prepared['question_ids'])} requests and {criteria} criteria "
        f"for response model {prepared['response_model']}"
    )
    print("Question IDs: " + ", ".join(prepared["question_ids"]))


def submit(args: argparse.Namespace) -> Path:
    prepared = prepare(args)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}")
    manifest = args.manifest or args.output.with_suffix(args.output.suffix + ".batch.json")
    if manifest.exists() and not args.overwrite:
        raise FileExistsError(f"Manifest exists: {manifest}")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    key = require_api_key()
    batch = api_json(
        "POST",
        API_ROOT,
        key,
        {"requests": prepared["requests"]},
        args.timeout,
    )
    record = {
        "prompt_version": PROMPT_VERSION,
        "created_at": utc_now(),
        "batch_id": batch["id"],
        "batch": batch,
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "rubrics": str(args.rubrics.resolve()),
        "rubrics_sha256": sha256(args.rubrics),
        "response_model": prepared["response_model"],
        "judge_model": args.judge_model,
        "question_ids": prepared["question_ids"],
        "max_tokens": args.max_tokens,
        "cache_system_ttl": args.cache_system_ttl,
        "output": str(args.output.resolve()),
    }
    atomic_json(manifest, record)
    print(f"Submitted batch {batch['id']} with {len(prepared['question_ids'])} requests")
    print(f"Manifest: {manifest}")
    return manifest


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "batch_id",
        "input",
        "input_sha256",
        "rubrics",
        "rubrics_sha256",
        "response_model",
        "judge_model",
        "question_ids",
        "output",
    ):
        if key not in manifest:
            raise ValueError(f"{path}: missing manifest field {key}")
    input_path = Path(manifest["input"])
    rubric_path = Path(manifest["rubrics"])
    if sha256(input_path) != manifest["input_sha256"]:
        raise ValueError("Input file changed after batch submission")
    if sha256(rubric_path) != manifest["rubrics_sha256"]:
        raise ValueError("Rubric file changed after batch submission")
    return manifest


def get_status(manifest: dict[str, Any], api_key: str, timeout: float) -> dict[str, Any]:
    return api_json(
        "GET", f"{API_ROOT}/{manifest['batch_id']}", api_key, timeout=timeout
    )


def status_command(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    status = get_status(manifest, require_api_key(), args.timeout)
    print(json.dumps(status, ensure_ascii=False, indent=2))


def wait_for_end(
    manifest: dict[str, Any], api_key: str, timeout: float, poll_seconds: float
) -> dict[str, Any]:
    while True:
        status = get_status(manifest, api_key, timeout)
        state = status.get("processing_status")
        counts = status.get("request_counts")
        print(f"Batch {manifest['batch_id']}: {state}; counts={counts}")
        if state == "ended":
            return status
        time.sleep(poll_seconds)


def extract_structured_message(message: dict[str, Any]) -> dict[str, Any]:
    pieces = [
        block["text"]
        for block in message.get("content", [])
        if block.get("type") == "text" and isinstance(block.get("text"), str)
    ]
    if not pieces:
        raise ValueError("Succeeded batch result contains no text block")
    return json.loads("".join(pieces))


def normalize_result(
    question_id: str,
    result_row: dict[str, Any],
    rubric: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    result = result_row.get("result")
    if not isinstance(result, dict) or result.get("type") != "succeeded":
        raise ValueError(f"{question_id}: batch request did not succeed: {result}")
    message = result.get("message")
    if not isinstance(message, dict):
        raise ValueError(f"{question_id}: succeeded result has no message")
    parsed = extract_structured_message(message)
    grades = parsed.get("grades") if isinstance(parsed, dict) else None
    items = rubric["rubric_items"]
    if isinstance(grades, dict):
        expected_keys = {str(item["item_number"]) for item in items}
        if set(grades) != expected_keys:
            raise ValueError(f"{question_id}: incorrect grade keys")
        converted_grades = []
        for item in items:
            raw_grade = grades[str(item["item_number"])]
            if not isinstance(raw_grade, dict):
                raise ValueError(f"{question_id}: grade is not an object")
            converted_grades.append(
                {"rubric_item": item["item_number"], **raw_grade}
            )
        grades = converted_grades
    if not isinstance(grades, list) or len(grades) != len(items):
        raise ValueError(f"{question_id}: incorrect grade count")

    criterion_scores = []
    for item, grade in zip(items, grades):
        if not isinstance(grade, dict):
            raise ValueError(f"{question_id}: grade is not an object")
        item_number = item["item_number"]
        if grade.get("rubric_item") != item_number:
            raise ValueError(f"{question_id}: rubric sequence mismatch")
        meets = grade.get("meets_criterion")
        rationale = grade.get("rationale")
        if not isinstance(meets, bool) or not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(f"{question_id}.{item_number}: invalid structured grade")
        minimum = item["min_points"]
        maximum = item["max_points"]
        awarded = (minimum if minimum < 0 else maximum) if meets else 0
        criterion_scores.append(
            {
                "criterion_number": item_number,
                "description": item["description"],
                "meets_criterion": meets,
                "score_given": awarded,
                "min_points": minimum,
                "max_points": maximum,
                "justification": rationale.strip(),
            }
        )
    total_score = sum(item["score_given"] for item in criterion_scores)
    maximum_score = sum(item["max_points"] for item in criterion_scores)
    minimum_score = sum(item["min_points"] for item in criterion_scores)
    return {
        "question_id": question_id,
        "question_number": int(question_id[1:]),
        "source": manifest["response_model"],
        "response_model": manifest["response_model"],
        "grader_model": manifest["judge_model"],
        "judge_model": manifest["judge_model"],
        "criterion_scores": criterion_scores,
        "total_score": total_score,
        "minimum_possible_score": minimum_score,
        "max_possible_score": maximum_score,
        "percentage": 100.0 * total_score / maximum_score if maximum_score else None,
        "batch_id": manifest["batch_id"],
        "prompt_version": manifest["prompt_version"],
        "input_sha256": manifest["input_sha256"],
        "rubrics_sha256": manifest["rubrics_sha256"],
        "graded_at": utc_now(),
        "provider_response_id": message.get("id"),
        "provider_model": message.get("model"),
        "usage": message.get("usage"),
        "stop_reason": message.get("stop_reason"),
    }


def collect(
    manifest_path: Path,
    wait: bool,
    poll_seconds: float,
    timeout: float,
    overwrite: bool,
) -> Path:
    manifest = load_manifest(manifest_path)
    api_key = require_api_key()
    status = (
        wait_for_end(manifest, api_key, timeout, poll_seconds)
        if wait
        else get_status(manifest, api_key, timeout)
    )
    if status.get("processing_status") != "ended":
        raise RuntimeError("Batch has not ended; rerun collect with --wait")
    results_url = status.get("results_url") or f"{API_ROOT}/{manifest['batch_id']}/results"
    raw = api_request("GET", results_url, api_key, timeout=timeout).decode("utf-8")
    result_rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    by_custom_id = {row["custom_id"]: row for row in result_rows}

    rubric_path = Path(manifest["rubrics"])
    rubrics = load_rubrics(rubric_path)
    normalized = []
    failures = []
    for question_id in manifest["question_ids"]:
        request_id = custom_id(question_id, manifest["response_model"])
        if request_id not in by_custom_id:
            failures.append(
                {
                    "question_id": question_id,
                    "custom_id": request_id,
                    "error": "Batch results omitted this custom_id",
                }
            )
            continue
        try:
            normalized.append(
                normalize_result(
                    question_id,
                    by_custom_id[request_id],
                    rubrics[question_id],
                    manifest,
                )
            )
        except ValueError as exc:
            failures.append(
                {
                    "question_id": question_id,
                    "custom_id": request_id,
                    "error": str(exc),
                    "provider_result": by_custom_id[request_id].get("result"),
                }
            )

    output = Path(manifest["output"])
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output}")
    if failures:
        partial = output.with_name(output.stem + ".partial" + output.suffix)
        failure_path = output.with_name(output.stem + ".failures.json")
        for path in (partial, failure_path):
            if path.exists() and not overwrite:
                raise FileExistsError(f"Recovery output exists: {path}")
        atomic_jsonl(partial, normalized)
        atomic_json(
            failure_path,
            {
                "batch_id": manifest["batch_id"],
                "successful_results": len(normalized),
                "failed_results": len(failures),
                "failures": failures,
            },
        )
        print(f"Preserved {len(normalized)} validated results: {partial}")
        print(f"Recorded {len(failures)} failures: {failure_path}")
        raise RuntimeError(
            f"Batch contained {len(failures)} failed result(s); final output was not created"
        )
    atomic_jsonl(output, normalized)
    print(f"Collected and validated {len(normalized)} results: {output}")
    return output


def validate_normalized_row(
    row: dict[str, Any], rubrics: dict[str, dict[str, Any]]
) -> str:
    question_id = normalize_question_id(row.get("question_id"))
    if question_id not in rubrics:
        raise ValueError(f"Normalized result has unknown question ID {question_id}")
    criteria = row.get("criterion_scores")
    items = rubrics[question_id]["rubric_items"]
    if not isinstance(criteria, list) or len(criteria) != len(items):
        raise ValueError(f"{question_id}: normalized criterion count is incorrect")
    for item, criterion in zip(items, criteria):
        number = item["item_number"]
        if not isinstance(criterion, dict) or criterion.get("criterion_number") != number:
            raise ValueError(f"{question_id}: normalized criterion sequence mismatch")
        if criterion.get("description") != item["description"]:
            raise ValueError(f"{question_id}.{number}: rubric description mismatch")
        if criterion.get("min_points") != item["min_points"]:
            raise ValueError(f"{question_id}.{number}: min_points mismatch")
        if criterion.get("max_points") != item["max_points"]:
            raise ValueError(f"{question_id}.{number}: max_points mismatch")
        meets = criterion.get("meets_criterion")
        if not isinstance(meets, bool):
            raise ValueError(f"{question_id}.{number}: meets_criterion is not boolean")
        expected_score = (
            item["min_points"] if item["min_points"] < 0 else item["max_points"]
        ) if meets else 0
        if criterion.get("score_given") != expected_score:
            raise ValueError(f"{question_id}.{number}: weighted score is incorrect")
    expected_total = sum(criterion["score_given"] for criterion in criteria)
    if row.get("total_score") != expected_total:
        raise ValueError(f"{question_id}: total_score is incorrect")
    return question_id


def merge_command(args: argparse.Namespace) -> None:
    rubrics = load_rubrics(args.rubrics)
    by_id: dict[tuple[str, str], dict[str, Any]] = {}
    signatures = set()
    observed_models = set()
    for path in args.input_result:
        data = read_json_or_jsonl(path)
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            raise ValueError(f"{path}: expected JSONL or a JSON array")
        for row in data:
            if not isinstance(row, dict):
                raise ValueError(f"{path}: normalized result is not an object")
            question_id = validate_normalized_row(row, rubrics)
            response_model = row.get("response_model")
            if not isinstance(response_model, str) or not response_model:
                raise ValueError(f"{path}: {question_id} has no response_model")
            key = (response_model, question_id)
            if key in by_id:
                raise ValueError(
                    f"Duplicate normalized result for {response_model} {question_id}"
                )
            by_id[key] = row
            observed_models.add(response_model)
            signatures.add(
                (
                    row.get("judge_model"),
                    row.get("prompt_version"),
                    row.get("input_sha256"),
                    row.get("rubrics_sha256"),
                )
            )
    if len(signatures) != 1:
        raise ValueError("Normalized result files have inconsistent provenance")
    expected_models = args.expected_response_model or sorted(observed_models)
    if len(expected_models) != len(set(expected_models)):
        raise ValueError("--expected-response-model contains duplicates")
    expected = {
        (model, question_id)
        for model in expected_models
        for question_id in rubrics
    }
    actual = set(by_id)
    if actual != expected:
        missing = sorted(
            expected - actual, key=lambda value: (value[0], int(value[1][1:]))
        )
        extra = sorted(
            actual - expected, key=lambda value: (value[0], int(value[1][1:]))
        )
        raise ValueError(
            "Merged results do not cover every expected model and Q1-Q282; "
            f"missing={missing or 'none'}, extra={extra or 'none'}"
        )
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}")
    ordered = [
        by_id[(model, f"Q{number}")]
        for model in expected_models
        for number in range(1, 283)
    ]
    atomic_jsonl(args.output, ordered)
    print(
        f"Merged and validated {len(ordered)} results across "
        f"{len(expected_models)} response model(s): {args.output}"
    )


def collect_command(args: argparse.Namespace) -> None:
    collect(
        args.manifest,
        args.wait,
        args.poll_seconds,
        args.timeout,
        args.overwrite,
    )


def run_command(args: argparse.Namespace) -> None:
    manifest = submit(args)
    collect(manifest, True, args.poll_seconds, args.timeout, args.overwrite)


def add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--rubrics", type=Path, required=True)
    parser.add_argument("--response-model")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--question-id", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument(
        "--cache-system-ttl", choices=("none", "5m", "1h"), default="none"
    )
    parser.add_argument(
        "--allow-full-run",
        action="store_true",
        help=f"Required when submitting more than {FULL_RUN_THRESHOLD} requests",
    )


def add_api_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--timeout", type=float, default=300.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    add_selection_arguments(validate_parser)
    validate_parser.set_defaults(function=validate_command)

    submit_parser = subparsers.add_parser("submit")
    add_selection_arguments(submit_parser)
    add_api_arguments(submit_parser)
    submit_parser.add_argument("--output", type=Path, required=True)
    submit_parser.add_argument("--manifest", type=Path)
    submit_parser.add_argument("--overwrite", action="store_true")
    submit_parser.set_defaults(function=submit)

    run_parser = subparsers.add_parser("run")
    add_selection_arguments(run_parser)
    add_api_arguments(run_parser)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--manifest", type=Path)
    run_parser.add_argument("--overwrite", action="store_true")
    run_parser.add_argument("--poll-seconds", type=float, default=15.0)
    run_parser.set_defaults(function=run_command)

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--manifest", type=Path, required=True)
    add_api_arguments(status_parser)
    status_parser.set_defaults(function=status_command)

    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--manifest", type=Path, required=True)
    collect_parser.add_argument("--wait", action="store_true")
    collect_parser.add_argument("--poll-seconds", type=float, default=15.0)
    collect_parser.add_argument("--overwrite", action="store_true")
    add_api_arguments(collect_parser)
    collect_parser.set_defaults(function=collect_command)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument(
        "--input-result", type=Path, action="append", required=True
    )
    merge_parser.add_argument("--expected-response-model", action="append")
    merge_parser.add_argument("--rubrics", type=Path, required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    merge_parser.add_argument("--overwrite", action="store_true")
    merge_parser.set_defaults(function=merge_command)
    return parser.parse_args()


def main() -> None:
    try:
        args = parse_args()
        args.function(args)
    except (APIError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
