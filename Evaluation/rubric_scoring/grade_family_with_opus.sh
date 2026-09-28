#!/bin/bash
set -u

if [ "$#" -lt 3 ]; then
  printf 'Usage: %s RESPONSE_FILE FAMILY_NAME MODEL [MODEL ...]\n' "$0" >&2
  exit 2
fi

response_file="$1"
family_name="$2"
shift 2
models=("$@")

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
grade_script="$script_dir/grade_anthropic_batch.py"
rubric_file="${RUBRICS_FILE:-$repo_root/Data/Rubrics/rubrics_all_questions_final_version.json}"
grade_dir="${OUTPUT_DIR:-$repo_root/Outputs/rubric_scoring}"
audit_dir="$grade_dir/recovery/$family_name"
final_output="$grade_dir/${family_name}_response_graded.jsonl"

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  printf 'ERROR: ANTHROPIC_API_KEY is not set\n' >&2
  exit 2
fi

if [ ! -f "$response_file" ]; then
  printf 'ERROR: Response file does not exist: %s\n' "$response_file" >&2
  exit 2
fi

mkdir -p "$audit_dir"

submission_failed=0
for model in "${models[@]}"; do
  output="$audit_dir/${model}_response_graded.jsonl"
  if ! "${PYTHON:-python}" -u "$grade_script" submit \
    --input "$response_file" \
    --rubrics "$rubric_file" \
    --response-model "$model" \
    --judge-model claude-opus-5 \
    --allow-full-run \
    --output "$output"
  then
    submission_failed=1
  fi
done

if [ "$submission_failed" -ne 0 ]; then
  printf 'ERROR: At least one model batch was not submitted; collection skipped\n' >&2
  exit 2
fi

collection_failed=0
for model in "${models[@]}"; do
  manifest="$audit_dir/${model}_response_graded.jsonl.batch.json"
  if ! "${PYTHON:-python}" -u "$grade_script" collect \
    --manifest "$manifest" \
    --wait
  then
    collection_failed=1
  fi
done

if [ "$collection_failed" -ne 0 ]; then
  printf 'ERROR: At least one model needs targeted recovery; final merge withheld\n' >&2
  exit 2
fi

merge_arguments=()
for model in "${models[@]}"; do
  merge_arguments+=(
    --input-result "$audit_dir/${model}_response_graded.jsonl"
    --expected-response-model "$model"
  )
done

"${PYTHON:-python}" -u "$grade_script" merge \
  "${merge_arguments[@]}" \
  --rubrics "$rubric_file" \
  --output "$final_output"
