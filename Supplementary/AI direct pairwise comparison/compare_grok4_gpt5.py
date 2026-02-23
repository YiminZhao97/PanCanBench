import json
import os
import sys
import openai
import tiktoken
sys.path.append('/home/yzhao4/PanCan-QA_LLM/Analysis/ai_direct_judge_vs_rubrics')
from AI_judger import AIDirectJudge
from datetime import datetime

# Load the response data
with open('/home/yzhao4/PanCan-QA_LLM/data_share/gpt5_response.jsonl', 'r', encoding='utf-8') as file:
    gpt5_results = json.load(file)

with open('/home/yzhao4/PanCan-QA_LLM/data_share/grok_response.jsonl', 'r', encoding='utf-8') as file:
    grok_results = json.load(file)


# Create comparisons between grok-4-latest and gpt-4.1 for all questions
comparisons_grok_vs_gpt5 = []

for i in range(len(gpt5_results)):
    gpt_item = gpt5_results[i]
    grok_item = grok_results[i]

    # Extract question and responses
    question = gpt_item.get('question', '')
    question_id = gpt_item.get('question_id', f'Q{i+1}')

    # Get o3 response
    gpt5_response = None
    if 'responses' in gpt_item and 'o3' in gpt_item['responses']:
        o3_response = gpt_item['responses']['o3']

    # Get grok-4-latest response
    grok_4_latest_response = None
    if 'responses' in grok_item and 'grok-4-latest' in grok_item['responses']:
        grok_4_latest_response = grok_item['responses']['grok-4-latest']

    # Only create comparison if both responses exist
    if o3_response and grok_4_latest_response:
        comparison = {
            'question': question,
            'question_number': int(question_id.replace('Q', '')),
            'response_a': grok_4_latest_response,
            'response_b': o3_response,
            'source_a': 'grok-4-latest',
            'source_b': 'o3'
        }
        comparisons_grok_vs_o3.append(comparison)

print(f"Created {len(comparisons_grok_vs_o3)} comparisons between grok-4-latest and o3")


# Initialize AI Judge with API key from environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY env var")

judge = AIDirectJudge(api_key=OPENAI_API_KEY, model="gpt-4.1")

# Perform the judgments
print("Starting AI-based comparison between grok-4-latest and o3...")
judgment_results = judge.judge_multiple_comparisons(comparisons_grok_vs_o3, delay_seconds=1.0)

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"/home/yzhao4/PanCan-QA_LLM/Analysis/ai_direct_judge_vs_rubrics/grok4_vs_o3_judgment_{timestamp}.json"
judge.save_judgment_results(judgment_results, output_file)

# Create summary
summary = judge.create_judgment_summary(judgment_results)
print("\n=== JUDGMENT SUMMARY ===")
print(f"Total comparisons: {summary.get('total_judgments', 0)}")
print(f"Winner distribution: {summary.get('winner_distribution', {})}")
print(f"Source performance: {summary.get('source_performance', {})}")
print(f"Average confidence: {summary.get('average_confidence', 0):.2f}/5")
print(f"High confidence judgments (≥4): {summary.get('high_confidence_judgments', 0)}")
print(f"Ties: {summary.get('ties', 0)}")
print(f"Errors: {summary.get('error_count', 0)}")

# Save summary
summary_file = f"/home/yzhao4/PanCan-QA_LLM/Analysis/ai_direct_judge_vs_rubrics/grok4_vs_o3_summary_{timestamp}.json"
with open(summary_file, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\nResults saved to: {output_file}")
print(f"Summary saved to: {summary_file}")

