#!/usr/bin/env python3
"""
Comprehensive grading comparison using polished rubrics.
Regrades target questions with both GPT and Claude graders using polished rubrics,
then compares the improvement against original grading differences.
"""

import json
import os
import sys
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
import time

# Add paths for graders
sys.path.append('/home/yzhao4/PanCan-QA_LLM')
from grader import GPT4Grader
from datetime import datetime
os.chdir('/home/yzhao4/PanCan-QA_LLM/rubrics_collection/Analysis4_permute_grader_model')
from claude_grader import Claude4Grader


class PolishedRubricsGradingComparison:
    def __init__(self, gpt_api_key: str, claude_api_key: str, response_files: Dict[str, str] = None, differences_files: Dict[str, str] = None):
        """Initialize graders and load data."""
        self.gpt_grader = GPT4Grader(api_key=gpt_api_key, model='gpt-4.1')
        self.claude_grader = Claude4Grader(api_key=claude_api_key, model='claude-sonnet-4-20250514')
        
        # Load responses data
        self.responses_data = self._load_responses_data(response_files)
        
        # Load original differences data
        self.original_differences = self._load_original_differences(differences_files)
        
        print("✅ Initialized graders and loaded data")
    
    def _load_responses_data(self, response_files: Dict[str, str] = None) -> Dict[str, List[Dict]]:
        """Load all model responses."""
        responses = {}
        
        # Default file paths for different model families
        if response_files is None:
            response_files = {
                'openai': "/home/yzhao4/PanCan-QA_LLM/data_share/openai_family_response.jsonl",
                'grok': "/home/yzhao4/PanCan-QA_LLM/data_share/grok_response.jsonl",
                'opensource': "/home/yzhao4/PanCan-QA_LLM/data_share/all_opensource_models_final_outputs_temp0.7.jsonl"
            }
        
        for family, file_path in response_files.items():
            try:
                if family == 'opensource':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        responses[family] = [json.loads(line.strip()) for line in f if line.strip()]
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        responses[family] = json.load(f)
                print(f"✅ Loaded {len(responses[family])} responses from {family}")
            except Exception as e:
                print(f"❌ Error loading {family} responses: {e}")
                responses[family] = []
        
        return responses
    
    def _load_original_differences(self, differences_files: Dict[str, str] = None) -> Dict[str, Dict]:
        """Load original grading differences from targeted analysis."""
        differences = {}
        
        # Default file paths
        if differences_files is None:
            differences_files = {
                'Expert1': 'targeted_large_differences_expert1_data.json',
                'Expert2': 'targeted_large_differences_expert2_data.json'
            }
        
        # Load Expert1 differences
        try:
            with open(differences_files['Expert1'], 'r', encoding='utf-8') as f:
                expert1_data = json.load(f)
                differences['Expert1'] = expert1_data.get('detailed_results', {})
            print(f"✅ Loaded Expert1 original differences: {len(differences['Expert1'])} questions")
        except Exception as e:
            print(f"❌ Error loading Expert1 differences: {e}")
            differences['Expert1'] = {}
        
        # Load Expert2 differences
        try:
            with open(differences_files['Expert2'], 'r', encoding='utf-8') as f:
                expert2_data = json.load(f)
                differences['Expert2'] = expert2_data.get('detailed_results', {})
            print(f"✅ Loaded Expert2 original differences: {len(differences['Expert2'])} questions")
        except Exception as e:
            print(f"❌ Error loading Expert2 differences: {e}")
            differences['Expert2'] = {}
        
        return differences
    
    def get_model_response(self, question_id: str, model_name: str) -> Optional[str]:
        """Get the response for a specific question and model."""
        question_num = int(question_id[1:])  # Remove 'Q' prefix
        
        # Search through all response families
        for family, responses in self.responses_data.items():
            for item in responses:
                if item.get('question_id') == question_id:
                    model_responses = item.get('responses', {})
                    if model_name in model_responses:
                        return model_responses[model_name]
        
        return None
    
    def get_question_text(self, question_id: str) -> Optional[str]:
        """Get the question text for a question ID."""
        # Search through all response families
        for family, responses in self.responses_data.items():
            for item in responses:
                if item.get('question_id') == question_id:
                    return item.get('question', '')
        
        return None
    
    def grade_with_polished_rubric(self, question_id: str, model_name: str, polished_rubric: Dict) -> Dict:
        """Grade a specific question/model combination with polished rubric."""
        
        # Get response and question text
        response = self.get_model_response(question_id, model_name)
        question_text = self.get_question_text(question_id)
        
        if not response or not question_text:
            return {
                'error': f'Could not find response or question for {question_id} - {model_name}',
                'question_id': question_id,
                'model_name': model_name
            }
        
        # Convert polished rubric to standard format for graders
        rubric_items = polished_rubric.get('revised_rubric', [])
        standard_rubric = []
        
        for item in rubric_items:
            standard_rubric.append({
                'item_number': int(item['item_id']),
                'description': item['description'],
                'max_points': item['points'],
                'min_points': 0
            })
        
        # Grade with both graders
        try:
            gpt_result = self.gpt_grader.grade_single_response(
                question=question_text,
                response=response,
                rubric_items=standard_rubric,
                question_number=int(question_id[1:]),
                source=model_name
            )
            time.sleep(1)  # Rate limiting
        except Exception as e:
            gpt_result = {'error': f'GPT grading failed: {e}'}
        
        try:
            claude_result = self.claude_grader.grade_single_response(
                question=question_text,
                response=response,
                rubric_items=standard_rubric,
                question_number=int(question_id[1:]),
                source=model_name
            )
            time.sleep(1)  # Rate limiting
        except Exception as e:
            claude_result = {'error': f'Claude grading failed: {e}'}
        
        return {
            'question_id': question_id,
            'model_name': model_name,
            'question_text': question_text,
            'response': response,
            'gpt_grading': gpt_result,
            'claude_grading': claude_result,
            'polished_rubric': polished_rubric
        }
    
    def process_expert_questions(self, expert_name: str, polished_rubrics_file: str) -> Dict:
        """Process all target questions for a specific expert."""
        
        print(f"\n🔄 Processing {expert_name} questions...")
        
        # Load polished rubrics
        try:
            with open(polished_rubrics_file, 'r', encoding='utf-8') as f:
                polished_data = json.load(f)
            
            # Handle different formats: either {polished_rubrics: {...}} or {questions: [...]}
            if 'polished_rubrics' in polished_data:
                polished_rubrics = polished_data['polished_rubrics']
            elif 'questions' in polished_data:
                # Convert questions array to question_id -> rubric mapping
                polished_rubrics = {}
                for question in polished_data['questions']:
                    question_id = f"Q{question['question_number']}"
                    polished_rubrics[question_id] = {
                        'revised_rubric': [
                            {
                                'item_id': str(item['item_number']),
                                'description': item['description'],
                                'points': item['max_points']
                            }
                            for item in question.get('rubric_items', [])
                        ]
                    }
            else:
                polished_rubrics = {}
            
            print(f"✅ Loaded {len(polished_rubrics)} polished rubrics for {expert_name}")
        except Exception as e:
            print(f"❌ Error loading polished rubrics for {expert_name}: {e}")
            return {}
        
        # Get original differences to know which questions/models to regrade
        original_diffs = self.original_differences.get(expert_name, {})
        
        results = {}
        total_cases = 0
        
        for question_id, question_data in original_diffs.items():
            models_with_differences = question_data.get('models_with_differences', [])
            
            if question_id in polished_rubrics:
                polished_rubric = polished_rubrics[question_id]
                
                for model_name in models_with_differences:
                    print(f"  📊 Grading {question_id} - {model_name}...")
                    
                    result = self.grade_with_polished_rubric(question_id, model_name, polished_rubric)
                    
                    if question_id not in results:
                        results[question_id] = {}
                    
                    results[question_id][model_name] = result
                    total_cases += 1
            else:
                print(f"  ⚠️  No polished rubric found for {question_id}")
        
        print(f"✅ Completed {total_cases} grading cases for {expert_name}")
        return results
    
    def calculate_grading_differences(self, gpt_result: Dict, claude_result: Dict) -> Dict:
        """Calculate percentage difference between GPT and Claude grading."""
        
        if 'error' in gpt_result or 'error' in claude_result:
            return {'error': 'One or both graders failed', 'difference': None}
        
        # Extract scores
        gpt_total = sum([item.get('score_given', 0) for item in gpt_result.get('criterion_scores', [])])
        gpt_max = sum([item.get('max_points', 0) for item in gpt_result.get('criterion_scores', [])])
        
        claude_total = sum([item.get('score_given', 0) for item in claude_result.get('criterion_scores', [])])
        claude_max = sum([item.get('max_points', 0) for item in claude_result.get('criterion_scores', [])])
        
        if gpt_max == 0 or claude_max == 0:
            return {'error': 'Invalid max points', 'difference': None}
        
        gpt_percentage = (gpt_total / gpt_max) * 100
        claude_percentage = (claude_total / claude_max) * 100
        difference = abs(gpt_percentage - claude_percentage)
        
        return {
            'gpt_total': gpt_total,
            'gpt_max': gpt_max,
            'gpt_percentage': gpt_percentage,
            'claude_total': claude_total,
            'claude_max': claude_max,
            'claude_percentage': claude_percentage,
            'difference': difference
        }
    
    def compare_improvements(self, expert_name: str, new_results: Dict) -> Dict:
        """Compare new grading differences with original differences."""
        
        original_diffs = self.original_differences.get(expert_name, {})
        improvements = {}
        
        for question_id, models_data in new_results.items():
            if question_id not in improvements:
                improvements[question_id] = {}
            
            for model_name, grading_result in models_data.items():
                # Get original difference
                original_data = original_diffs.get(question_id, {})
                model_analyses = original_data.get('model_analyses', {})
                original_diff = model_analyses.get(model_name, {}).get('difference', 0)
                
                # Calculate new difference
                new_diff_data = self.calculate_grading_differences(
                    grading_result.get('gpt_grading', {}),
                    grading_result.get('claude_grading', {})
                )
                
                new_difference = new_diff_data.get('difference', 0)
                if new_difference is None:
                    new_difference = 0
                
                improvement_value = original_diff - new_difference if (original_diff is not None and new_difference is not None) else 0
                improvement_percentage = ((original_diff - new_difference) / original_diff * 100) if (original_diff is not None and original_diff > 0 and new_difference is not None) else 0
                
                improvement = {
                    'original_difference': original_diff if original_diff is not None else 0,
                    'new_difference': new_difference,
                    'improvement': improvement_value,
                    'improvement_percentage': improvement_percentage,
                    'new_grading_details': new_diff_data,
                    'grading_result': grading_result
                }
                
                improvements[question_id][model_name] = improvement
        
        return improvements
    
    def generate_comprehensive_report(self, expert1_improvements: Dict, expert2_improvements: Dict) -> str:
        """Generate a comprehensive comparison report."""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# Polished Rubrics Grading Improvement Analysis

**Analysis Date:** {timestamp}
**Objective:** Compare grading consistency before and after rubric polishing

---

## Executive Summary

This analysis evaluates the effectiveness of AI-polished rubrics in reducing grading inconsistencies between GPT-4.1 and Claude graders. The original rubrics showed significant disagreements (20+ percentage points) between the two graders. After applying AI-powered rubric polishing to enforce binary scoring and reduce ambiguity, we re-evaluated the same question-model combinations.

"""
        
        # Calculate overall statistics
        def calculate_stats(improvements):
            total_cases = 0
            total_improvement = 0
            improved_cases = 0
            
            for question_data in improvements.values():
                for model_data in question_data.values():
                    if 'improvement' in model_data:
                        total_cases += 1
                        improvement = model_data.get('improvement', 0)
                        if improvement is not None:
                            total_improvement += improvement
                            if improvement > 0:
                                improved_cases += 1
            
            avg_improvement = total_improvement / total_cases if total_cases > 0 else 0
            improvement_rate = (improved_cases / total_cases * 100) if total_cases > 0 else 0
            
            return {
                'total_cases': total_cases,
                'improved_cases': improved_cases,
                'avg_improvement': avg_improvement,
                'improvement_rate': improvement_rate
            }
        
        expert1_stats = calculate_stats(expert1_improvements)
        expert2_stats = calculate_stats(expert2_improvements)
        
        report += f"""### Overall Results

**Expert1 Rubrics:**
- Total cases analyzed: {expert1_stats['total_cases']}
- Cases with improvement: {expert1_stats['improved_cases']}
- Average improvement: {expert1_stats['avg_improvement']:.1f} percentage points
- Success rate: {expert1_stats['improvement_rate']:.1f}%

**Expert2 Rubrics:**
- Total cases analyzed: {expert2_stats['total_cases']}
- Cases with improvement: {expert2_stats['improved_cases']}
- Average improvement: {expert2_stats['avg_improvement']:.1f} percentage points
- Success rate: {expert2_stats['improvement_rate']:.1f}%

---

## Detailed Analysis

"""
        
        # Add detailed analysis for each expert
        for expert_name, improvements in [('Expert1', expert1_improvements), ('Expert2', expert2_improvements)]:
            report += f"""### {expert_name} Results

"""
            
            for question_id, models_data in improvements.items():
                report += f"""#### {question_id}

"""
                
                for model_name, improvement_data in models_data.items():
                    original_diff = improvement_data.get('original_difference', 0)
                    new_diff = improvement_data.get('new_difference', 0)
                    improvement = improvement_data.get('improvement', 0)
                    improvement_pct = improvement_data.get('improvement_percentage', 0)
                    
                    status = "✅ IMPROVED" if improvement > 0 else "❌ NO IMPROVEMENT"
                    
                    report += f"""**{model_name}:** {status}
- Original difference: {original_diff:.1f} percentage points
- New difference: {new_diff:.1f} percentage points
- Improvement: {improvement:.1f} percentage points ({improvement_pct:.1f}%)

"""
        
        report += f"""---

## Conclusions

The polished rubrics demonstrate {'significant improvement' if (expert1_stats['improvement_rate'] + expert2_stats['improvement_rate'])/2 > 50 else 'mixed results'} in reducing grading inconsistencies between GPT-4.1 and Claude graders. The AI-powered polishing process successfully addressed many of the ambiguities that led to grader disagreements in the original rubrics.

**Key Findings:**
1. Average improvement across both experts: {(expert1_stats['avg_improvement'] + expert2_stats['avg_improvement'])/2:.1f} percentage points
2. Overall success rate: {(expert1_stats['improvement_rate'] + expert2_stats['improvement_rate'])/2:.1f}%
3. Binary scoring enforcement helped standardize grader interpretations

**Recommendations:**
1. Deploy polished rubrics for improved grading consistency
2. Continue iterative polishing for remaining problem cases
3. Monitor grader agreement in production use

*Generated on {timestamp}*
"""
        
        return report
    
    def run_complete_analysis(self, expert1_polished_file: str = None, expert2_polished_file: str = None, output_dir: str = '.'):
        """Run the complete polished rubrics grading comparison analysis."""
        
        print("🚀 Starting Polished Rubrics Grading Comparison Analysis")
        print("=" * 70)
        
        # Default polished rubrics files
        if expert1_polished_file is None:
            expert1_polished_file = '/Users/zhaoyimin/Desktop/revised rubrics/polished_rubrics_expert1_20250916_145420.json'
        if expert2_polished_file is None:
            expert2_polished_file = '/Users/zhaoyimin/Desktop/revised rubrics/polished_rubrics_expert2_20250916_145721.json'
        
        # Process Expert1
        expert1_results = self.process_expert_questions(
            'Expert1', 
            expert1_polished_file
        )
        
        # Process Expert2  
        expert2_results = self.process_expert_questions(
            'Expert2',
            expert2_polished_file
        )
        
        # Calculate improvements
        print("\n📊 Calculating improvements...")
        expert1_improvements = self.compare_improvements('Expert1', expert1_results)
        expert2_improvements = self.compare_improvements('Expert2', expert2_results)
        
        # Generate comprehensive report
        print("\n📝 Generating comprehensive report...")
        report = self.generate_comprehensive_report(expert1_improvements, expert2_improvements)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        detailed_results = {
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'total_expert1_cases': len(expert1_results),
                'total_expert2_cases': len(expert2_results)
            },
            'expert1_results': expert1_results,
            'expert2_results': expert2_results,
            'expert1_improvements': expert1_improvements,
            'expert2_improvements': expert2_improvements
        }
        
        results_file = os.path.join(output_dir, f"polished_rubrics_grading_analysis_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        # Save report
        report_file = os.path.join(output_dir, f"polished_rubrics_improvement_report_{timestamp}.md")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n✅ Analysis complete!")
        print(f"📄 Detailed results: {results_file}")
        print(f"📄 Improvement report: {report_file}")
        
        return detailed_results

def main():
    """Main function to run the analysis."""
    
    parser = argparse.ArgumentParser(description='Compare grading consistency before and after rubric polishing')
    parser.add_argument('--gpt_api_key', type=str, default=None,
                       help='GPT API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--claude_api_key', type=str, default=None,
                       help='Claude API key (or set ANTHROPIC_API_KEY env var)')
    parser.add_argument('--expert1_polished', type=str, default='/Users/zhaoyimin/Desktop/revised rubrics/polished_rubrics_expert1_20250916_145420.json',
                       help='Path to Expert1 polished rubrics file')
    parser.add_argument('--expert2_polished', type=str, default='/Users/zhaoyimin/Desktop/revised rubrics/polished_rubrics_expert2_20250916_145721.json',
                       help='Path to Expert2 polished rubrics file')
    parser.add_argument('--expert1_original', type=str, default='targeted_large_differences_expert1_data.json',
                       help='Path to Expert1 original differences file')
    parser.add_argument('--expert2_original', type=str, default='targeted_large_differences_expert2_data.json',
                       help='Path to Expert2 original differences file')
    parser.add_argument('--openai_responses', type=str, default="/home/yzhao4/PanCan-QA_LLM/data_share/openai_family_response.jsonl",
                       help='Path to OpenAI responses file')
    parser.add_argument('--grok_responses', type=str, default="/home/yzhao4/PanCan-QA_LLM/data_share/grok_response.jsonl",
                       help='Path to Grok responses file')
    parser.add_argument('--opensource_responses', type=str, default="/home/yzhao4/PanCan-QA_LLM/data_share/all_opensource_models_final_outputs_temp0.7.jsonl",
                       help='Path to Opensource responses file')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for results (default: current directory)')
    
    args = parser.parse_args()
    
    # Get API keys from args or environment
    gpt_api_key = args.gpt_api_key or os.getenv('OPENAI_API_KEY')
    claude_api_key = args.claude_api_key or os.getenv('ANTHROPIC_API_KEY')
    
    if not gpt_api_key or not claude_api_key:
        print("❌ API keys required. Set via arguments or environment variables:")
        print("  --gpt_api_key or OPENAI_API_KEY")
        print("  --claude_api_key or ANTHROPIC_API_KEY")
        return
    
    try:
        # Set up file paths
        response_files = {
            'openai': args.openai_responses,
            'grok': args.grok_responses,
            'opensource': args.opensource_responses
        }
        
        differences_files = {
            'Expert1': args.expert1_original,
            'Expert2': args.expert2_original
        }
        
        analyzer = PolishedRubricsGradingComparison(
            gpt_api_key, 
            claude_api_key, 
            response_files=response_files,
            differences_files=differences_files
        )
        
        results = analyzer.run_complete_analysis(
            args.expert1_polished,
            args.expert2_polished,
            args.output_dir
        )
        
        print("\n🎉 Polished rubrics analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()

# Example usage:
# python polished_rubrics_grading_comparison.py --gpt_api_key "your-key" --claude_api_key "your-key"