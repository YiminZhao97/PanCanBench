#!/usr/bin/env python3
"""
Comprehensive Analysis Workflow for Grader Model Permutation Study

This script runs the complete analysis pipeline for any fold:
1. Grading consistency analysis between GPT-4.1 and Claude graders
2. Identification of large grading differences (20+ percentage points)
3. Generation of AI prompts for rubric polishing
4. AI-powered rubric polishing using OpenAI
5. Comparison of grading improvements after polishing

Usage:
    python analysis_workflow.py --fold 3 --gpt_api_key "your-key" --claude_api_key "your-key"
    
Prerequisites:
    - claude_grade_fold{N}.sh and gpt_grade_fold{N}.sh must be completed first
    - All input data files must be present in expected locations
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# Add the main project directory to path
sys.path.append('/home/yzhao4/PanCan-QA_LLM/rubrics_collection/Analysis4_permute_grader_model')


class AnalysisWorkflow:
    """Orchestrates the complete analysis workflow for any fold."""
    
    def __init__(self, fold_number: int, gpt_api_key: str, claude_api_key: str, 
                 output_base_dir: str = None, expert1_rubric: str = None, 
                 expert2_rubric: str = None, scores_base_path: str = None,
                 scripts_base_path: str = None):
        """
        Initialize the workflow manager.
        
        Args:
            fold_number: The fold number to process (e.g., 3 for fold3)
            gpt_api_key: OpenAI API key for rubric polishing
            claude_api_key: Anthropic API key for grading comparison
            output_base_dir: Base directory for outputs (default: current directory)
            expert1_rubric: Explicit path to Expert1 rubric file (default: Fold{N}-Jesse.json)
            expert2_rubric: Explicit path to Expert2 rubric file (default: Fold{N}-Simone.json)
            scores_base_path: Base path for grading score files (default: project directory)
            scripts_base_path: Base path for Python scripts (default: project directory)
        """
        self.fold = fold_number
        self.gpt_api_key = gpt_api_key
        self.claude_api_key = claude_api_key
        self.output_base_dir = output_base_dir or f"./fold{fold_number}_analysis_results"
        
        # Set default paths if not provided
        default_base_path = "/Users/zhaoyimin/Desktop/PanCan QA/rubrics collection/Analysis4 permute grader model"
        self.scores_base_path = scores_base_path or default_base_path
        self.scripts_base_path = scripts_base_path or default_base_path
        self.expert1_rubric = expert1_rubric or f"{default_base_path}/Fold{fold_number}-Jesse.json"
        self.expert2_rubric = expert2_rubric or f"{default_base_path}/Fold{fold_number}-Simone.json"
        
        # Create output directory structure
        self.setup_output_directories()
        
        # Define file paths based on fold number
        self.setup_file_paths()
        
        print(f"🚀 Initialized Analysis Workflow for Fold {fold_number}")
        print(f"📁 Output directory: {self.output_base_dir}")
        print(f"📋 Expert1 rubric: {self.expert1_rubric}")
        print(f"📋 Expert2 rubric: {self.expert2_rubric}")
        print(f"📊 Scores base path: {self.scores_base_path}")
        print(f"🐍 Scripts base path: {self.scripts_base_path}")
    
    def setup_output_directories(self):
        """Create the output directory structure."""
        dirs_to_create = [
            self.output_base_dir,
            f"{self.output_base_dir}/consistency_analysis",
            f"{self.output_base_dir}/large_differences", 
            f"{self.output_base_dir}/rubric_polishing",
            f"{self.output_base_dir}/polished_grading_comparison"
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
    
    def setup_file_paths(self):
        """Set up all required file paths based on fold number."""
        
        self.input_files = {
            # Grading results (use scores_base_path)
            'gpt_expert1': f"{self.scores_base_path}/fold{self.fold}_gpt_scores_expert1.json",
            'gpt_expert2': f"{self.scores_base_path}/fold{self.fold}_gpt_scores_expert2.json", 
            'claude_expert1': f"{self.scores_base_path}/fold{self.fold}_claude_scores_expert1.json",
            'claude_expert2': f"{self.scores_base_path}/fold{self.fold}_claude_scores_expert2.json",
            
            # Rubric files (use explicit paths)
            'expert1_rubric': self.expert1_rubric,
            'expert2_rubric': self.expert2_rubric,
            
            # Response files (keep default paths for now)
            'openai_responses': "/home/yzhao4/PanCan-QA_LLM/data_share/openai_family_response.jsonl",
            'grok_responses': "/home/yzhao4/PanCan-QA_LLM/data_share/grok_response.jsonl", 
            'opensource_responses': "/home/yzhao4/PanCan-QA_LLM/data_share/all_opensource_models_final_outputs_temp0.7.jsonl"
        }
        
        # Output files
        self.output_files = {
            # Step 2: Large differences
            'expert1_differences_data': f"{self.output_base_dir}/large_differences/fold{self.fold}_large_differences_expert1_data.json",
            'expert1_differences_report': f"{self.output_base_dir}/large_differences/fold{self.fold}_large_differences_expert1_report.md",
            'expert2_differences_data': f"{self.output_base_dir}/large_differences/fold{self.fold}_large_differences_expert2_data.json", 
            'expert2_differences_report': f"{self.output_base_dir}/large_differences/fold{self.fold}_large_differences_expert2_report.md",
            
            # Step 3: Polishing prompts
            'expert1_prompts_json': f"{self.output_base_dir}/rubric_polishing/rubric_polishing_prompts_expert1.json",
            'expert1_prompts_md': f"{self.output_base_dir}/rubric_polishing/rubric_polishing_prompts_expert1.md",
            'expert2_prompts_json': f"{self.output_base_dir}/rubric_polishing/rubric_polishing_prompts_expert2.json",
            'expert2_prompts_md': f"{self.output_base_dir}/rubric_polishing/rubric_polishing_prompts_expert2.md",
            
            # Step 4: Polished rubrics
            'expert1_polished': f"{self.output_base_dir}/rubric_polishing/polished_rubrics_expert1.json",
            'expert2_polished': f"{self.output_base_dir}/rubric_polishing/polished_rubrics_expert2.json",
            'expert1_detailed': f"{self.output_base_dir}/rubric_polishing/detailed_rubric_polished_expert1.json",
            'expert2_detailed': f"{self.output_base_dir}/rubric_polishing/detailed_rubric_polished_expert2.json",
            'expert1_report': f"{self.output_base_dir}/rubric_polishing/report_rubric_polished_expert1.md",
            'expert2_report': f"{self.output_base_dir}/rubric_polishing/report_rubric_polished_expert2.md"
        }
        
        # Python script paths
        self.scripts = {
            'consistency_analysis': f"{self.scripts_base_path}/complete_grader_consistency_analysis.py",
            'large_differences': f"{self.scripts_base_path}/targeted_large_differences.py", 
            'polishing_generator': f"{self.scripts_base_path}/rubric_polishing_generator.py",
            'rubric_polisher': f"{self.scripts_base_path}/rubric_polisher.py",
            'polished_comparison': f"{self.scripts_base_path}/polished_rubrics_grading_comparison.py"
        }
    
    def check_prerequisites(self):
        """Check that all required input files exist."""
        print("\n🔍 Checking prerequisites...")
        missing_files = []
        
        for file_type, file_path in self.input_files.items():
            if not os.path.exists(file_path):
                missing_files.append(f"  ❌ {file_type}: {file_path}")
            else:
                print(f"  ✅ {file_type}: Found")
        
        if missing_files:
            print("\n❌ Missing required files:")
            for missing in missing_files:
                print(missing)
            print("\nPlease ensure grading is complete and all input files are available.")
            return False
        
        print("✅ All prerequisites satisfied!")
        return True
    
    def run_step(self, step_name: str, command: list, success_message: str):
        """Run a single workflow step with error handling."""
        print(f"\n{'='*60}")
        print(f"🔄 Step: {step_name}")
        print(f"{'='*60}")
        
        try:
            # Run the command
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"✅ {success_message}")
            
            # Print any output
            if result.stdout:
                print("Output:")
                print(result.stdout)
                
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error in {step_name}:")
            print(f"Return code: {e.returncode}")
            if e.stdout:
                print("STDOUT:")
                print(e.stdout)
            if e.stderr:
                print("STDERR:")
                print(e.stderr)
            return False
        except Exception as e:
            print(f"❌ Unexpected error in {step_name}: {e}")
            return False
    
    def step1_consistency_analysis(self):
        """Step 1: Run grading consistency analysis."""
        # Note: This script analyzes one expert at a time, so we'll run it twice
        
        # Run for Expert1
        command_expert1 = [
            'python', self.scripts['consistency_analysis'],
            '--gpt_grader_file', self.input_files['gpt_expert1'],
            '--claude_grader_file', self.input_files['claude_expert1'],
            '--output_dir', f"{self.output_base_dir}/consistency_analysis",
            '--output_prefix', f'expert1_fold{self.fold}'
        ]
        
        success1 = self.run_step(
            "Grading Consistency Analysis - Expert1",
            command_expert1,
            "Expert1 consistency analysis completed successfully!"
        )
        
        if not success1:
            return False
        
        # Run for Expert2
        command_expert2 = [
            'python', self.scripts['consistency_analysis'],
            '--gpt_grader_file', self.input_files['gpt_expert2'],
            '--claude_grader_file', self.input_files['claude_expert2'],
            '--output_dir', f"{self.output_base_dir}/consistency_analysis",
            '--output_prefix', f'expert2_fold{self.fold}'
        ]
        
        return self.run_step(
            "Grading Consistency Analysis - Expert2",
            command_expert2,
            "Expert2 consistency analysis completed successfully!"
        )
    
    def step2_large_differences(self):
        """Step 2: Identify large grading differences."""
        command = [
            'python', self.scripts['large_differences'],
            '--gpt_expert1', self.input_files['gpt_expert1'],
            '--gpt_expert2', self.input_files['gpt_expert2'],
            '--claude_expert1', self.input_files['claude_expert1'],
            '--claude_expert2', self.input_files['claude_expert2'],
            '--expert1_rubric', self.input_files['expert1_rubric'],
            '--expert2_rubric', self.input_files['expert2_rubric'],
            '--openai_responses', self.input_files['openai_responses'],
            '--grok_responses', self.input_files['grok_responses'],
            '--opensource_responses', self.input_files['opensource_responses'],
            '--fold', str(self.fold),
            '--output_dir', f"{self.output_base_dir}/large_differences"
        ]
        
        return self.run_step(
            "Large Differences Identification",
            command,
            "Large differences analysis completed successfully!"
        )
    
    def step3_generate_polishing_prompts(self):
        """Step 3: Generate AI prompts for rubric polishing."""
        command = [
            'python', self.scripts['polishing_generator'],
            '--expert_files', 
            self.output_files['expert1_differences_data'],
            self.output_files['expert2_differences_data'],
            '--output_dir', f"{self.output_base_dir}/rubric_polishing"
        ]
        
        return self.run_step(
            "Polishing Prompts Generation",
            command,
            "Polishing prompts generated successfully!"
        )
    
    def step4_polish_rubrics(self):
        """Step 4: Use AI to polish the rubrics."""
        # Note: This script processes one expert at a time, so we'll run it twice
        
        # Process Expert1
        command_expert1 = [
            'python', self.scripts['rubric_polisher'],
            '--api_key', self.gpt_api_key,
            '--input_files', self.output_files['expert1_prompts_json'],
            '--output_dir', f"{self.output_base_dir}/rubric_polishing"
        ]
        
        success1 = self.run_step(
            "AI Rubric Polishing - Expert1",
            command_expert1,
            "Expert1 rubric polishing completed successfully!"
        )
        
        if not success1:
            return False
        
        # Process Expert2
        command_expert2 = [
            'python', self.scripts['rubric_polisher'],
            '--api_key', self.gpt_api_key,
            '--input_files', self.output_files['expert2_prompts_json'],
            '--output_dir', f"{self.output_base_dir}/rubric_polishing"
        ]
        
        return self.run_step(
            "AI Rubric Polishing - Expert2",
            command_expert2,
            "Expert2 rubric polishing completed successfully!"
        )
    
    def step5_compare_improvements(self):
        """Step 5: Compare grading improvements after polishing."""
        command = [
            'python', self.scripts['polished_comparison'],
            '--gpt_api_key', self.gpt_api_key,
            '--claude_api_key', self.claude_api_key,
            '--expert1_polished', self.output_files['expert1_polished'],
            '--expert2_polished', self.output_files['expert2_polished'],
            '--expert1_original', self.output_files['expert1_differences_data'],
            '--expert2_original', self.output_files['expert2_differences_data'],
            '--output_dir', f"{self.output_base_dir}/polished_grading_comparison"
        ]
        
        return self.run_step(
            "Polished Rubrics Grading Comparison",
            command,
            "Grading comparison completed successfully!"
        )
    
    def run_complete_workflow(self):
        """Run the complete analysis workflow."""
        start_time = datetime.now()
        
        print(f"🚀 Starting Complete Analysis Workflow for Fold {self.fold}")
        print(f"📅 Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check prerequisites
        if not self.check_prerequisites():
            return False
        
        # Define workflow steps
        steps = [
            ("Consistency Analysis", self.step1_consistency_analysis),
            ("Large Differences Identification", self.step2_large_differences),
            ("Generate Polishing Prompts", self.step3_generate_polishing_prompts),
            ("AI Rubric Polishing", self.step4_polish_rubrics),
            ("Compare Improvements", self.step5_compare_improvements)
        ]
        
        # Execute steps
        completed_steps = 0
        for step_name, step_function in steps:
            if step_function():
                completed_steps += 1
                print(f"✅ {step_name} completed ({completed_steps}/{len(steps)})")
            else:
                print(f"❌ {step_name} failed. Stopping workflow.")
                break
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n{'='*60}")
        print(f"📊 WORKFLOW SUMMARY")
        print(f"{'='*60}")
        print(f"Fold: {self.fold}")
        print(f"Completed Steps: {completed_steps}/{len(steps)}")
        print(f"Duration: {duration}")
        print(f"Output Directory: {self.output_base_dir}")
        
        if completed_steps == len(steps):
            print("🎉 Complete workflow finished successfully!")
            print("\n📁 Generated Files:")
            for file_type, file_path in self.output_files.items():
                if os.path.exists(file_path):
                    print(f"  ✅ {file_type}: {file_path}")
                else:
                    print(f"  ❌ {file_type}: {file_path} (not found)")
            return True
        else:
            print("❌ Workflow completed with errors.")
            return False
    
    def run_individual_steps(self, steps: list):
        """Run individual workflow steps."""
        start_time = datetime.now()
        
        print(f"🚀 Starting Selected Steps for Fold {self.fold}")
        print(f"📅 Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Selected steps: {', '.join(steps)}")
        
        # Check prerequisites
        if not self.check_prerequisites():
            return False
        
        # Define step mapping
        step_functions = {
            'consistency': ("Consistency Analysis", self.step1_consistency_analysis),
            'differences': ("Large Differences Identification", self.step2_large_differences),
            'prompts': ("Generate Polishing Prompts", self.step3_generate_polishing_prompts),
            'polish': ("AI Rubric Polishing", self.step4_polish_rubrics),
            'compare': ("Compare Improvements", self.step5_compare_improvements)
        }
        
        # Execute selected steps
        completed_steps = 0
        total_steps = len(steps)
        
        for step_key in steps:
            if step_key in step_functions:
                step_name, step_function = step_functions[step_key]
                if step_function():
                    completed_steps += 1
                    print(f"✅ {step_name} completed ({completed_steps}/{total_steps})")
                else:
                    print(f"❌ {step_name} failed. Stopping workflow.")
                    break
            else:
                print(f"❌ Unknown step: {step_key}")
                print(f"Valid steps: {list(step_functions.keys())}")
                break
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n{'='*60}")
        print(f"📊 INDIVIDUAL STEPS SUMMARY")
        print(f"{'='*60}")
        print(f"Fold: {self.fold}")
        print(f"Completed Steps: {completed_steps}/{total_steps}")
        print(f"Duration: {duration}")
        print(f"Output Directory: {self.output_base_dir}")
        
        if completed_steps == total_steps:
            print("🎉 Selected steps completed successfully!")
            return True
        else:
            print("❌ Some steps completed with errors.")
            return False
    
    def generate_summary_report(self):
        """Generate a summary report of the entire analysis."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# Analysis Workflow Summary - Fold {self.fold}

**Generated:** {timestamp}
**Fold:** {self.fold}
**Output Directory:** {self.output_base_dir}

## Workflow Steps Completed

1. **Grading Consistency Analysis** ✅
   - Analyzed consistency between GPT-4.1 and Claude graders
   - Target models: gpt-4o, grok-4-latest, meta-llama_Llama-3.1-70B-Instruct

2. **Large Differences Identification** ✅
   - Identified cases with 20+ percentage point differences
   - Generated detailed analysis reports for Expert1 and Expert2

3. **Polishing Prompts Generation** ✅
   - Created AI prompts for rubric improvement
   - Based on identified grading disagreements

4. **AI Rubric Polishing** ✅
   - Used OpenAI GPT-4o to polish rubrics
   - Enforced binary scoring to reduce ambiguity

5. **Grading Improvements Comparison** ✅
   - Re-graded with polished rubrics
   - Measured improvement in grader consistency

## Key Outputs

### Large Differences Analysis
- Expert1 Data: `{self.output_files['expert1_differences_data']}`
- Expert1 Report: `{self.output_files['expert1_differences_report']}`
- Expert2 Data: `{self.output_files['expert2_differences_data']}`
- Expert2 Report: `{self.output_files['expert2_differences_report']}`

### Rubric Polishing
- Expert1 Polished Rubrics (Standard): `{self.output_files['expert1_polished']}`
- Expert1 Detailed Analysis: `{self.output_files['expert1_detailed']}`
- Expert1 Report: `{self.output_files['expert1_report']}`
- Expert2 Polished Rubrics (Standard): `{self.output_files['expert2_polished']}`
- Expert2 Detailed Analysis: `{self.output_files['expert2_detailed']}`
- Expert2 Report: `{self.output_files['expert2_report']}`

### Improvement Analysis
- Located in: `{self.output_base_dir}/polished_grading_comparison/`

## Next Steps

1. Review the generated reports to understand grading inconsistencies
2. Examine polished rubrics for quality and appropriateness
3. Analyze improvement metrics to assess effectiveness
4. Consider deploying polished rubrics for production grading

---

*Generated by Analysis Workflow v1.0 on {timestamp}*
"""
        
        summary_file = f"{self.output_base_dir}/fold{self.fold}_workflow_summary.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 Summary report saved: {summary_file}")

def main():
    """Main function to run the analysis workflow."""
    
    parser = argparse.ArgumentParser(description='Complete Analysis Workflow for Grader Model Permutation Study')
    parser.add_argument('--fold', type=int, required=True,
                       help='Fold number to process (e.g., 3 for fold3)')
    parser.add_argument('--gpt_api_key', type=str, default=None,
                       help='OpenAI API key for rubric polishing (or set OPENAI_API_KEY env var)')
    parser.add_argument('--claude_api_key', type=str, default=None,
                       help='Anthropic API key for grading comparison (or set ANTHROPIC_API_KEY env var)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Base output directory (default: ./fold{N}_analysis_results)')
    parser.add_argument('--expert1_rubric', type=str, default=None,
                       help='Path to Expert1 rubric file (default: Fold{N}-Jesse.json)')
    parser.add_argument('--expert2_rubric', type=str, default=None,
                       help='Path to Expert2 rubric file (default: Fold{N}-Simone.json)')
    parser.add_argument('--scores_base_path', type=str, default=None,
                       help='Base path for grading score files (default: project directory)')
    parser.add_argument('--scripts_base_path', type=str, default=None,
                       help='Base path for Python scripts (default: project directory)')
    parser.add_argument('--steps', type=str, nargs='+', 
                       choices=['consistency', 'differences', 'prompts', 'polish', 'compare', 'all'],
                       default=['all'],
                       help='Which steps to run (default: all)')
    
    args = parser.parse_args()
    
    # Get API keys from args or environment
    gpt_api_key = args.gpt_api_key or os.getenv('OPENAI_API_KEY')
    claude_api_key = args.claude_api_key or os.getenv('ANTHROPIC_API_KEY')
    
    if not gpt_api_key:
        print("❌ OpenAI API key required. Set via --gpt_api_key or OPENAI_API_KEY env var")
        return
    
    if not claude_api_key:
        print("❌ Anthropic API key required. Set via --claude_api_key or ANTHROPIC_API_KEY env var")
        return
    
    try:
        # Initialize workflow
        workflow = AnalysisWorkflow(
            fold_number=args.fold,
            gpt_api_key=gpt_api_key,
            claude_api_key=claude_api_key,
            output_base_dir=args.output_dir,
            expert1_rubric=args.expert1_rubric,
            expert2_rubric=args.expert2_rubric,
            scores_base_path=args.scores_base_path,
            scripts_base_path=args.scripts_base_path
        )
        
        # Run workflow
        if 'all' in args.steps:
            success = workflow.run_complete_workflow()
        else:
            # Run individual steps
            success = workflow.run_individual_steps(args.steps)
        
        # Generate summary
        if success:
            workflow.generate_summary_report()
            print(f"\n🎉 Analysis workflow for Fold {args.fold} completed successfully!")
        else:
            print(f"\n❌ Analysis workflow for Fold {args.fold} completed with errors.")
            
    except Exception as e:
        print(f"❌ Workflow failed with error: {e}")
        raise

if __name__ == "__main__":
    main()

# Example usage:
# python analysis_workflow.py --fold 3 --gpt_api_key "your-openai-key" --claude_api_key "your-claude-key"
# python analysis_workflow.py --fold 4 --output_dir "./fold4_results"