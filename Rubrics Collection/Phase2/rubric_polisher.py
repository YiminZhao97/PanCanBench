#!/usr/bin/env python3
"""
AI-powered rubric polisher that takes polishing prompts and generates improved rubrics.
Uses OpenAI API to polish rubrics based on grading disagreements.
"""

import json
import os
import re
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
import time
import openai
# You'll need to install openai: pip install openai

class RubricPolisher:
    """AI-powered rubric polishing system using OpenAI."""
    
    def __init__(self, model="gpt-4o", api_key=None):
        """
        Initialize the rubric polisher.
        
        Args:
            model: OpenAI model name (e.g., "gpt-4o", "gpt-4-turbo")
            api_key: API key (if None, will use OPENAI_API_KEY environment variable)
        """
        self.model = model
        
        # Initialize OpenAI client
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            # Will use OPENAI_API_KEY environment variable
            self.client = openai.OpenAI()
        
        print(f"Initialized RubricPolisher with {self.model}")
    
    def call_openai(self, prompt: str, max_retries: int = 3) -> str:
        """Call OpenAI API with retry logic."""
        instructions = """
            You are an expert medical educator and assessment specialist. Your task is to polish medical rubrics to reduce grading ambiguity and enforce 
            binary scoring. Always follow the exact format requested in the deliverables.
            """
        for attempt in range(max_retries):
            try:
                # Set up request parameters
                request_params = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": instructions},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                }
                
                # Handle different model types
                if any(self.model.startswith(prefix) for prefix in ['o3', 'o1']):
                    request_params["max_completion_tokens"] = 3000
                else:
                    request_params["max_tokens"] = 3000
                
                # Create completion
                response = self.client.chat.completions.create(**request_params)
                return response.choices[0].message.content
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise e
    
    def _parse_table_content(self, content: str, result: Dict[str, Any]) -> None:
        """Parse table content to extract rubric items."""
        try:
            # Split content into lines and process each line
            lines = content.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                
                # Skip empty lines and markdown table separators
                if not line or line.startswith('|---') or line.startswith('|-'):
                    continue
                
                # Look for table rows with | delimiters
                if '|' in line:
                    # Split by | and clean each part
                    parts = [part.strip() for part in line.split('|')]
                    
                    # Remove empty first/last parts (common in markdown tables)
                    parts = [part for part in parts if part]
                    
                    # We need at least 3 parts: item_id, description, points
                    if len(parts) >= 3:
                        item_id = parts[0].strip()
                        description = parts[1].strip()
                        points_str = parts[2].strip()
                        
                        # Skip header rows or placeholder rows
                        if (item_id == '#' or 
                            'item sentence' in description.lower() or 
                            'must be explicitly satisfied' in description.lower() or
                            item_id.lower() == 'item id'):
                            continue
                        
                        # Try to extract points as integer
                        try:
                            if points_str.isdigit():
                                points = int(points_str)
                            else:
                                # Look for numbers in the points field
                                import re
                                point_match = re.search(r'\d+', points_str)
                                if point_match:
                                    points = int(point_match.group())
                                else:
                                    continue  # Skip if no valid points found
                        except (ValueError, AttributeError):
                            continue
                        
                        # Add the parsed item
                        result['revised_rubric'].append({
                            'item_id': item_id,
                            'description': description,
                            'points': points
                        })
                        
        except Exception as e:
            result['parsing_errors'].append(f"Table parsing error: {e}")

    def parse_polished_rubric(self, ai_response: str, question_id: str) -> Dict[str, Any]:
        """Parse AI response to extract structured rubric data."""
        
        # Initialize result structure
        result = {
            'question_id': question_id,
            'revised_rubric': [],
            'grader_spec': '',
            'change_log': '',
            'raw_response': ai_response,
            'parsing_errors': []
        }
        
        try:
            # Enhanced table patterns to handle different formats including Q103's format
            table_patterns = [
                # Standard format: | # | Rubric item | Points |
                r'\|.*?#.*?\|.*?Rubric item.*?\|.*?Points.*?\|.*?\n((?:\|.*?\|.*?\|.*?\|\n?)*)',
                # Standard format: | Item ID | Rubric Item | Points |
                r'\|.*?Item ID.*?\|.*?Rubric Item.*?\|.*?Points.*?\|.*?\n((?:\|.*?\|.*?\|.*?\|\n?)*)',
                # Q103 format: | # | Item (binary; must meet exact condition to pass) | Points |
                r'\|.*?#.*?\|.*?Item.*?\(.*?binary.*?\).*?\|.*?Points.*?\|.*?\n((?:\|.*?\|.*?\|.*?\|\n?)*)',
                # Generic Item format: | # | Item | Points |
                r'\|.*?#.*?\|.*?Item.*?\|.*?Points.*?\|.*?\n((?:\|.*?\|.*?\|.*?\|\n?)*)',
                # Section-based patterns
                r'## ?1\).*?revised rubric.*?\n(.*?)(?=## ?2\)|$)',
                r'1\).*?Revised Rubric.*?\n\n(.*?)(?=2\)|$)',
                # Look for any table structure in the response
                r'(\|.*?\|.*?\|.*?\|(?:\n\|.*?\|.*?\|.*?\|)*)',
            ]
            
            table_found = False
            for pattern in table_patterns:
                table_match = re.search(pattern, ai_response, re.DOTALL | re.IGNORECASE)
                if table_match:
                    table_content = table_match.group(1).strip()
                    self._parse_table_content(table_content, result)
                    if result['revised_rubric']:
                        table_found = True
                        break
            
            # If no table found with patterns, try parsing the entire response
            if not table_found:
                self._parse_table_content(ai_response, result)
            
            # Validate that we got essential data
            if not result['revised_rubric']:
                result['parsing_errors'].append("No revised rubric table found")
            
        except Exception as e:
            result['parsing_errors'].append(f"General parsing error: {e}")
        
        return result
    
    def polish_single_rubric(self, question_id: str, prompt_data: Dict[str, Any]) -> Dict[str, Any]:
        """Polish a single rubric using OpenAI."""
        
        question_number = prompt_data['question_number']
        prompt = prompt_data['prompt']
        
        print(f"  Polishing Q{question_number}...")
        
        try:
            # Call OpenAI to polish the rubric
            ai_response = self.call_openai(prompt)
            
            # Parse the response
            parsed_result = self.parse_polished_rubric(ai_response, question_id)
            
            # Add metadata
            parsed_result.update({
                'original_question': prompt_data['question_text'],
                'models_with_differences': prompt_data['models_with_differences'],
                'original_rubric_items': prompt_data['original_rubric_items'],
                'polished_at': datetime.now().isoformat(),
                'ai_model': self.model,
                'success': len(parsed_result['parsing_errors']) == 0
            })
            
            return parsed_result
            
        except Exception as e:
            print(f"    Error polishing Q{question_number}: {e}")
            return {
                'question_id': question_id,
                'error': str(e),
                'original_question': prompt_data['question_text'],
                'polished_at': datetime.now().isoformat(),
                'ai_model': self.model,
                'success': False
            }
    
    def polish_rubrics_from_file(self, input_file: str, output_prefix: str = None) -> Dict[str, Any]:
        """Polish all rubrics from a prompt file."""
        
        # Load input data
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading input file {input_file}: {e}")
        
        if 'polishing_prompts' not in data:
            raise ValueError("Input file must contain 'polishing_prompts' key")
        
        expert_name = data.get('metadata', {}).get('expert', 'Unknown')
        prompts = data['polishing_prompts']
        
        print(f"Polishing {len(prompts)} rubrics for {expert_name}...")
        
        # Process each prompt
        polished_results = {}
        total_prompts = len(prompts)
        
        for i, (question_id, prompt_data) in enumerate(prompts.items(), 1):
            print(f"Progress: {i}/{total_prompts}")
            result = self.polish_single_rubric(question_id, prompt_data)
            polished_results[question_id] = result
            
            # Brief pause to be respectful to API
            time.sleep(1)
        
        # Create output structure
        output_data = {
            'metadata': {
                'expert': expert_name,
                'polished_at': datetime.now().isoformat(),
                'ai_model': self.model,
                'total_questions': len(prompts),
                'successful_polishes': len([r for r in polished_results.values() if r.get('success', False)]),
                'failed_polishes': len([r for r in polished_results.values() if not r.get('success', False)]),
                'source_file': input_file
            },
            'polished_rubrics': polished_results
        }
        
        # Generate output files
        if not output_prefix:
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_prefix = base_name.replace('polishing_prompts', 'polished')
        
        # Save JSON results with detailed prefix
        output_dir = os.path.dirname(output_prefix)
        base_name = os.path.basename(output_prefix)
        json_output = os.path.join(output_dir, f"detailed_{base_name}.json")
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Saved detailed polished rubrics: {json_output}")
        
        # Generate markdown report with report prefix
        md_output = os.path.join(output_dir, f"report_{base_name}.md")
        markdown_content = self.generate_markdown_report(output_data)
        with open(md_output, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        print(f"Saved markdown report: {md_output}")
        
        return output_data
    
    def generate_markdown_report(self, output_data: Dict[str, Any]) -> str:
        """Generate a markdown report of polished rubrics."""
        
        metadata = output_data['metadata']
        results = output_data['polished_rubrics']
        
        markdown_content = f"""# Polished Rubrics - {metadata['expert']}

**Polished:** {metadata['polished_at'][:19]}
**AI Model:** {metadata['ai_model']}
**Total Questions:** {metadata['total_questions']}
**Successful:** {metadata['successful_polishes']}
**Failed:** {metadata['failed_polishes']}

---

"""
        
        # Process each polished rubric
        for question_id, result in results.items():
            if 'error' in result:
                markdown_content += f"""## {question_id} ❌ FAILED

**Question:** {result.get('original_question', 'Unknown')}
**Error:** {result['error']}

---

"""
                continue
            
            markdown_content += f"""## {question_id}

**Question:** {result['original_question']}
**Models with Differences:** {', '.join(result['models_with_differences'])}

### Revised Rubric (Binary)

| Item ID | Rubric Item | Points |
|---------|-------------|--------|
"""
            
            total_points = 0
            for item in result.get('revised_rubric', []):
                markdown_content += f"| {item['item_id']} | {item['description']} | {item['points']} |\n"
                total_points += item['points']
            
            markdown_content += f"\n**Total Points:** {total_points}\n\n"
            
            # Show parsing errors if any
            if result.get('parsing_errors'):
                markdown_content += f"""### ⚠️ Parsing Issues

"""
                for error in result['parsing_errors']:
                    markdown_content += f"- {error}\n"
                markdown_content += "\n"
            
            markdown_content += "---\n\n"
        
        # Add summary
        if metadata['failed_polishes'] > 0:
            markdown_content += f"""## Summary

- ✅ Successfully polished: {metadata['successful_polishes']} rubrics
- ❌ Failed to polish: {metadata['failed_polishes']} rubrics

### Failed Questions
"""
            for question_id, result in results.items():
                if 'error' in result:
                    markdown_content += f"- {question_id}: {result['error']}\n"
        
        markdown_content += f"""

---

*Generated on {metadata['polished_at'][:19]}*
*Polished using {metadata['ai_model']}*
"""
        
        return markdown_content
    
    def export_polished_rubrics_to_standard_format(self, polished_data: Dict[str, Any], output_file: str = None) -> str:
        """
        Export polished rubrics to the standard rubrics JSON format used in the project.
        
        Args:
            polished_data: The output from polish_rubrics_from_file()
            output_file: Optional output file path. If None, generates based on expert name.
            
        Returns:
            The path to the generated file
        """
        
        expert_name = polished_data['metadata']['expert']
        polished_rubrics = polished_data['polished_rubrics']
        
        # Generate output filename if not provided
        if not output_file:
            output_file = f"polished_rubrics_{expert_name.lower()}.json"
        
        # Convert to standard format
        standard_format = {
            "questions": []
        }
        
        for question_id, rubric_result in polished_rubrics.items():
            if not rubric_result.get('success', False):
                print(f"  ⚠️  Skipping {question_id} - polishing failed")
                continue
                
            # Extract question number from question_id (e.g., "Q60" -> 60)
            try:
                question_number = int(question_id[1:])  # Remove 'Q' prefix
            except (ValueError, IndexError):
                print(f"  ⚠️  Could not parse question number from {question_id}")
                continue
            
            question_entry = {
                "question_number": question_number,
                "question_text": rubric_result.get('original_question', ''),
                "rubric_items": []
            }
            
            # Convert revised rubric items to standard format
            for item in rubric_result.get('revised_rubric', []):
                try:
                    item_number = int(item['item_id'])
                except (ValueError, KeyError):
                    print(f"  ⚠️  Could not parse item number from {item.get('item_id', 'unknown')}")
                    continue
                
                rubric_item = {
                    "item_number": item_number,
                    "description": item['description'],
                    "min_points": 0,  # Binary scoring - either 0 or max points
                    "max_points": item['points']
                }
                
                question_entry["rubric_items"].append(rubric_item)
            
            # Sort rubric items by item number
            question_entry["rubric_items"].sort(key=lambda x: x["item_number"])
            
            standard_format["questions"].append(question_entry)
        
        # Sort questions by question number
        standard_format["questions"].sort(key=lambda x: x["question_number"])
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(standard_format, f, indent=2, ensure_ascii=False)
        
        print(f"Exported {len(standard_format['questions'])} polished rubrics to: {output_file}")
        return output_file

def main():
    """Main function to process rubric polishing."""
    
    parser = argparse.ArgumentParser(description='AI-powered rubric polisher')
    parser.add_argument('--model', type=str, default='gpt-4o',
                       help='OpenAI model to use (default: gpt-4o)')
    parser.add_argument('--api_key', type=str, default=None,
                       help='OpenAI API key (if not set as environment variable)')
    parser.add_argument('--input_files', type=str, nargs='+', required=True,
                       help='Input JSON files with polishing prompts')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results (default: current directory)')
    
    args = parser.parse_args()
    
    print("AI-Powered Rubric Polisher")
    print("=" * 40)
    
    # API key should be set as environment variable:
    # export OPENAI_API_KEY="your-key-here"
    # Or you can pass it directly with --api_key
    
    try:
        # Initialize polisher
        polisher = RubricPolisher(model=args.model, api_key=args.api_key)
        
        # Process specified input files
        for input_file in args.input_files:
            if os.path.exists(input_file):
                print(f"\nProcessing {input_file}...")
                try:
                    # Generate output prefix 
                    base_name = os.path.splitext(os.path.basename(input_file))[0]
                    output_prefix = base_name.replace('polishing_prompts', 'polished')
                    
                    if args.output_dir:
                        os.makedirs(args.output_dir, exist_ok=True)
                        output_prefix = os.path.join(args.output_dir, output_prefix)
                    
                    result = polisher.polish_rubrics_from_file(input_file, output_prefix)
                    
                    # Print summary
                    metadata = result['metadata']
                    print(f"  ✅ Expert: {metadata['expert']}")
                    print(f"  ✅ Successful: {metadata['successful_polishes']}/{metadata['total_questions']}")
                    if metadata['failed_polishes'] > 0:
                        print(f"  ❌ Failed: {metadata['failed_polishes']}")
                    
                    # Generate standard format export
                    try:
                        # Construct output file path in the same directory as other outputs
                        expert_name = result['metadata']['expert']
                        standard_output_file = os.path.join(args.output_dir, f"polished_rubrics_{expert_name.lower()}.json") if args.output_dir else f"polished_rubrics_{expert_name.lower()}.json"
                        standard_file = polisher.export_polished_rubrics_to_standard_format(result, standard_output_file)
                        print(f"  ✅ Standard format: {standard_file}")
                    except Exception as e:
                        print(f"  ⚠️  Error generating standard format: {e}")
                    
                except Exception as e:
                    print(f"  ❌ Error processing {input_file}: {e}")
            else:
                print(f"  ⚠️  File not found: {input_file}")
        
        print(f"\n{'='*40}")
        print("Rubric polishing complete!")
        print("\nNext steps:")
        print("1. Review the generated markdown reports")
        print("2. Validate the polished rubrics")
        print("3. Implement the improved rubrics in your grading system")
        
    except Exception as e:
        print(f"Error initializing polisher: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have OPENAI_API_KEY set as environment variable")
        print("2. Install required packages: pip install openai")
        print("3. Check your internet connection")

if __name__ == "__main__":
    main()