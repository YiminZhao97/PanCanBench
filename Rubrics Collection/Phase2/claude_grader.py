#!/usr/bin/env python3
"""
Claude-4 Automated Grader for Pancreatic Cancer Q&A Responses
Uses Anthropic Claude-4 to grade responses against provided rubrics
"""

import json
import anthropic
import os
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

class Claude4Grader:
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        """
        Initialize the Claude-4 grader
        
        Args:
            api_key (str): Anthropic API key
            model (str): Model to use (default: claude-3-5-sonnet-20241022)
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.grading_history = []
    
    def load_qa_responses(self, qa_file_path: str) -> List[Dict]:
        """Load Q&A responses from JSON file"""
        try:
            with open(qa_file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except Exception as e:
            print(f"Error loading Q&A file: {e}")
            return []
    
    def load_rubrics(self, rubrics_file_path: str) -> Dict:
        """Load rubrics from JSON file"""
        try:
            with open(rubrics_file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except Exception as e:
            print(f"Error loading rubrics file: {e}")
            return {}
    
    def create_grading_prompt(self, question: str, response: str, rubric_items: List[Dict]) -> str:
        """
        Create a detailed grading prompt for Claude-4
        
        Args:
            question (str): The original question
            response (str): The response to be graded
            rubric_items (List[Dict]): List of rubric criteria
            
        Returns:
            str: Formatted grading prompt
        """
        
        rubric_text = ""
        for item in rubric_items:
            rubric_text += f"""
{item['item_number']}. {item['description']}
   - Maximum Points: {item['max_points']}
   - Minimum Points: {item.get('min_points', 0)}
"""
        
        prompt = f"""
You are an expert medical educator and grader specializing in pancreatic cancer education. Your task is to grade a response to a pancreatic cancer question using the provided rubrics.

QUESTION:
{question}

RESPONSE TO GRADE:
{response}

GRADING RUBRICS:
{rubric_text}

GRADING INSTRUCTIONS:
1. Evaluate the response against each rubric criterion
2. Make a binary decision for each criterion: if the response includes the required information, award full credit; otherwise, assign zero credit.
3. For rubrics with min_points < 0: if the response does not include the undesirable content, assign zero credit; otherwise, assign min_points. 
4. Consider medical accuracy, completeness, clarity, and appropriateness
5. Be objective and consistent in your grading

REQUIRED OUTPUT FORMAT (JSON):
{{
    "criterion_scores": [
        {{
            "criterion_number": <criterion number>,
            "description": "<criterion description>",
            "score_given": <score assigned>,
            "max_points": <maximum possible points>,
            "justification": "<detailed explanation for the score>"
        }},
        ...
    ]
}}

Please provide your grading as valid JSON only, with no additional text or markdown formatting.
"""
        return prompt
    
    def grade_single_response(self, question: str, response: str, rubric_items: List[Dict], 
                            question_number: int, source: str) -> Dict:
        """
        Grade a single response using Claude-4
        
        Args:
            question (str): The question
            response (str): The response to grade
            rubric_items (List[Dict]): Rubric criteria
            question_number (int): Question number
            source (str): Source of the response (e.g., gpt-4o, Claude, etc.)
            
        Returns:
            Dict: Grading results
        """
        
        try:
            prompt = self.create_grading_prompt(question, response, rubric_items)
            
            print(f"  Debug - Making API call to Claude...")
            
            # Call Claude API
            response_obj = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.1,  # Low temperature for consistent grading
                system="You are an expert medical educator grading responses about pancreatic cancer. Always respond with valid JSON format only, with no additional text or markdown formatting.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            print(f"  Debug - API call completed successfully")
            print(f"  Debug - Response object type: {type(response_obj)}")
            print(f"  Debug - Response content type: {type(response_obj.content)}")
            print(f"  Debug - Content length: {len(response_obj.content)}")
            
            if not response_obj.content or len(response_obj.content) == 0:
                raise ValueError("No content in Claude response")
            
            # Get the raw response
            raw_response = response_obj.content[0].text.strip()
            
            # Debug: Print the raw response
            print(f"  Debug - Raw Claude response length: {len(raw_response)}")
            print(f"  Debug - Raw response type: {type(raw_response)}")
            if raw_response:
                print(f"  Debug - First 200 chars: {repr(raw_response[:200])}")
            else:
                print(f"  Debug - Raw response is empty!")
            
            # Check if response is empty
            if not raw_response:
                raise ValueError("Empty response from Claude")
            
            # Sometimes Claude returns JSON wrapped in markdown code blocks, clean it
            if raw_response.startswith('```json'):
                raw_response = raw_response.replace('```json', '').replace('```', '').strip()
                print(f"  Debug - Cleaned markdown, new length: {len(raw_response)}")
            elif raw_response.startswith('```'):
                raw_response = raw_response.replace('```', '').strip()
                print(f"  Debug - Cleaned markdown, new length: {len(raw_response)}")
            
            print(f"  Debug - Final response to parse: {repr(raw_response[:100])}")
            
            # Parse the JSON response
            grading_result = json.loads(raw_response)
            
            # Add metadata
            grading_result.update({
                "question_number": question_number,
                "source": source,
                "graded_at": datetime.now().isoformat(),
                "grader_model": self.model
            })
            
            # Add to history
            self.grading_history.append(grading_result)
            
            return grading_result
            
        except json.JSONDecodeError as e:
            print(f"Error parsing Claude response for Q{question_number}: {e}")
            print(f"Raw response: {raw_response if 'raw_response' in locals() else 'No response captured'}")
            return self._create_error_result(question_number, source, f"JSON parse error: {e}")
        
        except Exception as e:
            print(f"Error grading Q{question_number}: {e}")
            return self._create_error_result(question_number, source, f"Grading error: {e}")
    
    def _create_error_result(self, question_number: int, source: str, error_msg: str) -> Dict:
        """Create an error result when grading fails"""
        return {
            "question_number": question_number,
            "source": source,
            "total_score": 0,
            "max_possible_score": 0,
            "percentage": 0,
            "error": error_msg,
            "graded_at": datetime.now().isoformat(),
            "grader_model": self.model
        }
    
    def grade_all_responses(self, qa_data: List[Dict], rubrics_data: Dict, 
                          delay_seconds: float = 1.0) -> List[Dict]:
        """
        Grade all responses using the rubrics
        
        Args:
            qa_data (List[Dict]): Q&A response data
            rubrics_data (Dict): Rubrics data
            delay_seconds (float): Delay between API calls to avoid rate limits
            
        Returns:
            List[Dict]: List of all grading results
        """
        
        all_results = []
        total_questions = len(qa_data)
        
        print(f"Starting to grade {total_questions} responses using Claude...")
        
        for i, qa_item in enumerate(qa_data, 1):
            question_number = qa_item.get('question_number')
            question = qa_item.get('question')
            response = qa_item.get('response')
            source = qa_item.get('source')
            
            print(f"Grading Q{question_number} ({i}/{total_questions}) - Source: {source}")
            
            # Find corresponding rubric
            rubric_question = self._find_rubric_for_question(rubrics_data, question_number)
            
            if not rubric_question:
                print(f"  Warning: No rubric found for Q{question_number}")
                continue
            
            # Grade the response
            result = self.grade_single_response(
                question=question,
                response=response,
                rubric_items=rubric_question.get('rubric_items', []),
                question_number=question_number,
                source=source
            )
            
            all_results.append(result)
            
            if 'error' not in result:
                print(f"  ✓ Successfully graded")
            else:
                print(f"  ✗ Error: {result['error']}")
            
            # Rate limiting delay
            if delay_seconds > 0 and i < total_questions:
                time.sleep(delay_seconds)
        
        print(f"\nCompleted grading {len(all_results)} responses with Claude!")
        return all_results
    
    def _find_rubric_for_question(self, rubrics_data: Dict, question_number: int) -> Optional[Dict]:
        """Find the rubric for a specific question number"""
        questions = rubrics_data.get('questions', [])
        for question in questions:
            if question.get('question_number') == question_number:
                return question
        return None
    
    def save_grading_results(self, results: List[Dict], output_file: str):
        """Save grading results to JSON file"""
        grading_summary = {
            "grading_metadata": {
                "total_responses_graded": len(results),
                "grader_model": self.model,
                "grading_completed_at": datetime.now().isoformat()
            },
            "grading_results": results
        }
        
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(grading_summary, file, indent=2, ensure_ascii=False)
        
        print(f"Grading results saved to: {output_file}")

# Alternative function for grading specific questions only
def grade_specific_questions(api_key: str, qa_file: str, rubrics_file: str, 
                           question_numbers: List[int]):
    """Grade only specific questions"""
    
    grader = Claude4Grader(api_key=api_key)
    
    # Load data
    qa_data = grader.load_qa_responses(qa_file)
    rubrics_data = grader.load_rubrics(rubrics_file)
    
    # Filter to specific questions
    filtered_qa = [qa for qa in qa_data if qa.get('question_number') in question_numbers]
    
    print(f"Grading {len(filtered_qa)} specific questions with Claude: {question_numbers}")
    
    # Grade the filtered responses
    results = grader.grade_all_responses(filtered_qa, rubrics_data)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"claude_selective_grading_{timestamp}.json"
    grader.save_grading_results(results, output_file)
    
    return results

if __name__ == "__main__":
    print("Claude4Grader is ready to use!")
    print("Import this module and initialize with: grader = Claude4Grader(api_key='your-key')")