#!/usr/bin/env python3
import json
import openai
import os
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

class GPTGrader:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        """
        Initialize the GPT-4 grader
        
        Args:
            api_key (str): OpenAI API key
            model (str): Model to use (default: gpt-4)
        """
        self.client = openai.OpenAI(api_key=api_key)
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
        
    def load_synthetic_rubrics(self, rubrics_file_path: str) -> Dict:
        """Load rubrics from JSONL file (one JSON object per line) and convert to standard format"""
        try:
            rubrics = {"questions": []}
            with open(rubrics_file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    if line.strip():  # Skip empty lines
                        raw_entry = json.loads(line)

                        # Extract question number from question_id (e.g., "Q1" -> 1)
                        question_id = raw_entry.get('question_id', '')
                        question_number = int(question_id.replace('Q', '')) if question_id.startswith('Q') else 0

                        # Convert rubric items to standard format
                        converted_items = []
                        for idx, item in enumerate(raw_entry.get('rubric_items', []), start=1):
                            points = item.get('points', 0)
                            converted_item = {
                                'item_number': idx,
                                'description': item.get('item_description', ''),
                                'min_points': points if points < 0 else 0,
                                'max_points': 0 if points < 0 else points
                            }
                            converted_items.append(converted_item)

                        # Create formatted question entry
                        formatted_entry = {
                            'question_number': question_number,
                            'question_text': raw_entry.get('question', ''),
                            'rubric_items': converted_items
                        }

                        rubrics["questions"].append(formatted_entry)

            return rubrics
        except Exception as e:
            print(f"Error loading synthetic rubrics file: {e}")
            return {}
    
    
    def create_grading_prompt(self, question: str, response: str, rubric_items: List[Dict]) -> str:
        """
        Create a detailed grading prompt for GPT-4
        
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

Please provide your grading as valid JSON only, with no additional text.
"""
        return prompt
    
    def grade_single_response(self, question: str, response: str, rubric_items: List[Dict], 
                            question_number: int, source: str) -> Dict:
        """
        Grade a single response using GPT
        
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
            
            if self.model == "gpt-4.1":
            # Call GPT-4 API
                response_obj = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert medical educator grading responses about pancreatic cancer. Always respond with valid JSON format only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,  # Low temperature for consistent grading
                    max_tokens=2000
                )
                grading_result = json.loads(response_obj.choices[0].message.content)
            elif self.model == "gpt-5":
                response_obj = self.client.responses.create(
                    model="gpt-5",
                    instructions="You are an expert medical educator grading responses about pancreatic cancer. Always respond with valid JSON format only.",
                    input=prompt
                    )
                grading_result = json.loads(response_obj.output[1].content[0].text)
   
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
            print(f"Error parsing GPT-4 response for Q{question_number}: {e}")
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
        
        print(f"Starting to grade {total_questions} responses...")
        
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
                pass
            else:
                print(f"  ✗ Error: {result['error']}")
            
            # Rate limiting delay
            if delay_seconds > 0 and i < total_questions:
                time.sleep(delay_seconds)
        
        print(f"\nCompleted grading {len(all_results)} responses!")
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
    
    def create_grading_summary(self, results: List[Dict]) -> Dict:
        """Create a summary of grading results"""
        
        successful_gradings = [r for r in results if 'error' not in r]
        
        if not successful_gradings:
            return {"error": "No successful gradings to summarize"}
        
        # Calculate overall statistics
        total_scores = [r['total_score'] for r in successful_gradings]
        max_scores = [r['max_possible_score'] for r in successful_gradings]
        percentages = [r['percentage'] for r in successful_gradings]
        
        # Group by source
        by_source = {}
        for result in successful_gradings:
            source = result['source']
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(result['percentage'])
        
        source_averages = {source: sum(scores)/len(scores) 
                          for source, scores in by_source.items()}
        
        summary = {
            "overall_statistics": {
                "total_responses_graded": len(successful_gradings),
                "average_score": sum(total_scores) / len(total_scores),
                "average_max_score": sum(max_scores) / len(max_scores),
                "average_percentage": sum(percentages) / len(percentages),
                "highest_percentage": max(percentages),
                "lowest_percentage": min(percentages)
            },
            "performance_by_source": source_averages,
            "grade_distribution": self._calculate_grade_distribution(percentages),
            "top_performing_responses": sorted(successful_gradings, 
                                             key=lambda x: x['percentage'], 
                                             reverse=True)[:5],
            "responses_needing_improvement": sorted([r for r in successful_gradings if r['percentage'] < 80], 
                                                  key=lambda x: x['percentage'])[:5]
        }
        
        return summary
    
    def _calculate_grade_distribution(self, percentages: List[float]) -> Dict[str, int]:
        """Calculate grade distribution"""
        distribution = {
            "A (90-100%)": 0,
            "B (80-89%)": 0, 
            "C (70-79%)": 0,
            "D (60-69%)": 0,
            "F (<60%)": 0
        }
        
        for percentage in percentages:
            if percentage >= 90:
                distribution["A (90-100%)"] += 1
            elif percentage >= 80:
                distribution["B (80-89%)"] += 1
            elif percentage >= 70:
                distribution["C (70-79%)"] += 1
            elif percentage >= 60:
                distribution["D (60-69%)"] += 1
            else:
                distribution["F (<60%)"] += 1
        
        return distribution
    
    def export_to_excel(self, results: List[Dict], output_file: str):
        """Export grading results to Excel file"""
        try:
            # Create main results DataFrame
            main_data = []
            for result in results:
                if 'error' not in result:
                    main_data.append({
                        'Question_Number': result['question_number'],
                        'Source': result['source'],
                        'Total_Score': result['total_score'],
                        'Max_Score': result['max_possible_score'],
                        'Percentage': result['percentage'],
                        'Overall_Feedback': result.get('overall_feedback', ''),
                        'Graded_At': result['graded_at']
                    })
            
            df_main = pd.DataFrame(main_data)
            
            # Create detailed criteria DataFrame
            criteria_data = []
            for result in results:
                if 'error' not in result and 'criterion_scores' in result:
                    for criterion in result['criterion_scores']:
                        criteria_data.append({
                            'Question_Number': result['question_number'],
                            'Source': result['source'],
                            'Criterion_Number': criterion['criterion_number'],
                            'Criterion_Description': criterion['description'],
                            'Score_Given': criterion['score_given'],
                            'Max_Points': criterion['max_points'],
                            'Justification': criterion['justification']
                        })
            
            df_criteria = pd.DataFrame(criteria_data)
            
            # Save to Excel with multiple sheets
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                df_main.to_excel(writer, sheet_name='Summary', index=False)
                df_criteria.to_excel(writer, sheet_name='Detailed_Criteria', index=False)
            
            print(f"Results exported to Excel: {output_file}")
            
        except Exception as e:
            print(f"Error exporting to Excel: {e}")




