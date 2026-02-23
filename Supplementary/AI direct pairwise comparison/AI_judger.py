import openai
from datetime import datetime
import json
import time
from typing import Dict, List

class AIDirectJudge:
    def __init__(self, api_key: str, model: str = "gpt-5"):
        """
        Initialize the AI Direct Judge
        
        Args:
            api_key (str): OpenAI API key
            model (str): Model to use for judgment (default: gpt-4)
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.judgment_history = []
    
    def create_judgment_prompt(self, question: str, response_a: str, response_b: str, 
                              source_a: str, source_b: str) -> str:
        """
        Create a judgment prompt for comparing two responses
        
        Args:
            question (str): The pancreatic cancer question
            response_a (str): First response to compare
            response_b (str): Second response to compare
            source_a (str): Source of first response (e.g., gpt-4, claude-3.5)
            source_b (str): Source of second response
            
        Returns:
            str: Formatted judgment prompt
        """
        
        prompt = f"""
You are an expert pancreatic cancer physician and medical educator with extensive experience in patient communication and medical education. Your task is to compare two responses to a pancreatic cancer question and determine which one is better.

QUESTION:
{question}

RESPONSE A (from {source_a}):
{response_a}

RESPONSE B (from {source_b}):
{response_b}

EVALUATION CRITERIA:
Please evaluate both responses based on the following criteria:
1. Medical Accuracy: Are the facts and medical information correct?
2. Completeness: Does the response address all aspects of the question?
3. Clarity and Communication: Is the response clear, understandable, and well-organized?
4. Appropriateness: Is the tone and level of detail appropriate for a patient/family member?
5. Actionable Guidance: Does the response provide practical, actionable advice?
6. Safety: Does the response include appropriate safety considerations and warnings?
7. Empathy and Support: Is the response compassionate and supportive?

INSTRUCTIONS:
1. Carefully analyze both responses against each criterion
2. Consider the overall quality and usefulness for a patient/family member
3. Make a clear judgment about which response is better overall
4. Provide detailed reasoning for your decision

REQUIRED OUTPUT FORMAT (JSON):
{{
    "winner": "A" or "B" or "TIE",
    "confidence": <1-5 scale, where 5 is very confident>,
    "overall_reasoning": "<comprehensive explanation of your decision>"
}}

Please provide your judgment as valid JSON only, with no additional text.
"""
        return prompt
    
    def judge_responses(self, question: str, response_a: str, response_b: str, 
                       source_a: str, source_b: str, question_number: int = None) -> Dict:
        """
        Judge which of two responses is better
        
        Args:
            question (str): The question
            response_a (str): First response
            response_b (str): Second response
            source_a (str): Source of first response
            source_b (str): Source of second response
            question_number (int): Optional question number for tracking
            
        Returns:
            Dict: Judgment results
        """
        
        try:
            prompt = self.create_judgment_prompt(question, response_a, response_b, source_a, source_b)

            response_obj = self.client.responses.create(
                    model="gpt-5",
                    instructions="You are an expert pancreatic cancer physician making objective comparisons between medical responses. Always respond with valid JSON format only.",
                    input=prompt
            )
            judgment_result= json.loads(response_obj.output[1].content[0].text)
            
            # Parse the JSON response
            #judgment_result = json.loads(response_obj.choices[0].message.content)
            
            # Add metadata
            judgment_result.update({
                "question_number": question_number,
                "source_a": source_a,
                "source_b": source_b,
                "judged_at": datetime.now().isoformat(),
                "judge_model": self.model
            })
            
            # Add to history
            self.judgment_history.append(judgment_result)
            
            return judgment_result
            
        except json.JSONDecodeError as e:
            print(f"Error parsing AI judge response for Q{question_number}: {e}")
            return self._create_error_result(question_number, source_a, source_b, f"JSON parse error: {e}")
        
        except Exception as e:
            print(f"Error judging Q{question_number}: {e}")
            return self._create_error_result(question_number, source_a, source_b, f"Judgment error: {e}")
    
    def _create_error_result(self, question_number: int, source_a: str, source_b: str, error_msg: str) -> Dict:
        """Create an error result when judgment fails"""
        return {
            "question_number": question_number,
            "source_a": source_a,
            "source_b": source_b,
            "winner": "ERROR",
            "error": error_msg,
            "judged_at": datetime.now().isoformat(),
            "judge_model": self.model
        }
    
    def judge_multiple_comparisons(self, comparisons: List[Dict], delay_seconds: float = 1.0) -> List[Dict]:
        """
        Judge multiple response comparisons
        
        Args:
            comparisons (List[Dict]): List of comparison data with keys:
                - question, response_a, response_b, source_a, source_b, question_number
            delay_seconds (float): Delay between API calls
            
        Returns:
            List[Dict]: List of judgment results
        """
        
        all_results = []
        total_comparisons = len(comparisons)
        
        print(f"Starting to judge {total_comparisons} response comparisons...")
        
        for i, comparison in enumerate(comparisons, 1):
            question = comparison.get('question')
            response_a = comparison.get('response_a')
            response_b = comparison.get('response_b')
            source_a = comparison.get('source_a')
            source_b = comparison.get('source_b')
            question_number = comparison.get('question_number', i)
            
            print(f"Judging Q{question_number} ({i}/{total_comparisons}) - {source_a} vs {source_b}")
            
            result = self.judge_responses(
                question=question,
                response_a=response_a,
                response_b=response_b,
                source_a=source_a,
                source_b=source_b,
                question_number=question_number
            )
            
            all_results.append(result)
            
            if 'error' not in result:
                winner = result.get('winner', 'UNKNOWN')
                confidence = result.get('confidence', 'N/A')
                print(f"  Winner: {winner} (Confidence: {confidence}/5)")
            else:
                print(f"  ✗ Error: {result['error']}")
            
            # Rate limiting delay
            if delay_seconds > 0 and i < total_comparisons:
                time.sleep(delay_seconds)
        
        print(f"\nCompleted judging {len(all_results)} comparisons!")
        return all_results
    
    def save_judgment_results(self, results: List[Dict], output_file: str):
        """Save judgment results to JSON file"""
        judgment_summary = {
            "judgment_metadata": {
                "total_comparisons": len(results),
                "judge_model": self.model,
                "judgment_completed_at": datetime.now().isoformat()
            },
            "judgment_results": results
        }
        
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(judgment_summary, file, indent=2, ensure_ascii=False)
        
        print(f"Judgment results saved to: {output_file}")
    
    def create_judgment_summary(self, results: List[Dict]) -> Dict:
        """Create a summary of judgment results"""
        
        successful_judgments = [r for r in results if 'error' not in r and r.get('winner') != 'ERROR']
        
        if not successful_judgments:
            return {"error": "No successful judgments to summarize"}
        
        # Count winners
        winner_counts = {"A": 0, "B": 0, "TIE": 0}
        source_wins = {}
        
        for result in successful_judgments:
            winner = result.get('winner')
            if winner in winner_counts:
                winner_counts[winner] += 1
            
            # Track wins by source
            source_a = result.get('source_a')
            source_b = result.get('source_b')
            
            if winner == 'A':
                source_wins[source_a] = source_wins.get(source_a, 0) + 1
            elif winner == 'B':
                source_wins[source_b] = source_wins.get(source_b, 0) + 1
        
        # Calculate confidence statistics
        confidences = [r.get('confidence', 0) for r in successful_judgments if r.get('confidence')]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        summary = {
            "total_judgments": len(successful_judgments),
            "winner_distribution": winner_counts,
            "source_performance": source_wins,
            "average_confidence": avg_confidence,
            "high_confidence_judgments": len([r for r in successful_judgments if r.get('confidence', 0) >= 4]),
            "ties": winner_counts["TIE"],
            "error_count": len([r for r in results if 'error' in r or r.get('winner') == 'ERROR'])
        }
        
        return summary