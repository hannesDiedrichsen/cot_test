import re
import logging
from typing import Optional, Dict, Any

class AnswerExtractor:
    def __init__(self):
        # Primary extraction patterns for "The final answer is:" format
        self.extraction_patterns = {
            "numeric": r'The final answer is:\s*([+-]?\d+(?:\.\d+)?)',
            "multiple_choice": r'The final answer is:\s*([A-E])',
            "binary": r'The final answer is:\s*(yes|no|true|false)',
            "letters": r'The final answer is:\s*([a-zA-Z]+)'
        }
        
        # Fallback patterns for when primary pattern fails
        self.fallback_patterns = {
            "numeric": [
                r'([+-]?\d+(?:\.\d+)?)\s*[.!?]?\s*$',  # Last number in text
                r'=\s*([+-]?\d+(?:\.\d+)?)',           # Number after equals
                r'([+-]?\d+(?:\.\d+)?)'                # Any number (last match)
            ],
            "multiple_choice": [
                r'([A-E])\s*[.!?]?\s*$',              # Last letter A-E
                r'([A-E])'                             # Any letter A-E (last match)
            ],
            "binary": [
                r'(yes|no|true|false)\s*[.!?]?\s*$',      # Last binary answer
                r'(yes|no|true|false)'                     # Any binary answer (last match)
            ],
            "letters": [
                r'([a-zA-Z]+)\s*[.!?]?\s*$',          # Last letters
                r'([a-zA-Z]+)'                         # Any letters (last match)
            ]
        }
    
    def extract_answer(self, response: str, answer_type: str, dataset_name: str = None) -> Optional[str]:
        if not response or not response.strip():
            return None
        
        response_clean = response.strip()
        
        # Try primary pattern first
        primary_answer = self._extract_with_primary_pattern(response_clean, answer_type)
        if primary_answer:
            return self._normalize_answer(primary_answer, answer_type, dataset_name)
        
        # Fall back to secondary patterns
        fallback_answer = self._extract_with_fallback_patterns(response_clean, answer_type)
        if fallback_answer:
            return self._normalize_answer(fallback_answer, answer_type, dataset_name)
        
        # Log extraction failure
        logging.warning(f"Failed to extract answer from response: {response_clean[:100]}...")
        return None
    
    def _extract_with_primary_pattern(self, response: str, answer_type: str) -> Optional[str]:
        pattern = self.extraction_patterns.get(answer_type)
        if not pattern:
            return None
        
        match = re.search(pattern, response, re.IGNORECASE)
        return match.group(1) if match else None
    
    def _extract_with_fallback_patterns(self, response: str, answer_type: str) -> Optional[str]:
        fallback_patterns = self.fallback_patterns.get(answer_type, [])
        
        for pattern in fallback_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                return matches[-1]  # Return last match
        
        return None
    
    def _normalize_answer(self, answer: str, answer_type: str, dataset_name: str = None) -> str:
        answer = answer.strip()
        
        if answer_type == "numeric":
            # Remove any non-numeric characters except decimal point and minus
            answer = re.sub(r'[^\d.-]', '', answer)
            try:
                # Convert to number and back to remove leading zeros
                if '.' in answer:
                    return str(float(answer))
                else:
                    return str(int(answer))
            except ValueError:
                return answer
        
        elif answer_type == "multiple_choice":
            return answer.upper()
        
        elif answer_type == "binary":
            answer_lower = answer.lower()
            # Normalize binary answers
            if answer_lower in ['yes', 'y', 'true', 't']:
                return 'yes'
            elif answer_lower in ['no', 'n', 'false', 'f']:
                return 'no'
            return answer_lower
        
        elif answer_type == "letters":
            return answer.lower()
        
        return answer
    
    def validate_answer_format(self, answer: str, answer_type: str, dataset_name: str = None) -> bool:
        if not answer:
            return False
        
        if answer_type == "numeric":
            try:
                float(answer)
                return True
            except ValueError:
                return False
        
        elif answer_type == "multiple_choice":
            return answer.upper() in ['A', 'B', 'C', 'D', 'E']
        
        elif answer_type == "binary":
            return answer.lower() in ['yes', 'no', 'true', 'false']
        
        elif answer_type == "letters":
            return answer.isalpha()
        
        return True
    
    def compare_answers(self, predicted: str, ground_truth: str, answer_type: str, dataset_name: str = None) -> bool:
        if not predicted or not ground_truth:
            return False
        
        # Normalize both answers for comparison
        pred_normalized = self._normalize_answer(predicted, answer_type, dataset_name)
        truth_normalized = self._normalize_answer(ground_truth, answer_type, dataset_name)
        
        if answer_type == "numeric":
            try:
                return float(pred_normalized) == float(truth_normalized)
            except ValueError:
                return pred_normalized == truth_normalized
        
        else:
            return pred_normalized == truth_normalized