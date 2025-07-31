import json
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Question:
    id: str
    question: str
    answer: str
    dataset: str
    type: str

class DatasetLoader:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.datasets_config = config['datasets']
    
    def load_all_datasets(self, sample_size: int = None) -> List[Question]:
        all_questions = []
        
        for dataset_name, dataset_config in self.datasets_config.items():
            questions = self.load_dataset(dataset_name, sample_size)
            all_questions.extend(questions)
        
        return all_questions
    
    def load_dataset(self, dataset_name: str, sample_size: int = None) -> List[Question]:
        if dataset_name not in self.datasets_config:
            raise ValueError(f"Dataset {dataset_name} not found in config")
        
        dataset_config = self.datasets_config[dataset_name]
        path = Path(dataset_config['path'])
        format_type = dataset_config['format']
        answer_type = dataset_config['type']
        
        if format_type == 'json':
            return self._load_json_dataset(dataset_name, path, answer_type, sample_size)
        elif format_type == 'jsonl':
            return self._load_jsonl_dataset(dataset_name, path, answer_type, sample_size)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _load_json_dataset(self, dataset_name: str, path: Path, answer_type: str, sample_size: int = None) -> List[Question]:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        questions = []
        
        if dataset_name == 'AddSub':
            for i, item in enumerate(data):
                if sample_size and i >= sample_size:
                    break
                questions.append(Question(
                    id=f"{dataset_name}_{item.get('iIndex', i)}",
                    question=item['sQuestion'].strip(),
                    answer=str(int(item['lSolutions'][0])) if isinstance(item['lSolutions'][0], float) else str(item['lSolutions'][0]),
                    dataset=dataset_name,
                    type=answer_type
                ))
        
        elif dataset_name == 'MultiArith':
            for i, item in enumerate(data):
                if sample_size and i >= sample_size:
                    break
                questions.append(Question(
                    id=f"{dataset_name}_{item.get('iIndex', i)}",
                    question=item['sQuestion'].strip(),
                    answer=str(int(item['lSolutions'][0])) if isinstance(item['lSolutions'][0], float) else str(item['lSolutions'][0]),
                    dataset=dataset_name,
                    type=answer_type
                ))
        
        elif dataset_name == 'SingleEq':
            for i, item in enumerate(data):
                if sample_size and i >= sample_size:
                    break
                questions.append(Question(
                    id=f"{dataset_name}_{item.get('iIndex', i)}",
                    question=item['sQuestion'].strip(),
                    answer=str(int(item['lSolutions'][0])),  # Convert float to int then string
                    dataset=dataset_name,
                    type=answer_type
                ))
        
        elif dataset_name == 'coin_flip':
            # coin_flip has a different structure: {"examples": [...]}
            examples = data.get('examples', data)  # Handle both structures
            for i, item in enumerate(examples):
                if sample_size and i >= sample_size:
                    break
                questions.append(Question(
                    id=f"{dataset_name}_{i}",
                    question=item['question'].strip(),
                    answer=item['answer'].strip().lower(),
                    dataset=dataset_name,
                    type=answer_type
                ))
        
        elif dataset_name == 'last_letters':
            # last_letters has a different structure: {"examples": [...]}
            examples = data.get('examples', data)  # Handle both structures
            for i, item in enumerate(examples):
                if sample_size and i >= sample_size:
                    break
                questions.append(Question(
                    id=f"{dataset_name}_{i}",
                    question=item['question'].strip(),
                    answer=item['answer'].strip().lower(),
                    dataset=dataset_name,
                    type=answer_type
                ))
        
        # New math/logic datasets
        elif dataset_name in ['MATH', 'GSM8K', 'TheoremQA']:
            for i, item in enumerate(data):
                if sample_size and i >= sample_size:
                    break
                questions.append(Question(
                    id=f"{dataset_name}_{item.get('iIndex', i)}",
                    question=item['sQuestion'].strip(),
                    answer=str(int(float(item['lSolutions'][0]))) if item['lSolutions'][0].replace('.','').replace('-','').isdigit() else str(item['lSolutions'][0]),
                    dataset=dataset_name,
                    type=answer_type
                ))
        
        elif dataset_name in ['HellaSwag', 'ARC_Challenge', 'LogiQA', 'Winogrande', 'TruthfulQA']:
            for i, item in enumerate(data):
                if sample_size and i >= sample_size:
                    break
                questions.append(Question(
                    id=f"{dataset_name}_{item.get('iIndex', i)}",
                    question=item['sQuestion'].strip(),
                    answer=item['lSolutions'][0].strip(),
                    dataset=dataset_name,
                    type=answer_type
                ))
        
        elif dataset_name == 'BoolQ':
            # BoolQ has the same structure as coin_flip and last_letters
            examples = data.get('examples', data)
            for i, item in enumerate(examples):
                if sample_size and i >= sample_size:
                    break
                questions.append(Question(
                    id=f"{dataset_name}_{i}",
                    question=item['question'].strip(),
                    answer=item['answer'].strip().lower(),
                    dataset=dataset_name,
                    type=answer_type
                ))
        
        return questions
    
    def _load_jsonl_dataset(self, dataset_name: str, path: Path, answer_type: str, sample_size: int = None) -> List[Question]:
        questions = []
        
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if sample_size and i >= sample_size:
                    break
                
                item = json.loads(line.strip())
                
                if dataset_name == 'CommonsenseQA':
                    # Format question with multiple choice options
                    stem = item['question']['stem'].strip()
                    choices = item['question']['choices']
                    choices_text = "\n".join([f"{choice['label']}) {choice['text']}" for choice in choices])
                    full_question = f"{stem}\n\n{choices_text}"
                    
                    questions.append(Question(
                        id=f"{dataset_name}_{item['id']}",
                        question=full_question,
                        answer=item['answerKey'].strip(),
                        dataset=dataset_name,
                        type=answer_type
                    ))
                
                elif dataset_name == 'grade-school-math':
                    answer_match = item['answer'].split('####')[-1].strip()
                    questions.append(Question(
                        id=f"{dataset_name}_{i}",
                        question=item['question'].strip(),
                        answer=answer_match,
                        dataset=dataset_name,
                        type=answer_type
                    ))
        
        return questions