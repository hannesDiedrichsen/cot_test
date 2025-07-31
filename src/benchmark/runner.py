import asyncio
import json
import logging
import yaml
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import asdict
import os

from src.datasets.dataset_loader import DatasetLoader, Question
from src.models.gemini_client import GeminiClient
from src.evaluation.answer_extractor import AnswerExtractor

class BenchmarkRunner:
    def __init__(self, config_path: str = "config/config.yaml"):
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.dataset_loader = DatasetLoader(self.config)
        self.answer_extractor = AnswerExtractor()
        
        # Check API key environment variable
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        self.gemini_client = GeminiClient(self.config)
        
        # Setup logging
        self._setup_logging()
        
        # Results storage
        self.results = []
        
        # Store experiment parameters for results
        self.experiment_params = {
            "thinking_budget": self.config['api'].get('thinking_budget'),
            "temperature": 0.0,  # Fixed in gemini_client.py
            "max_retries": self.config['api']['max_retries'],
            "rate_limit_per_minute": self.config['api']['rate_limit_per_minute'],
            "timeout_seconds": self.config['api']['timeout_seconds']
        }
    
    def _setup_logging(self):
        log_config = self.config['logging']
        log_file = log_config['log_file']
        log_level = getattr(logging, log_config['log_level'])
        
        # Create logs directory
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    async def run_benchmark(self, 
                          datasets: List[str] = None, 
                          models: List[str] = None, 
                          methods: List[str] = None,
                          sample_size: int = None):
        
        # Use defaults if not specified
        datasets = datasets or list(self.config['datasets'].keys())
        models = models or self.config['models']
        methods = methods or ['zero_shot', 'zero_shot_cot']
        
        self.logger.info(f"Starting benchmark with datasets: {datasets}, models: {models}, methods: {methods}")
        
        # Load all questions
        all_questions = []
        for dataset_name in datasets:
            questions = self.dataset_loader.load_dataset(dataset_name, sample_size)
            all_questions.extend(questions)
        
        self.logger.info(f"Loaded {len(all_questions)} questions total")
        
        # Run benchmark for all combinations
        total_runs = len(all_questions) * len(models) * len(methods)
        current_run = 0
        
        for model in models:
            for method in methods:
                for question in all_questions:
                    current_run += 1
                    
                    self.logger.info(f"Processing {current_run}/{total_runs}: {model} - {method} - {question.dataset} - {question.id}")
                    
                    result = await self._process_single_question(question, model, method)
                    self.results.append(result)
                    
                    # Log result
                    self.logger.info(f"Result: {result['is_correct']} - Extracted: '{result['extracted_answer']}' - Ground Truth: '{result['ground_truth']}'")
        
        self.logger.info(f"Benchmark completed. Processed {len(self.results)} results.")
        
        # Save results
        await self._save_results()
        
        # Print summary
        self._print_summary()
    
    async def _process_single_question(self, question: Question, model: str, method: str) -> Dict[str, Any]:
        timestamp = datetime.now().isoformat()
        
        try:
            # Get response from Gemini
            response = await self.gemini_client.ask_question(
                question.question, 
                method, 
                question.type, 
                model
            )
            
            # Extract answer
            extracted_answer = None
            if not response.error:
                extracted_answer = self.answer_extractor.extract_answer(
                    response.stage2_response, 
                    question.type, 
                    question.dataset
                )
            
            # Compare answers
            is_correct = False
            if extracted_answer:
                is_correct = self.answer_extractor.compare_answers(
                    extracted_answer, 
                    question.answer, 
                    question.type, 
                    question.dataset
                )
            
            return {
                "id": question.id,
                "dataset": question.dataset,
                "model": model,
                "method": method,
                "timestamp": timestamp,
                "question": question.question,
                "stage1_prompt": response.stage1_prompt,
                "stage1_response": response.stage1_response,
                "stage2_prompt": response.stage2_prompt,
                "stage2_response": response.stage2_response,
                "extracted_answer": extracted_answer,
                "ground_truth": question.answer,
                "is_correct": is_correct,
                "processing_time_ms": response.total_time_ms,
                "error": response.error,
                # Add experiment parameters
                "thinking_budget": self.experiment_params["thinking_budget"],
                "temperature": self.experiment_params["temperature"],
                "max_retries": self.experiment_params["max_retries"],
                "rate_limit_per_minute": self.experiment_params["rate_limit_per_minute"],
                "timeout_seconds": self.experiment_params["timeout_seconds"]
            }
            
        except Exception as e:
            self.logger.error(f"Error processing question {question.id}: {str(e)}")
            
            return {
                "id": question.id,
                "dataset": question.dataset,
                "model": model,
                "method": method,
                "timestamp": timestamp,
                "question": question.question,
                "stage1_prompt": "",
                "stage1_response": "",
                "stage2_prompt": "",
                "stage2_response": "",
                "extracted_answer": None,
                "ground_truth": question.answer,
                "is_correct": False,
                "processing_time_ms": 0,
                "error": str(e),
                # Add experiment parameters
                "thinking_budget": self.experiment_params["thinking_budget"],
                "temperature": self.experiment_params["temperature"],
                "max_retries": self.experiment_params["max_retries"],
                "rate_limit_per_minute": self.experiment_params["rate_limit_per_minute"],
                "timeout_seconds": self.experiment_params["timeout_seconds"]
            }
    
    async def _save_results(self):
        # Save detailed results as JSON with experiment metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"results/benchmark_results_{timestamp}.json"
        
        Path(results_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Create results structure with metadata
        results_with_metadata = {
            "experiment_metadata": {
                "timestamp": timestamp,
                "total_questions": len(self.results),
                "parameters": self.experiment_params
            },
            "results": self.results
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_with_metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Results saved to {results_file}")
        
        # Save summary CSV
        summary_file = f"results/benchmark_summary_{timestamp}.csv"
        import pandas as pd
        
        df = pd.DataFrame(self.results)
        df.to_csv(summary_file, index=False)
        
        self.logger.info(f"Summary saved to {summary_file}")
        
        # Generate comparison chart
        chart_file = f"results/benchmark_chart_{timestamp}.png"
        self._generate_comparison_chart(df, chart_file)
        self.logger.info(f"Comparison chart saved to {chart_file}")
    
    def _print_summary(self):
        if not self.results:
            return
        
        # Calculate accuracy by dataset, model, method
        import pandas as pd
        df = pd.DataFrame(self.results)
        
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        # Show experiment parameters
        print("Experiment Parameters:")
        for key, value in self.experiment_params.items():
            print(f"  {key}: {value}")
        print()
        
        # Overall accuracy
        overall_accuracy = df['is_correct'].mean()
        print(f"Overall Accuracy: {overall_accuracy:.3f} ({df['is_correct'].sum()}/{len(df)})")
        
        # Accuracy by method
        print("\nAccuracy by Method:")
        method_accuracy = df.groupby('method')['is_correct'].agg(['mean', 'sum', 'count'])
        for method, stats in method_accuracy.iterrows():
            print(f"  {method}: {stats['mean']:.3f} ({stats['sum']}/{stats['count']})")
        
        # Accuracy by model
        print("\nAccuracy by Model:")
        model_accuracy = df.groupby('model')['is_correct'].agg(['mean', 'sum', 'count'])
        for model, stats in model_accuracy.iterrows():
            print(f"  {model}: {stats['mean']:.3f} ({stats['sum']}/{stats['count']})")
        
        # Accuracy by dataset
        print("\nAccuracy by Dataset:")
        dataset_accuracy = df.groupby('dataset')['is_correct'].agg(['mean', 'sum', 'count'])
        for dataset, stats in dataset_accuracy.iterrows():
            print(f"  {dataset}: {stats['mean']:.3f} ({stats['sum']}/{stats['count']})")
        
        # Detailed breakdown
        print("\nDetailed Breakdown (Dataset x Method x Model):")
        detailed = df.groupby(['dataset', 'method', 'model'])['is_correct'].agg(['mean', 'sum', 'count'])
        for (dataset, method, model), stats in detailed.iterrows():
            print(f"  {dataset} | {method} | {model}: {stats['mean']:.3f} ({stats['sum']}/{stats['count']})")
        
        # Chart info
        print("\n📊 VISUAL RESULTS:")
        print("  • Comparison charts automatically generated in results/ folder")
        print("  • Check benchmark_chart_TIMESTAMP.png for detailed visualizations")
        if df['model'].nunique() > 1:
            print("  • Multi-model comparison available in benchmark_chart_TIMESTAMP_models.png")
        
        # Error analysis
        errors = df[df['error'].notna()]
        if len(errors) > 0:
            print(f"\nErrors encountered: {len(errors)}")
            error_counts = errors['error'].value_counts()
            for error, count in error_counts.head(5).items():
                print(f"  {error}: {count}")
        
        print("="*80)
    
    def _generate_comparison_chart(self, df, chart_file: str):
        """Generate bar chart comparing Zero Shot vs Zero Shot CoT performance"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Chart 1: Accuracy by Dataset and Method
        dataset_method_acc = df.groupby(['dataset', 'method'])['is_correct'].agg(['mean', 'sum', 'count']).reset_index()
        dataset_method_acc['accuracy'] = dataset_method_acc['mean']
        
        # Pivot for grouped bar chart
        pivot_data = dataset_method_acc.pivot(index='dataset', columns='method', values='accuracy')
        
        # Plot grouped bar chart
        pivot_data.plot(kind='bar', ax=ax1, width=0.8, color=['#2E86AB', '#A23B72'])
        ax1.set_title('Accuracy by Dataset: Zero Shot vs Zero Shot CoT', fontsize=14, fontweight='bold', pad=20)
        ax1.set_xlabel('Dataset', fontweight='bold')
        ax1.set_ylabel('Accuracy', fontweight='bold')
        ax1.set_ylim(0, 1.1)
        ax1.legend(title='Method', title_fontsize=12, fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for container in ax1.containers:
            ax1.bar_label(container, fmt='%.2f', fontsize=9, padding=3)
        
        # Rotate x-axis labels
        ax1.tick_params(axis='x', rotation=45)
        
        # Chart 2: Overall Method Comparison
        method_acc = df.groupby('method')['is_correct'].agg(['mean', 'sum', 'count']).reset_index()
        method_acc['accuracy'] = method_acc['mean']
        
        bars = ax2.bar(method_acc['method'], method_acc['accuracy'], 
                      color=['#2E86AB', '#A23B72'], width=0.6, alpha=0.8)
        
        ax2.set_title('Overall Accuracy Comparison', fontsize=14, fontweight='bold', pad=20)
        ax2.set_xlabel('Method', fontweight='bold')
        ax2.set_ylabel('Accuracy', fontweight='bold')
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels and sample counts
        for i, (bar, row) in enumerate(zip(bars, method_acc.itertuples())):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}\n({row.sum}/{row.count})', 
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save chart
        plt.savefig(chart_file, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        # Also create a detailed model comparison if multiple models
        if df['model'].nunique() > 1:
            self._generate_model_comparison_chart(df, chart_file.replace('.png', '_models.png'))
    
    def _generate_model_comparison_chart(self, df, chart_file: str):
        """Generate detailed model comparison chart"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.style.use('default')
        sns.set_palette("Set2")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Group by dataset, method, and model
        detailed_acc = df.groupby(['dataset', 'method', 'model'])['is_correct'].mean().reset_index()
        detailed_acc['accuracy'] = detailed_acc['is_correct']
        
        # Create grouped bar chart
        datasets = detailed_acc['dataset'].unique()
        methods = detailed_acc['method'].unique()
        models = detailed_acc['model'].unique()
        
        x = range(len(datasets))
        width = 0.35 / len(models)
        
        for i, method in enumerate(methods):
            for j, model in enumerate(models):
                data = detailed_acc[(detailed_acc['method'] == method) & 
                                  (detailed_acc['model'] == model)]
                
                # Align data with datasets
                y_values = []
                for dataset in datasets:
                    dataset_data = data[data['dataset'] == dataset]
                    if len(dataset_data) > 0:
                        y_values.append(dataset_data['accuracy'].iloc[0])
                    else:
                        y_values.append(0)
                
                x_pos = [pos + (i * len(models) + j) * width for pos in x]
                label = f"{method} - {model}"
                
                ax.bar(x_pos, y_values, width, label=label, alpha=0.8)
        
        ax.set_title('Detailed Model and Method Comparison by Dataset', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Dataset', fontweight='bold')
        ax.set_ylabel('Accuracy', fontweight='bold')
        ax.set_xticks([pos + width * (len(methods) * len(models) - 1) / 2 for pos in x])
        ax.set_xticklabels(datasets, rotation=45)
        ax.set_ylim(0, 1.1)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(chart_file, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()