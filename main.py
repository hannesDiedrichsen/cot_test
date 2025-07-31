#!/usr/bin/env python3
import argparse
import asyncio
import sys
from pathlib import Path

import dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.benchmark.runner import BenchmarkRunner

def parse_args():
    parser = argparse.ArgumentParser(description="Zero Shot CoT Benchmark for Gemini Models")
    
    parser.add_argument(
        "--datasets", 
        nargs="+", 
        choices=["AddSub", "CommonsenseQA", "MultiArith", "SingleEq", "coin_flip", "grade-school-math", "last_letters", 
                "GSM8K", "ARC_Challenge", "BoolQ", "Winogrande"],
        help="Datasets to benchmark (default: all)"
    )
    
    parser.add_argument(
        "--models",
        nargs="+", 
        default=["gemini-2.5-flash", "gemini-1.5-pro"],
        help="Models to test (default: gemini-2.5-flash, gemini-1.5-pro)"
    )
    
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["zero_shot", "zero_shot_cot"],
        default=["zero_shot", "zero_shot_cot"],
        help="Methods to test (default: both)"
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Number of questions per dataset (default: all)"
    )
    
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config file (default: config/config.yaml)"
    )
    
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable Gemini's internal thinking (sets thinking_budget=0)"
    )
    
    parser.add_argument(
        "--thinking-budget",
        type=int,
        help="Set custom thinking budget (0=no thinking, >0=limited thinking)"
    )
    
    return parser.parse_args()

async def main():
    args = parse_args()
    
    print("Zero Shot Chain-of-Thought Benchmark")
    print("====================================")
    print(f"Datasets: {args.datasets or 'all'}")
    print(f"Models: {args.models}")
    print(f"Methods: {args.methods}")
    print(f"Sample size: {args.sample_size or 'all'}")
    print()
    
    try:
        runner = BenchmarkRunner(args.config)
        
        # Override thinking configuration if specified
        if args.no_thinking:
            runner.config['api']['thinking_budget'] = 0
            runner.experiment_params['thinking_budget'] = 0
            print("🧠 Gemini thinking disabled (thinking_budget=0)")
        elif args.thinking_budget is not None:
            runner.config['api']['thinking_budget'] = args.thinking_budget
            runner.experiment_params['thinking_budget'] = args.thinking_budget
            print(f"🧠 Gemini thinking budget set to: {args.thinking_budget}")
        
        # Update the GeminiClient's thinking_budget as well
        runner.gemini_client.thinking_budget = runner.config['api'].get('thinking_budget')
        
        await runner.run_benchmark(
            datasets=args.datasets,
            models=args.models,
            methods=args.methods,
            sample_size=args.sample_size
        )
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    dotenv.load_dotenv()
    
    asyncio.run(main())