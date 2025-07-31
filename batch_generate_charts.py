#!/usr/bin/env python3
"""
Batch script to generate charts for multiple benchmark result files.

Usage:
  python batch_generate_charts.py
  python batch_generate_charts.py --directory results/
  python batch_generate_charts.py --pattern "*benchmark_results*.json"
"""
import argparse
import json
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime
import glob

# Import chart generation functions
from generate_charts import generate_comparison_chart, generate_model_comparison_chart, load_data_from_json, load_data_from_csv, validate_data

def process_file(input_file: Path, output_dir: Path) -> bool:
    """Process a single benchmark file and generate charts"""
    try:
        print(f"\n📊 Processing: {input_file.name}")
        
        # Load data
        if input_file.suffix.lower() == '.json':
            df = load_data_from_json(input_file)
        elif input_file.suffix.lower() == '.csv':
            df = load_data_from_csv(input_file)
        else:
            print(f"⚠️  Skipping unsupported file format: {input_file}")
            return False
        
        # Validate data
        validate_data(df)
        
        # Extract timestamp from filename or use current time
        filename_stem = input_file.stem
        if 'benchmark_results_' in filename_stem:
            timestamp = filename_stem.replace('benchmark_results_', '')
        elif 'benchmark_summary_' in filename_stem:
            timestamp = filename_stem.replace('benchmark_summary_', '')
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate main comparison chart
        main_chart = output_dir / f"chart_{timestamp}.png"
        generate_comparison_chart(df, main_chart)
        
        # Generate model comparison chart if multiple models
        if 'model' in df.columns and df['model'].nunique() > 1:
            model_chart = output_dir / f"chart_{timestamp}_models.png"
            generate_model_comparison_chart(df, model_chart)
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {input_file}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Batch generate charts for multiple benchmark result files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all files in results/ directory
  python batch_generate_charts.py
  
  # Process files in custom directory
  python batch_generate_charts.py --directory data/benchmarks/
  
  # Process only JSON files matching pattern
  python batch_generate_charts.py --pattern "*results*.json"
  
  # Custom output directory
  python batch_generate_charts.py --output charts/
        """
    )
    
    parser.add_argument(
        '--directory', '-d',
        default='results/',
        help='Directory to search for benchmark files (default: results/)'
    )
    
    parser.add_argument(
        '--pattern', '-p',
        default='benchmark_*.{json,csv}',
        help='File pattern to match (default: benchmark_*.{json,csv})'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output directory for charts (default: same as input directory)'
    )
    
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Search recursively in subdirectories'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    search_dir = Path(args.directory)
    if not search_dir.exists():
        print(f"❌ Error: Directory {search_dir} does not exist")
        sys.exit(1)
    
    output_dir = Path(args.output) if args.output else search_dir / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find matching files
    if args.recursive:
        # Search recursively
        json_files = list(search_dir.rglob("benchmark_*.json"))
        csv_files = list(search_dir.rglob("benchmark_*.csv"))
    else:
        # Search in current directory only
        json_files = list(search_dir.glob("benchmark_*.json"))
        csv_files = list(search_dir.glob("benchmark_*.csv"))
    
    all_files = sorted(json_files + csv_files)
    
    if not all_files:
        print(f"❌ No benchmark files found in {search_dir}")
        print(f"   Looking for pattern: benchmark_*.{{json,csv}}")
        sys.exit(1)
    
    print(f"🔍 Found {len(all_files)} benchmark files:")
    for file in all_files:
        print(f"   • {file}")
    
    print(f"\n📁 Output directory: {output_dir}")
    
    # Process files
    successful = 0
    failed = 0
    
    for file in all_files:
        if process_file(file, output_dir):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n🎉 Batch processing completed!")
    print(f"   ✅ Successfully processed: {successful} files")
    if failed > 0:
        print(f"   ❌ Failed to process: {failed} files")
    print(f"   📊 Charts saved to: {output_dir}")

if __name__ == "__main__":
    main()