#!/usr/bin/env python3
"""
Standalone script to generate comparison charts from existing benchmark results.

Usage:
  python generate_charts.py results/benchmark_results_20250729_125508.json
  python generate_charts.py --input results/ --latest
  python generate_charts.py --input results/benchmark_summary_20250729_125508.csv
"""
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from datetime import datetime
import glob

def generate_comparison_chart(df, output_file: str):
    """Generate combined bar chart comparing Zero Shot vs Zero Shot CoT performance"""
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create single figure for combined visualization
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    
    # Prepare dataset-specific data
    dataset_method_acc = df.groupby(['dataset', 'method'])['is_correct'].agg(['mean', 'sum', 'count']).reset_index()
    dataset_method_acc['accuracy'] = dataset_method_acc['mean']
    
    # Get overall method comparison
    method_acc = df.groupby('method')['is_correct'].agg(['mean', 'sum', 'count']).reset_index()
    method_acc['accuracy'] = method_acc['mean']
    
    # Pivot for grouped bar chart
    pivot_data = dataset_method_acc.pivot(index='dataset', columns='method', values='accuracy')
    
    # Handle missing methods (fill with 0)
    if 'zero_shot' not in pivot_data.columns:
        pivot_data['zero_shot'] = 0
    if 'zero_shot_cot' not in pivot_data.columns:
        pivot_data['zero_shot_cot'] = 0
    
    # Ensure consistent column order
    pivot_data = pivot_data[['zero_shot', 'zero_shot_cot']]
    
    # Add overall performance as an additional "dataset"
    overall_data = {}
    for _, row in method_acc.iterrows():
        overall_data[row['method']] = row['accuracy']
    
    # Add overall performance row to pivot data
    pivot_data.loc['Overall'] = [overall_data.get('zero_shot', 0), overall_data.get('zero_shot_cot', 0)]
    
    # Create the bar positions
    datasets = list(pivot_data.index)
    n_datasets = len(datasets)
    
    # Set up positions and colors
    x_pos = range(n_datasets)
    width = 0.35
    
    # Consistent colors for both dataset-specific and overall
    colors = ['#2E86AB', '#A23B72']  # Blue and Purple for both datasets and overall
    
    # Plot dataset-specific bars
    dataset_bars_zero = []
    dataset_bars_cot = []
    
    for i, dataset in enumerate(datasets[:-1]):  # All except 'Overall'
        zero_val = pivot_data.loc[dataset, 'zero_shot']
        cot_val = pivot_data.loc[dataset, 'zero_shot_cot']
        
        bar1 = ax.bar(i - width/2, zero_val, width, color=colors[0], alpha=0.85)
        bar2 = ax.bar(i + width/2, cot_val, width, color=colors[1], alpha=0.85)
        
        dataset_bars_zero.extend(bar1)
        dataset_bars_cot.extend(bar2)
    
    # Plot overall performance bars (last position) with different colors
    overall_idx = len(datasets) - 1
    zero_overall = pivot_data.loc['Overall', 'zero_shot']
    cot_overall = pivot_data.loc['Overall', 'zero_shot_cot']
    
    overall_bar1 = ax.bar(overall_idx - width/2, zero_overall, width, color=colors[0], alpha=0.85)
    overall_bar2 = ax.bar(overall_idx + width/2, cot_overall, width, color=colors[1], alpha=0.85)
    
    # Set title and labels with larger fonts
    ax.set_title('Performance Comparison: Zero Shot vs Zero Shot CoT', 
                fontsize=32, fontweight='bold', pad=35)
    ax.set_xlabel('Dataset', fontweight='bold', fontsize=26, labelpad=20)
    ax.set_ylabel('Accuracy', fontweight='bold', fontsize=26, labelpad=20)
    ax.set_ylim(0, 1.3)  # Increased upper limit to give more space for legend
    
    # Set x-axis labels and positions with larger fonts
    ax.set_xticks(x_pos)
    ax.set_xticklabels(datasets, fontsize=20)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', labelsize=20)
    
    # Create simplified legend with consistent colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[0], alpha=0.85, label='Zero Shot'),
        Patch(facecolor=colors[1], alpha=0.85, label='Zero Shot CoT')
    ]
    
    legend = ax.legend(handles=legend_elements, title='Method', title_fontsize=22, fontsize=20, 
                      loc='upper left', bbox_to_anchor=(0.02, 0.98))
    legend.get_title().set_fontweight('bold')
    
    # Grid styling
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels on all bars
    all_bars = dataset_bars_zero + dataset_bars_cot + list(overall_bar1) + list(overall_bar2)
    all_values = (list(pivot_data.loc[datasets[:-1], 'zero_shot']) + 
                 list(pivot_data.loc[datasets[:-1], 'zero_shot_cot']) + 
                 [zero_overall, cot_overall])
    
    for bar, value in zip(all_bars, all_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{value:.3f}', ha='center', va='bottom', 
               fontsize=18, fontweight='bold')
    
    # Add sample counts inside overall bars to avoid overlap
    for _, row in method_acc.iterrows():
        if row['method'] == 'zero_shot':
            bar = overall_bar1[0]
        else:
            bar = overall_bar2[0]
        
        # Place sample counts inside the bars (in the middle)
        bar_height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., bar_height * 0.5,
               f"({row['sum']}/{row['count']})", ha='center', va='center', 
               fontsize=12, fontweight='bold', style='italic', 
               color='white', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    # Adjust layout
    plt.tight_layout(pad=2.0)
    
    # Save chart with higher DPI for better quality
    plt.savefig(output_file, dpi=400, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✓ Combined chart saved to: {output_file}")

def generate_model_comparison_chart(df, output_file: str):
    """Generate detailed model comparison chart with enhanced fonts"""
    plt.style.use('default')
    sns.set_palette("Set2")
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
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
            method_label = "Zero Shot" if method == 'zero_shot' else "Zero Shot CoT"
            label = f"{method_label} - {model}"
            
            ax.bar(x_pos, y_values, width, label=label, alpha=0.8)
    
    # Enhanced title and labels with larger fonts
    ax.set_title('Detailed Model and Method Comparison by Dataset', 
                fontsize=26, fontweight='bold', pad=25)
    ax.set_xlabel('Dataset', fontweight='bold', fontsize=20, labelpad=15)
    ax.set_ylabel('Accuracy', fontweight='bold', fontsize=20, labelpad=15)
    ax.set_xticks([pos + width * (len(methods) * len(models) - 1) / 2 for pos in x])
    ax.set_xticklabels(datasets, rotation=45, fontsize=16)
    ax.set_ylim(0, 1.1)
    
    # Enhanced legend with larger fonts
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                      fontsize=16, title_fontsize=18)
    legend.get_title().set_fontweight('bold')
    
    # Enhanced grid and tick labels
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.tick_params(axis='y', labelsize=16)

    plt.tight_layout(pad=2.0)
    plt.savefig(output_file, dpi=400, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✓ Model comparison chart saved to: {output_file}")

def load_data_from_json(json_file: Path) -> pd.DataFrame:
    """Load benchmark data from JSON results file"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not data:
        raise ValueError(f"No data found in {json_file}")
    
    df = pd.DataFrame(data)
    print(f"✓ Loaded {len(df)} results from {json_file}")
    return df

def load_data_from_csv(csv_file: Path) -> pd.DataFrame:
    """Load benchmark data from CSV summary file"""
    df = pd.read_csv(csv_file)
    
    if df.empty:
        raise ValueError(f"No data found in {csv_file}")
    
    print(f"✓ Loaded {len(df)} results from {csv_file}")
    return df

def find_latest_results(results_dir: Path) -> Path:
    """Find the most recent benchmark results file"""
    # Look for JSON files first
    json_files = list(results_dir.glob("benchmark_results_*.json"))
    if json_files:
        latest = max(json_files, key=lambda p: p.stat().st_mtime)
        print(f"✓ Found latest JSON results: {latest}")
        return latest
    
    # Fall back to CSV files
    csv_files = list(results_dir.glob("benchmark_summary_*.csv"))
    if csv_files:
        latest = max(csv_files, key=lambda p: p.stat().st_mtime)
        print(f"✓ Found latest CSV results: {latest}")
        return latest
    
    raise FileNotFoundError(f"No benchmark result files found in {results_dir}")

def validate_data(df: pd.DataFrame):
    """Validate that the DataFrame has required columns"""
    required_cols = ['dataset', 'method', 'is_correct']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if df['method'].nunique() < 2:
        print("⚠️  Warning: Only one method found. Charts will still be generated but comparison may be limited.")
    
    print(f"✓ Data validation passed: {len(df)} results, {df['dataset'].nunique()} datasets, {df['method'].nunique()} methods")

def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison charts from benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate charts from specific JSON file
  python generate_charts.py results/benchmark_results_20250729_125508.json
  
  # Generate charts from specific CSV file
  python generate_charts.py results/benchmark_summary_20250729_125508.csv
  
  # Generate charts from latest results in directory
  python generate_charts.py --input results/ --latest
  
  # Custom output directory
  python generate_charts.py results/data.json --output custom_charts/
        """
    )
    
    parser.add_argument(
        'input_file', 
        nargs='?',
        help='Path to benchmark results file (JSON or CSV) or directory'
    )
    
    parser.add_argument(
        '--input', '-i',
        help='Input file or directory path'
    )
    
    parser.add_argument(
        '--latest',
        action='store_true',
        help='Use latest results file from input directory'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output directory for charts (default: same as input file directory)'
    )
    
    args = parser.parse_args()
    
    # Determine input file
    input_path = args.input_file or args.input
    if not input_path:
        parser.error("Please provide an input file or use --input")
    
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"❌ Error: Input path {input_path} does not exist")
        sys.exit(1)
    
    # Handle directory input with --latest flag
    if input_path.is_dir():
        if args.latest:
            try:
                input_file = find_latest_results(input_path)
            except FileNotFoundError as e:
                print(f"❌ Error: {e}")
                sys.exit(1)
        else:
            print(f"❌ Error: {input_path} is a directory. Use --latest flag or specify a file.")
            sys.exit(1)
    else:
        input_file = input_path
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = input_file.parent
    
    try:
        # Load data
        print(f"📊 Generating charts from: {input_file}")
        
        if input_file.suffix.lower() == '.json':
            df = load_data_from_json(input_file)
        elif input_file.suffix.lower() == '.csv':
            df = load_data_from_csv(input_file)
        else:
            raise ValueError(f"Unsupported file format: {input_file.suffix}")
        
        # Validate data
        validate_data(df)
        
        # Generate timestamp for output files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate main comparison chart
        main_chart = output_dir / f"benchmark_chart_{timestamp}.png"
        generate_comparison_chart(df, main_chart)
        
        # Generate model comparison chart if multiple models
        if 'model' in df.columns and df['model'].nunique() > 1:
            model_chart = output_dir / f"benchmark_chart_{timestamp}_models.png"
            generate_model_comparison_chart(df, model_chart)
        else:
            print("ℹ️  Skipping model comparison chart (single model or no model column)")
        
        print(f"\n🎉 Chart generation completed!")
        print(f"📁 Output directory: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()