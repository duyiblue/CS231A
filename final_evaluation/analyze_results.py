#!/usr/bin/env python3

import numpy as np
from pathlib import Path
import argparse

# Optional visualization dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from scipy import stats
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Visualization libraries not available ({e})")
    print("Install with: pip install matplotlib seaborn pandas scipy")
    VISUALIZATION_AVAILABLE = False

def load_results(results_file: str) -> dict:
    """Load the evaluation results from numpy file."""
    return np.load(results_file, allow_pickle=True).item()

def analyze_results(results_dict: dict):
    """Analyze and print comprehensive statistics."""
    scores = results_dict['scores']
    object_names = results_dict['object_names']
    grasp_types = results_dict['grasp_types']
    
    print("="*80)
    print("COMPREHENSIVE GRASP EVALUATION ANALYSIS")
    print("="*80)
    
    # Basic statistics
    print(f"\nDataset Overview:")
    print(f"  Total objects: {len(object_names)}")
    print(f"  Successful evaluations: {results_dict['successful_evaluations']}")
    print(f"  Total attempts: {results_dict['total_evaluations']}")
    print(f"  Success rate: {results_dict['successful_evaluations']/results_dict['total_evaluations']*100:.1f}%")
    
    # Per-method statistics
    print(f"\nPer-Method Statistics:")
    print("-" * 50)
    
    method_stats = {}
    for j, method in enumerate(grasp_types):
        valid_scores = scores[:, j][~np.isnan(scores[:, j])]
        if len(valid_scores) > 0:
            stats_dict = {
                'count': len(valid_scores),
                'mean': np.mean(valid_scores),
                'std': np.std(valid_scores),
                'min': np.min(valid_scores),
                'max': np.max(valid_scores),
                'median': np.median(valid_scores),
                'q25': np.percentile(valid_scores, 25),
                'q75': np.percentile(valid_scores, 75)
            }
            method_stats[method] = stats_dict
            
            print(f"{method.upper():>15}:")
            print(f"  Count: {stats_dict['count']:>3}")
            print(f"  Mean:  {stats_dict['mean']:.3f} ± {stats_dict['std']:.3f}")
            print(f"  Range: [{stats_dict['min']:.3f}, {stats_dict['max']:.3f}]")
            print(f"  Median: {stats_dict['median']:.3f}")
            print(f"  IQR:   [{stats_dict['q25']:.3f}, {stats_dict['q75']:.3f}]")
            print()
    
    # Comparative analysis
    print("\nComparative Analysis:")
    print("-" * 50)
    
    # Compare methods pairwise
    methods_with_data = [m for m in grasp_types if m in method_stats]
    
    for i, method1 in enumerate(methods_with_data):
        for method2 in methods_with_data[i+1:]:
            # Find objects that have both methods
            method1_idx = grasp_types.index(method1)
            method2_idx = grasp_types.index(method2)
            
            valid_mask = ~np.isnan(scores[:, method1_idx]) & ~np.isnan(scores[:, method2_idx])
            if np.sum(valid_mask) > 1:
                scores1 = scores[valid_mask, method1_idx]
                scores2 = scores[valid_mask, method2_idx]
                
                # Count improvements
                improvements = np.sum(scores2 > scores1)
                total_pairs = len(scores1)
                
                print(f"{method1.upper()} vs {method2.upper()} (n={total_pairs}):")
                print(f"  Mean difference: {np.mean(scores2 - scores1):+.3f}")
                print(f"  Improvements: {improvements}/{total_pairs} ({improvements/total_pairs*100:.1f}%)")
                
                # Paired t-test (if scipy available)
                if VISUALIZATION_AVAILABLE:
                    try:
                        t_stat, p_value = stats.ttest_rel(scores1, scores2)
                        print(f"  Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")
                    except:
                        print(f"  Paired t-test: not available")
                print()
    
    return method_stats

def create_visualizations(results_dict: dict, output_dir: str = "plots"):
    """Create comprehensive visualizations of the results."""
    if not VISUALIZATION_AVAILABLE:
        print("Skipping visualizations - required libraries not available")
        return
        
    scores = results_dict['scores']
    object_names = results_dict['object_names']
    grasp_types = results_dict['grasp_types']
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Box plot comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Prepare data for box plot
    plot_data = []
    for j, method in enumerate(grasp_types):
        valid_scores = scores[:, j][~np.isnan(scores[:, j])]
        if len(valid_scores) > 0:
            for score in valid_scores:
                plot_data.append({'Method': method, 'y_PGS': score})
    
    if plot_data:
        df = pd.DataFrame(plot_data)
        sns.boxplot(data=df, x='Method', y='y_PGS', ax=ax)
        ax.set_title('Grasp Success Rate (y_PGS) by Method', fontsize=14, fontweight='bold')
        ax.set_ylabel('y_PGS Score', fontsize=12)
        ax.set_xlabel('Grasp Method', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / 'boxplot_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Scatter plot matrix for pairwise comparisons
    methods_with_data = []
    method_indices = []
    for j, method in enumerate(grasp_types):
        if np.any(~np.isnan(scores[:, j])):
            methods_with_data.append(method)
            method_indices.append(j)
    
    if len(methods_with_data) >= 2:
        n_methods = len(methods_with_data)
        fig, axes = plt.subplots(n_methods, n_methods, figsize=(4*n_methods, 4*n_methods))
        
        if n_methods == 1:
            axes = np.array([[axes]])
        elif n_methods == 2:
            axes = axes.reshape(2, 2)
        
        for i, method1 in enumerate(methods_with_data):
            for j, method2 in enumerate(methods_with_data):
                ax = axes[i, j]
                
                if i == j:
                    # Diagonal: histogram
                    method_idx = method_indices[i]
                    valid_scores = scores[:, method_idx][~np.isnan(scores[:, method_idx])]
                    if len(valid_scores) > 0:
                        ax.hist(valid_scores, bins=20, alpha=0.7, edgecolor='black')
                        ax.set_title(f'{method1}', fontweight='bold')
                else:
                    # Off-diagonal: scatter plot
                    method1_idx = method_indices[i]
                    method2_idx = method_indices[j]
                    
                    valid_mask = ~np.isnan(scores[:, method1_idx]) & ~np.isnan(scores[:, method2_idx])
                    if np.sum(valid_mask) > 0:
                        x_vals = scores[valid_mask, method2_idx]
                        y_vals = scores[valid_mask, method1_idx]
                        
                        ax.scatter(x_vals, y_vals, alpha=0.6, s=50)
                        
                        # Add diagonal line
                        min_val = min(np.min(x_vals), np.min(y_vals))
                        max_val = max(np.max(x_vals), np.max(y_vals))
                        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                        
                        # Correlation
                        if len(x_vals) > 1:
                            corr = np.corrcoef(x_vals, y_vals)[0, 1]
                            ax.text(0.05, 0.95, f'r={corr:.3f}', transform=ax.transAxes, 
                                   bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
                
                if i == n_methods - 1:
                    ax.set_xlabel(method2, fontsize=10)
                if j == 0:
                    ax.set_ylabel(method1, fontsize=10)
        
        plt.suptitle('Pairwise Method Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'pairwise_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Performance per object (top 20 objects)
    if len(object_names) > 0:
        # Calculate mean performance per object (across available methods)
        object_means = np.nanmean(scores, axis=1)
        sorted_indices = np.argsort(object_means)[::-1]  # Sort descending
        
        top_n = min(20, len(object_names))
        top_indices = sorted_indices[:top_n]
        
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))
        
        x_pos = np.arange(top_n)
        width = 0.25
        
        for j, method in enumerate(grasp_types):
            method_scores = scores[top_indices, j]
            offset = (j - 1) * width
            
            # Only plot if there are valid scores
            if np.any(~np.isnan(method_scores)):
                ax.bar(x_pos + offset, method_scores, width, 
                      label=method, alpha=0.8)
        
        ax.set_xlabel('Objects (Top 20 by Average Performance)', fontsize=12)
        ax.set_ylabel('y_PGS Score', fontsize=12)
        ax.set_title('Performance Comparison Across Top Objects', fontsize=14, fontweight='bold')
        
        # Truncate object names for readability
        truncated_names = [name[:30] + '...' if len(name) > 30 else name 
                          for name in [object_names[i] for i in top_indices]]
        ax.set_xticks(x_pos)
        ax.set_xticklabels(truncated_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'top_objects_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\nVisualization plots saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze grasp evaluation results')
    parser.add_argument('results_file', type=str, help='Path to the results .npy file')
    parser.add_argument('--output_dir', type=str, default='plots', 
                       help='Directory to save plots')
    parser.add_argument('--no_plots', action='store_true', 
                       help='Skip creating plots')
    
    args = parser.parse_args()
    
    if not Path(args.results_file).exists():
        print(f"Error: Results file not found: {args.results_file}")
        return
    
    # Load and analyze results
    results_dict = load_results(args.results_file)
    method_stats = analyze_results(results_dict)
    
    # Create visualizations
    if not args.no_plots:
        if VISUALIZATION_AVAILABLE:
            try:
                create_visualizations(results_dict, args.output_dir)
            except Exception as e:
                print(f"Warning: Could not create plots: {e}")
                print("Analysis completed without visualization.")
        else:
            print("Skipping plots - visualization libraries not installed")
            print("To enable plots, install: pip install matplotlib seaborn pandas scipy")

if __name__ == "__main__":
    main() 