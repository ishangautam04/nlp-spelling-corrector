#!/usr/bin/env python3
"""
Generate Comprehensive Comparative Plots for Spelling Correction Algorithms
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_results():
    """Load all result files"""
    results = {}
    
    # Load evaluation results
    try:
        with open('spelling_correction_evaluation_report.txt', 'r') as f:
            content = f.read()
            results['evaluation'] = content
    except FileNotFoundError:
        print("Warning: evaluation report not found")
    
    # Load adaptive ensemble results
    try:
        with open('adaptive_ensemble_results.json', 'r') as f:
            results['adaptive'] = json.load(f)
    except FileNotFoundError:
        print("Warning: adaptive ensemble results not found")
    
    return results

def parse_evaluation_report(content):
    """Parse evaluation report to extract data"""
    data = {
        'methods': [],
        'accuracies': [],
        'times': [],
        'error_types': defaultdict(dict)
    }
    
    # Parse overall accuracy
    lines = content.split('\n')
    in_main_table = False
    in_algorithm_strengths = False
    current_algorithm = None
    
    for line in lines:
        # Parse main results table
        if 'METHOD' in line and 'ACCURACY' in line:
            in_main_table = True
            continue
        
        if in_main_table and line.strip().startswith('-'):
            continue
        
        if in_main_table and line.strip():
            parts = line.split()
            if len(parts) >= 4 and parts[3] in ['individual', 'ensemble']:
                method = parts[0]
                try:
                    accuracy = float(parts[1])
                    time = float(parts[2])
                    data['methods'].append(method)
                    data['accuracies'].append(accuracy)
                    data['times'].append(time)
                except ValueError:
                    pass
            elif not any(char.isdigit() for char in line):
                in_main_table = False
        
        # Parse error type performance from algorithm strengths section
        if 'ALGORITHM STRENGTHS' in line:
            in_algorithm_strengths = True
            continue
        
        if in_algorithm_strengths:
            # Detect algorithm name
            if 'excels at:' in line.lower():
                current_algorithm = line.split()[0].lower()
            # Parse error type performance
            elif current_algorithm and '•' in line and ':' in line:
                parts = line.split(':')
                if len(parts) == 2:
                    error_type = parts[0].strip().replace('•', '').strip()
                    try:
                        accuracy = float(parts[1].strip())
                        data['error_types'][error_type][current_algorithm] = accuracy
                    except ValueError:
                        pass
    
    return data

def create_comparison_plots(eval_data, adaptive_data):
    """Create comprehensive comparison plots"""
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    
    # ==================== Plot 1: Overall Accuracy Comparison ====================
    ax1 = plt.subplot(2, 3, 1)
    
    methods = eval_data['methods']
    accuracies = [acc * 100 for acc in eval_data['accuracies']]
    
    # Add adaptive ensemble if available
    if adaptive_data:
        methods.append('adaptive_ensemble')
        accuracies.append(adaptive_data['accuracy'] * 100)
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    bars = ax1.bar(range(len(methods)), accuracies, color=colors[:len(methods)], alpha=0.8, edgecolor='black')
    
    ax1.set_xlabel('Algorithm', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Overall Accuracy Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels([m.replace('_', '\n') for m in methods], rotation=0, ha='center', fontsize=9)
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # ==================== Plot 2: Processing Time Comparison ====================
    ax2 = plt.subplot(2, 3, 2)
    
    times = eval_data['times'].copy()
    time_methods = eval_data['methods'].copy()
    
    # Add adaptive ensemble time if available (estimate based on slowest method)
    if adaptive_data:
        time_methods.append('adaptive_ensemble')
        times.append(max(eval_data['times']))  # Ensemble uses all methods
    
    print(f"DEBUG: time_methods length = {len(time_methods)}, times length = {len(times)}")
    print(f"DEBUG: time_methods = {time_methods}")
    print(f"DEBUG: times = {times}")
    
    colors_time = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    bars = ax2.barh(range(len(time_methods)), times, color=colors_time[:len(time_methods)], alpha=0.8, edgecolor='black')
    
    ax2.set_ylabel('Algorithm', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Processing Time (seconds)', fontsize=11, fontweight='bold')
    ax2.set_title('Processing Speed Comparison', fontsize=13, fontweight='bold')
    ax2.set_yticks(range(len(time_methods)))
    ax2.set_yticklabels(time_methods, fontsize=10)
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, time) in enumerate(zip(bars, times)):
        width = bar.get_width()
        ax2.text(width + 0.2, bar.get_y() + bar.get_height()/2.,
                f'{time:.1f}s', ha='left', va='center', fontsize=9, fontweight='bold')
    
    # ==================== Plot 3: Accuracy vs Speed Scatter ====================
    ax3 = plt.subplot(2, 3, 3)
    
    accs = eval_data['accuracies']
    times_plot = eval_data['times']
    methods_plot = eval_data['methods']
    
    colors_scatter = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for i, (acc, time, method) in enumerate(zip(accs, times_plot, methods_plot)):
        ax3.scatter(time, acc * 100, s=300, c=colors_scatter[i], alpha=0.7, 
                   edgecolors='black', linewidths=2, label=method)
        ax3.annotate(method, (time, acc * 100), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9, fontweight='bold')
    
    if adaptive_data:
        ensemble_time = max(eval_data['times'])
        ax3.scatter(ensemble_time, adaptive_data['accuracy'] * 100, s=400, c='#9b59b6', 
                   alpha=0.7, edgecolors='black', linewidths=2, marker='*', 
                   label='adaptive_ensemble')
        ax3.annotate('adaptive_ensemble', (ensemble_time, adaptive_data['accuracy'] * 100), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
    
    ax3.set_xlabel('Processing Time (seconds)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Accuracy vs Speed Trade-off', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-0.5, max(times_plot) + 1)
    ax3.set_ylim(30, 100)
    
    # Add quadrant labels
    ax3.axhline(y=80, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(x=3, color='gray', linestyle='--', alpha=0.5)
    ax3.text(0.5, 95, 'Fast &\nAccurate', fontsize=10, style='italic', alpha=0.6, ha='center')
    ax3.text(5.5, 95, 'Slow &\nAccurate', fontsize=10, style='italic', alpha=0.6, ha='center')
    
    # ==================== Plot 4: Error Type Performance Heatmap ====================
    ax4 = plt.subplot(2, 3, 4)
    
    # Prepare heatmap data
    error_types_list = ['keyboard', 'phonetic', 'character', 'simple_typo']
    methods_list = eval_data['methods']
    
    # Create matrix
    heatmap_data = []
    for error_type in error_types_list:
        row = []
        for method in methods_list:
            # Find accuracy for this combination from evaluation report
            acc = 0
            if error_type in eval_data['error_types']:
                acc = eval_data['error_types'][error_type].get(method, 0)
            row.append(acc * 100 if acc < 1 else acc)
        heatmap_data.append(row)
    
    # Create heatmap
    im = ax4.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=50, vmax=100)
    
    ax4.set_xticks(range(len(methods_list)))
    ax4.set_xticklabels(methods_list, rotation=45, ha='right', fontsize=9)
    ax4.set_yticks(range(len(error_types_list)))
    ax4.set_yticklabels(error_types_list, fontsize=10)
    ax4.set_title('Performance by Error Type (%)', fontsize=13, fontweight='bold')
    
    # Add text annotations
    for i in range(len(error_types_list)):
        for j in range(len(methods_list)):
            if heatmap_data[i][j] > 0:
                text = ax4.text(j, i, f'{heatmap_data[i][j]:.0f}',
                              ha="center", va="center", color="black", 
                              fontsize=9, fontweight='bold')
    
    plt.colorbar(im, ax=ax4, label='Accuracy (%)')
    
    # ==================== Plot 5: Adaptive Ensemble Weights ====================
    ax5 = plt.subplot(2, 3, 5)
    
    if adaptive_data and 'weights' in adaptive_data:
        weights_methods = list(adaptive_data['weights'].keys())
        weights_values = list(adaptive_data['weights'].values())
        
        colors_weights = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        bars = ax5.bar(range(len(weights_methods)), weights_values, 
                      color=colors_weights, alpha=0.8, edgecolor='black')
        
        ax5.set_xlabel('Algorithm', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Learned Weight', fontsize=11, fontweight='bold')
        ax5.set_title('Adaptive Ensemble: Learned Weights', fontsize=13, fontweight='bold')
        ax5.set_xticks(range(len(weights_methods)))
        ax5.set_xticklabels([m.replace('_', '\n') for m in weights_methods], 
                           rotation=0, ha='center', fontsize=9)
        ax5.grid(axis='y', alpha=0.3)
        ax5.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Equal weight')
        
        # Add value labels
        for bar, weight in zip(bars, weights_values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                    f'{weight:.3f}', ha='center', va='bottom', 
                    fontsize=9, fontweight='bold')
        
        ax5.legend(fontsize=9)
    else:
        ax5.text(0.5, 0.5, 'No adaptive ensemble\ndata available', 
                ha='center', va='center', fontsize=12, transform=ax5.transAxes)
        ax5.set_xticks([])
        ax5.set_yticks([])
    
    # ==================== Plot 6: Ranking Summary ====================
    ax6 = plt.subplot(2, 3, 6)
    
    # Prepare ranking data
    ranking_data = []
    for method, acc, time in zip(eval_data['methods'], eval_data['accuracies'], eval_data['times']):
        ranking_data.append({
            'method': method,
            'accuracy': acc * 100,
            'time': time,
            'score': (acc * 100) / (time + 0.1)  # Speed-accuracy score
        })
    
    # Sort by accuracy
    ranking_data.sort(key=lambda x: x['accuracy'], reverse=True)
    
    # Create ranking table
    y_pos = np.arange(len(ranking_data))
    
    # Clear axis
    ax6.axis('tight')
    ax6.axis('off')
    
    # Create table
    table_data = []
    table_data.append(['Rank', 'Algorithm', 'Accuracy', 'Time', 'Score'])
    
    for i, item in enumerate(ranking_data):
        rank = f"#{i+1}"
        method = item['method']
        acc = f"{item['accuracy']:.1f}%"
        time = f"{item['time']:.1f}s"
        score = f"{item['score']:.1f}"
        
        table_data.append([rank, method, acc, time, score])
    
    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.1, 0.3, 0.2, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows by rank
    colors_rank = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#95a5a6']
    for i in range(1, len(table_data)):
        for j in range(5):
            if i <= len(colors_rank):
                table[(i, j)].set_facecolor(colors_rank[i-1])
                table[(i, j)].set_alpha(0.3)
    
    ax6.set_title('Algorithm Rankings', fontsize=13, fontweight='bold', pad=20)
    
    # ==================== Final Layout ====================
    plt.suptitle('Comprehensive Spelling Correction Algorithm Comparison', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    return fig

def create_detailed_comparison_plot(eval_data, adaptive_data):
    """Create detailed error-type-specific comparison"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Detailed Error Type Analysis', fontsize=16, fontweight='bold')
    
    error_types = ['keyboard', 'phonetic', 'character', 'simple_typo']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for idx, error_type in enumerate(error_types):
        ax = axes[idx // 2, idx % 2]
        
        if error_type in eval_data['error_types']:
            methods = list(eval_data['error_types'][error_type].keys())
            accuracies = [eval_data['error_types'][error_type][m] * 100 
                         if eval_data['error_types'][error_type][m] < 1 
                         else eval_data['error_types'][error_type][m] 
                         for m in methods]
            
            bars = ax.bar(range(len(methods)), accuracies, 
                         color=colors[:len(methods)], alpha=0.8, edgecolor='black')
            
            ax.set_xlabel('Algorithm', fontsize=11, fontweight='bold')
            ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
            ax.set_title(f'{error_type.upper()} Errors', fontsize=13, fontweight='bold')
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels([m.replace('_', '\n') for m in methods], 
                              rotation=45, ha='right', fontsize=9)
            ax.set_ylim(0, 100)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                       f'{acc:.1f}%', ha='center', va='bottom', 
                       fontsize=9, fontweight='bold')
            
            # Highlight best performer
            best_idx = accuracies.index(max(accuracies))
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(3)
    
    plt.tight_layout()
    return fig

def main():
    """Generate all comparison plots"""
    print("="*70)
    print("GENERATING COMPARATIVE PLOTS")
    print("="*70)
    
    # Load results
    print("\nLoading results...")
    results = load_results()
    
    if 'evaluation' not in results:
        print("Error: No evaluation results found!")
        print("Please run 'python evaluate_spelling_methods.py' first")
        return
    
    # Parse evaluation data
    print("Parsing evaluation data...")
    eval_data = parse_evaluation_report(results['evaluation'])
    
    print(f"Found {len(eval_data['methods'])} algorithms")
    print(f"Methods: {eval_data['methods']}")
    
    # Get adaptive ensemble data
    adaptive_data = results.get('adaptive', None)
    if adaptive_data:
        print(f"Adaptive ensemble accuracy: {adaptive_data['accuracy']:.3f}")
    
    # Create comprehensive comparison plots
    print("\nGenerating comprehensive comparison plots...")
    fig1 = create_comparison_plots(eval_data, adaptive_data)
    
    filename1 = 'comprehensive_algorithm_comparison.png'
    fig1.savefig(filename1, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename1}")
    
    # Create detailed error type plots
    print("Generating detailed error type analysis...")
    fig2 = create_detailed_comparison_plot(eval_data, adaptive_data)
    
    filename2 = 'detailed_error_type_analysis.png'
    fig2.savefig(filename2, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename2}")
    
    print("\n" + "="*70)
    print("PLOT GENERATION COMPLETE!")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"1. {filename1}")
    print(f"2. {filename2}")
    print("\nYou can now view these plots to compare algorithm performance!")
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    main()
