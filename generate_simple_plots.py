#!/usr/bin/env python3
"""
Generate Two Key Comparative Plots for Spelling Correction Algorithms
1. Overall Accuracy Comparison
2. Adaptive Ensemble Learned Weights
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_results():
    """Load result files"""
    results = {}
    
    # Load evaluation results
    try:
        with open('spelling_correction_evaluation_report.txt', 'r') as f:
            content = f.read()
            results['evaluation'] = content
    except FileNotFoundError:
        print("Warning: evaluation report not found")
        results['evaluation'] = None
    
    # Load adaptive ensemble results
    try:
        with open('adaptive_ensemble_results.json', 'r') as f:
            results['adaptive'] = json.load(f)
    except FileNotFoundError:
        print("Warning: adaptive ensemble results not found")
        results['adaptive'] = None
    
    return results

def parse_evaluation_report(content):
    """Parse evaluation report to extract accuracy data"""
    data = {
        'methods': [],
        'accuracies': []
    }
    
    if not content:
        return data
    
    lines = content.split('\n')
    in_main_table = False
    
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
                    data['methods'].append(method)
                    data['accuracies'].append(accuracy)
                except ValueError:
                    pass
            elif not any(char.isdigit() for char in line):
                in_main_table = False
    
    return data

def create_plots(eval_data, adaptive_data):
    """Create the two requested plots"""
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ==================== Plot 1: Overall Accuracy Comparison ====================
    
    methods = eval_data['methods'].copy()
    accuracies = [acc * 100 for acc in eval_data['accuracies']]
    
    # Add adaptive ensemble if available
    if adaptive_data:
        methods.append('Adaptive\nEnsemble')
        accuracies.append(adaptive_data['accuracy'] * 100)
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    bars = ax1.bar(range(len(methods)), accuracies, color=colors[:len(methods)], 
                   alpha=0.8, edgecolor='black', linewidth=2)
    
    ax1.set_xlabel('Algorithm', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Overall Accuracy Comparison', fontsize=15, fontweight='bold', pad=20)
    ax1.set_xticks(range(len(methods)))
    
    # Format method names for display
    display_names = []
    for m in methods:
        if '\n' in m:
            display_names.append(m)
        else:
            display_names.append(m.replace('_', '\n').title())
    
    ax1.set_xticklabels(display_names, rotation=0, ha='center', fontsize=11)
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                f'{acc:.1f}%', ha='center', va='bottom', 
                fontsize=12, fontweight='bold')
    
    # Highlight best performer
    best_idx = accuracies.index(max(accuracies))
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(4)
    
    # Add a star to the best performer
    best_x = bars[best_idx].get_x() + bars[best_idx].get_width()/2.
    best_y = bars[best_idx].get_height()
    ax1.scatter(best_x, best_y + 5, s=500, marker='*', color='gold', 
               edgecolors='black', linewidths=2, zorder=5)
    
    # ==================== Plot 2: Adaptive Ensemble Learned Weights ====================
    
    if adaptive_data and 'weights' in adaptive_data:
        weights_methods = list(adaptive_data['weights'].keys())
        weights_values = list(adaptive_data['weights'].values())
        
        # Format method names
        display_weight_names = [m.replace('_', '\n').title() for m in weights_methods]
        
        colors_weights = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        bars2 = ax2.bar(range(len(weights_methods)), weights_values, 
                       color=colors_weights, alpha=0.8, edgecolor='black', linewidth=2)
        
        ax2.set_xlabel('Algorithm', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Learned Weight', fontsize=13, fontweight='bold')
        ax2.set_title('Adaptive Ensemble: Learned Weights', fontsize=15, fontweight='bold', pad=20)
        ax2.set_xticks(range(len(weights_methods)))
        ax2.set_xticklabels(display_weight_names, rotation=0, ha='center', fontsize=11)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add reference line at 1.0
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.6, linewidth=2, label='Equal weight (1.0)')
        
        # Add value labels
        for bar, weight in zip(bars2, weights_values):
            height = bar.get_height()
            y_offset = 0.04 if height > 1.0 else -0.08
            va = 'bottom' if height > 1.0 else 'top'
            ax2.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                    f'{weight:.3f}', ha='center', va=va, 
                    fontsize=12, fontweight='bold')
        
        # Highlight highest and lowest weights
        max_idx = weights_values.index(max(weights_values))
        min_idx = weights_values.index(min(weights_values))
        
        bars2[max_idx].set_edgecolor('green')
        bars2[max_idx].set_linewidth(4)
        
        bars2[min_idx].set_edgecolor('orange')
        bars2[min_idx].set_linewidth(4)
        
        ax2.legend(fontsize=11, loc='upper right')
        ax2.set_ylim(0, max(weights_values) * 1.2)
        
    else:
        ax2.text(0.5, 0.5, 'No adaptive ensemble\ndata available', 
                ha='center', va='center', fontsize=14, transform=ax2.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title('Adaptive Ensemble: Learned Weights', fontsize=15, fontweight='bold', pad=20)
    
    # ==================== Final Layout ====================
    plt.suptitle('Spelling Correction Algorithm Performance', 
                fontsize=17, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig

def main():
    """Generate the two plots"""
    print("="*70)
    print("GENERATING COMPARISON PLOTS")
    print("="*70)
    
    # Load results
    print("\nLoading results...")
    results = load_results()
    
    if not results['evaluation']:
        print("Error: No evaluation results found!")
        print("Please run 'python evaluate_spelling_methods.py' first")
        return
    
    # Parse evaluation data
    print("Parsing evaluation data...")
    eval_data = parse_evaluation_report(results['evaluation'])
    
    if not eval_data['methods']:
        print("Error: Could not parse evaluation data!")
        return
    
    print(f"Found {len(eval_data['methods'])} algorithms")
    print(f"Methods: {eval_data['methods']}")
    
    # Get adaptive ensemble data
    adaptive_data = results.get('adaptive', None)
    if adaptive_data:
        print(f"Adaptive ensemble accuracy: {adaptive_data['accuracy']:.3f}")
        print(f"Learned weights: {adaptive_data.get('weights', {})}")
    else:
        print("No adaptive ensemble data found")
    
    # Create plots
    print("\nGenerating plots...")
    fig = create_plots(eval_data, adaptive_data)
    
    filename = 'algorithm_comparison_plots.png'
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    
    print("\n" + "="*70)
    print("PLOT GENERATION COMPLETE!")
    print("="*70)
    print(f"\nGenerated file: {filename}")
    print("\nThe plot shows:")
    print("  1. Overall Accuracy Comparison (left)")
    print("  2. Adaptive Ensemble Learned Weights (right)")
    print("\nYou can now view the plot to see algorithm performance!")
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    main()
