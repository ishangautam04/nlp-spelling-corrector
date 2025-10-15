#!/usr/bin/env python3
"""
Generate Proper Flowchart with Standard Shapes
Traditional flowchart with rectangles, diamonds, circles, parallelograms
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon, Wedge
import numpy as np

def create_simple_flowchart():
    """Create proper flowchart with standard shapes"""
    
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Title
    ax.text(5, 13.3, 'Spelling Correction System Flowchart', 
            ha='center', fontsize=16, fontweight='bold')
    
    # ============ START (Circle/Oval) ============
    start_circle = mpatches.Ellipse((5, 12.5), 1.5, 0.5, 
                                    facecolor='white', edgecolor='black', linewidth=2)
    ax.add_patch(start_circle)
    ax.text(5, 12.5, 'START', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Arrow down
    ax.annotate('', xy=(5, 11.8), xytext=(5, 12.25),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # ============ INPUT (Parallelogram) ============
    input_para = Polygon([(3.2, 11.8), (6.8, 11.8), (6.3, 11.1), (2.7, 11.1)],
                         facecolor='white', edgecolor='black', linewidth=2)
    ax.add_patch(input_para)
    ax.text(4.75, 11.45, 'Input: Misspelled Text', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    # Arrow down
    ax.annotate('', xy=(5, 10.6), xytext=(5, 11.1),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # ============ PROCESS BOX 1: Tokenization (Rectangle) ============
    rect1 = Rectangle((3.5, 10), 3, 0.6, facecolor='white', 
                     edgecolor='black', linewidth=2)
    ax.add_patch(rect1)
    ax.text(5, 10.3, 'Tokenization &', ha='center', fontsize=10, fontweight='bold')
    ax.text(5, 10.1, 'Normalization', ha='center', fontsize=9)
    
    # Arrow
    ax.annotate('', xy=(5, 9.5), xytext=(5, 10),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # ============ PROCESS BOX 2: Error Detection (Rectangle) ============
    rect2 = Rectangle((3.5, 8.9), 3, 0.6, facecolor='white', 
                     edgecolor='black', linewidth=2)
    ax.add_patch(rect2)
    ax.text(5, 9.2, 'Error Detection', ha='center', fontsize=10, fontweight='bold')
    
    # Arrow
    ax.annotate('', xy=(5, 8.4), xytext=(5, 8.9),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # ============ DECISION DIAMOND: Multiple Algorithms? ============
    diamond_points = [(5, 8.3), (3.5, 7.5), (5, 6.7), (6.5, 7.5)]
    diamond = Polygon(diamond_points, facecolor='white', edgecolor='black', linewidth=2)
    ax.add_patch(diamond)
    ax.text(5, 7.6, 'Apply', ha='center', fontsize=10, fontweight='bold')
    ax.text(5, 7.4, 'Multiple', ha='center', fontsize=9)
    ax.text(5, 7.2, 'Algorithms?', ha='center', fontsize=9)
    
    # Arrow down (YES path)
    ax.annotate('', xy=(5, 6.3), xytext=(5, 6.7),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(5.3, 6.5, 'YES', fontsize=9, fontweight='bold')
    
    # ============ FOUR ALGORITHM BOXES (Rectangles) ============
    rect_algos = Rectangle((1, 5.5), 8, 0.8, facecolor='lightgray', 
                           edgecolor='black', linewidth=1, linestyle='--')
    ax.add_patch(rect_algos)
    ax.text(5, 6.1, 'Parallel Algorithm Execution', ha='center', fontsize=10, fontweight='bold')
    
    algorithms = [
        ('PySpellChecker\n(92.6%)', 1.8),
        ('AutoCorrect\n(91.2%)', 3.6),
        ('Frequency\n(75.3%)', 5.4),
        ('Levenshtein\n(64.2%)', 7.2)
    ]
    
    for name, x in algorithms:
        rect = Rectangle((x-0.6, 5.5), 1.2, 0.6, facecolor='white', 
                        edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, 5.8, name, ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Arrows from algorithms converging
    for _, x in algorithms:
        ax.annotate('', xy=(5, 4.9), xytext=(x, 5.5),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # ============ PROCESS BOX: Compare Results (Rectangle) ============
    rect_compare = Rectangle((3.5, 4.3), 3, 0.6, facecolor='white', 
                             edgecolor='black', linewidth=2)
    ax.add_patch(rect_compare)
    ax.text(5, 4.6, 'Compare Results', ha='center', fontsize=10, fontweight='bold')
    
    # Arrow
    ax.annotate('', xy=(5, 3.8), xytext=(5, 4.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # ============ PROCESS BOX: Calculate Metrics (Rectangle) ============
    rect_metrics = Rectangle((3.5, 3.2), 3, 0.6, facecolor='white', 
                            edgecolor='black', linewidth=2)
    ax.add_patch(rect_metrics)
    ax.text(5, 3.5, 'Calculate Metrics', ha='center', fontsize=10, fontweight='bold')
    ax.text(5, 3.32, '(Accuracy, Speed)', ha='center', fontsize=8)
    
    # Arrow
    ax.annotate('', xy=(5, 2.7), xytext=(5, 3.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # ============ DECISION DIAMOND: Best Algorithm Selected? ============
    diamond2_points = [(5, 2.6), (3.5, 1.9), (5, 1.2), (6.5, 1.9)]
    diamond2 = Polygon(diamond2_points, facecolor='white', edgecolor='black', linewidth=2)
    ax.add_patch(diamond2)
    ax.text(5, 2.0, 'Best', ha='center', fontsize=10, fontweight='bold')
    ax.text(5, 1.8, 'Algorithm', ha='center', fontsize=9)
    ax.text(5, 1.6, 'Selected?', ha='center', fontsize=9)
    
    # Arrow down (YES path)
    ax.annotate('', xy=(5, 0.8), xytext=(5, 1.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(5.3, 1.0, 'YES', fontsize=9, fontweight='bold')
    
    # ============ OUTPUT (Parallelogram) ============
    output_para = Polygon([(3.2, 0.8), (6.8, 0.8), (6.3, 0.2), (2.7, 0.2)],
                         facecolor='white', edgecolor='black', linewidth=2)
    ax.add_patch(output_para)
    ax.text(4.75, 0.5, 'Output: Corrected Text', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    # Side note box
    note_rect = Rectangle((0.3, 6.5), 1.8, 2, facecolor='white', 
                          edgecolor='black', linewidth=1, linestyle='--')
    ax.add_patch(note_rect)
    ax.text(1.2, 8.3, 'Metrics:', ha='center', fontsize=9, fontweight='bold')
    ax.text(0.4, 8.0, '• Accuracy', ha='left', fontsize=8)
    ax.text(0.4, 7.7, '• Speed', ha='left', fontsize=8)
    ax.text(0.4, 7.4, '• Error Type', ha='left', fontsize=8)
    ax.text(0.4, 7.1, '• Comparison', ha='left', fontsize=8)
    ax.text(0.4, 6.8, '• Ranking', ha='left', fontsize=8)
    
    # Dataset info at bottom
    ax.text(5, 0.7, 'Dataset: 433 samples, 4 error types', 
            ha='center', fontsize=9, style='italic')
    ax.text(5, 0.4, '(Keyboard, Phonetic, Character, Simple Typo)', 
            ha='center', fontsize=8, style='italic')
    
    plt.tight_layout()
    return fig

def create_algorithm_comparison_architecture():
    """Create simplified architecture focusing on individual algorithm comparison"""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Spelling Correction System: Algorithm Comparison Architecture', 
            ha='center', fontsize=18, fontweight='bold')
    
    # ==================== Layer 1: Input Processing ====================
    rect1 = FancyBboxPatch((2.5, 8), 5, 0.9, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='#BBDEFB', linewidth=2.5)
    ax.add_patch(rect1)
    ax.text(5, 8.6, 'Layer 1: Input Processing', ha='center', fontsize=13, fontweight='bold')
    ax.text(5, 8.3, 'Tokenization | Normalization | Error Detection', ha='center', fontsize=10)
    
    # Arrow down
    ax.annotate('', xy=(5, 7.6), xytext=(5, 8),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    
    # ==================== Layer 2: Correction Algorithms ====================
    rect2 = FancyBboxPatch((0.5, 4.8), 9, 2.6, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='#C5E1A5', linewidth=2.5)
    ax.add_patch(rect2)
    ax.text(5, 7.2, 'Layer 2: Spelling Correction Algorithms (Parallel Processing)', 
            ha='center', fontsize=13, fontweight='bold')
    
    # Four algorithm boxes with details
    algorithms = [
        ('PySpellChecker', 'Dictionary-based', '92.6%', '#4CAF50', 1.5, 5.3),
        ('AutoCorrect', 'Pattern-based', '91.2%', '#2196F3', 3.5, 5.3),
        ('Frequency-based', 'Statistical', '75.3%', '#FF9800', 5.5, 5.3),
        ('Levenshtein', 'Edit Distance', '64.2%', '#F44336', 7.5, 5.3)
    ]
    
    for name, method, accuracy, color, x, y in algorithms:
        # Algorithm box
        box = FancyBboxPatch((x-0.8, y), 1.6, 1.5, boxstyle="round,pad=0.08",
                            edgecolor='black', facecolor=color, linewidth=2, alpha=0.7)
        ax.add_patch(box)
        
        # Algorithm name
        ax.text(x, y+1.2, name, ha='center', va='center', 
                fontsize=10, fontweight='bold', color='white')
        
        # Method type
        ax.text(x, y+0.85, method, ha='center', va='center', 
                fontsize=8, color='white', style='italic')
        
        # Accuracy badge
        badge = FancyBboxPatch((x-0.4, y+0.35), 0.8, 0.35, boxstyle="round,pad=0.03",
                              edgecolor='white', facecolor='white', linewidth=2)
        ax.add_patch(badge)
        ax.text(x, y+0.52, f'Acc: {accuracy}', ha='center', va='center', 
                fontsize=9, fontweight='bold', color='black')
        
        # Speed indicator (below box)
        if name == 'AutoCorrect':
            ax.text(x, y-0.2, '⚡ Fastest (1.0s)', ha='center', fontsize=8, 
                   fontweight='bold', color='green')
        elif name == 'PySpellChecker':
            ax.text(x, y-0.2, '🐌 Slower (5.4s)', ha='center', fontsize=8, 
                   fontweight='bold', color='orange')
        elif name == 'Frequency-based':
            ax.text(x, y-0.2, '🐌 Slower (5.3s)', ha='center', fontsize=8, 
                   fontweight='bold', color='orange')
        else:  # Levenshtein
            ax.text(x, y-0.2, '🐌 Slowest (5.5s)', ha='center', fontsize=8, 
                   fontweight='bold', color='red')
        
        # Arrow from layer 1 to each algorithm
        arrow = FancyArrowPatch((5, 7.6), (x, 6.8),
                               arrowstyle='->', mutation_scale=20, linewidth=2, 
                               color='gray', alpha=0.6)
        ax.add_patch(arrow)
    
    # ==================== Layer 3: Result Comparison & Selection ====================
    
    # Convergence arrows
    for x, _ in [(1.5, 5.3), (3.5, 5.3), (5.5, 5.3), (7.5, 5.3)]:
        arrow = FancyArrowPatch((x, 5.3), (5, 3.8),
                               arrowstyle='->', mutation_scale=20, linewidth=2, 
                               color='gray', alpha=0.6)
        ax.add_patch(arrow)
    
    rect3 = FancyBboxPatch((2, 2.8), 6, 1, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='#FFE082', linewidth=2.5)
    ax.add_patch(rect3)
    ax.text(5, 3.6, 'Layer 3: Result Comparison & Best Selection', 
            ha='center', fontsize=13, fontweight='bold')
    ax.text(5, 3.25, 'Compare Outputs | Analyze Performance | Select Best Result', 
            ha='center', fontsize=10)
    
    # Arrow down
    ax.annotate('', xy=(5, 2.4), xytext=(5, 2.8),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    
    # ==================== Layer 4: Output Generation ====================
    rect4 = FancyBboxPatch((2.5, 1.5), 5, 0.8, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='#C8E6C9', linewidth=2.5)
    ax.add_patch(rect4)
    ax.text(5, 2, 'Layer 4: Corrected Text Output', 
            ha='center', fontsize=13, fontweight='bold')
    ax.text(5, 1.75, 'Final Corrected Spelling', ha='center', fontsize=10)
    
    # ==================== Evaluation Module (Side) ====================
    eval_box = FancyBboxPatch((8.2, 1.5), 1.6, 2.3, boxstyle="round,pad=0.08",
                             edgecolor='red', facecolor='#FFCDD2', linewidth=2.5, linestyle='--')
    ax.add_patch(eval_box)
    ax.text(9, 3.6, 'Evaluation', ha='center', fontsize=11, fontweight='bold')
    ax.text(9, 3.35, 'Metrics', ha='center', fontsize=11, fontweight='bold')
    ax.text(8.35, 3.0, '• Accuracy', ha='left', fontsize=9)
    ax.text(8.35, 2.75, '• Speed', ha='left', fontsize=9)
    ax.text(8.35, 2.5, '• Error Type', ha='left', fontsize=9)
    ax.text(8.35, 2.25, '• Comparison', ha='left', fontsize=9)
    ax.text(8.35, 2.0, '• Analysis', ha='left', fontsize=9)
    ax.text(8.35, 1.75, '• Ranking', ha='left', fontsize=9)
    
    # Feedback arrow from evaluation to algorithms
    ax.annotate('', xy=(7.5, 6.5), xytext=(8.2, 2.5),
                arrowprops=dict(arrowstyle='<-', lw=2, color='red', linestyle='--'))
    ax.text(7.8, 4.5, 'Feedback', fontsize=9, color='red', fontweight='bold', rotation=55)
    
    # ==================== Performance Summary Box ====================
    summary_box = FancyBboxPatch((0.3, 0.2), 3.5, 1.1, boxstyle="round,pad=0.08",
                                edgecolor='blue', facecolor='#E3F2FD', linewidth=2)
    ax.add_patch(summary_box)
    ax.text(2.05, 1.15, 'Performance Summary', ha='center', fontsize=11, fontweight='bold')
    ax.text(0.45, 0.9, '🥇 Best Accuracy: PySpellChecker (92.6%)', ha='left', fontsize=8.5)
    ax.text(0.45, 0.65, '⚡ Fastest: AutoCorrect (1.0s)', ha='left', fontsize=8.5)
    ax.text(0.45, 0.4, '⚖️ Best Balance: AutoCorrect (91.2%, 1.0s)', ha='left', fontsize=8.5)
    
    # ==================== Key Strengths Box ====================
    strengths_box = FancyBboxPatch((4.2, 0.2), 3.5, 1.1, boxstyle="round,pad=0.08",
                                  edgecolor='green', facecolor='#E8F5E9', linewidth=2)
    ax.add_patch(strengths_box)
    ax.text(5.95, 1.15, 'Algorithm Strengths', ha='center', fontsize=11, fontweight='bold')
    ax.text(4.35, 0.9, '✓ Keyboard Errors: PySpellChecker (98.4%)', ha='left', fontsize=8.5)
    ax.text(4.35, 0.65, '✓ Phonetic Errors: AutoCorrect (76.9%)', ha='left', fontsize=8.5)
    ax.text(4.35, 0.4, '✓ Simple Typos: AutoCorrect (96.0%)', ha='left', fontsize=8.5)
    
    # ==================== Legend ====================
    legend_elements = [
        mpatches.Patch(facecolor='#4CAF50', edgecolor='black', label='Best Accuracy', alpha=0.7),
        mpatches.Patch(facecolor='#2196F3', edgecolor='black', label='Best Speed', alpha=0.7),
        mpatches.Patch(facecolor='#FF9800', edgecolor='black', label='Moderate', alpha=0.7),
        mpatches.Patch(facecolor='#F44336', edgecolor='black', label='Baseline', alpha=0.7),
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
             frameon=True, fancybox=True, shadow=True, title='Performance Level')
    
    # ==================== Bottom Note ====================
    ax.text(5, -0.3, 'Data Flow: Input → Parallel Processing → Comparison → Output',
            ha='center', fontsize=10, style='italic', color='gray', fontweight='bold')
    ax.text(5, -0.55, 'Evaluated on 433 samples | 4 Error Types: Keyboard, Phonetic, Character, Simple Typo',
            ha='center', fontsize=9, color='gray')
    
    plt.tight_layout()
    return fig

def main():
    """Generate both diagrams"""
    print("="*70)
    print("GENERATING ARCHITECTURE DIAGRAMS")
    print("="*70)
    
    # Generate simple flowchart
    print("\n1. Creating simple flowchart...")
    fig1 = create_simple_flowchart()
    filename1 = 'architecture_simple_flowchart.png'
    fig1.savefig(filename1, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✓ Saved: {filename1}")
    
    # Generate detailed comparison architecture
    print("\n2. Creating detailed architecture diagram...")
    fig2 = create_algorithm_comparison_architecture()
    filename2 = 'architecture_comparison_only.png'
    fig2.savefig(filename2, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✓ Saved: {filename2}")
    
    print("\n" + "="*70)
    print("DIAGRAM GENERATION COMPLETE!")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  1. {filename1} - Simple black & white flowchart")
    print(f"  2. {filename2} - Detailed colorful architecture")
    print("\nBoth diagrams are ready for your paper!")
    print("Use the simple flowchart for cleaner, minimal presentation.")
    print("Use the detailed diagram for comprehensive architecture overview.")
    
    plt.show()

if __name__ == "__main__":
    main()
