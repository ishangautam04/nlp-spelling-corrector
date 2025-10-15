#!/usr/bin/env python3
"""
Generate Proper Flowchart with Standard Shapes
Traditional flowchart with rectangles, diamonds, circles, parallelograms
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, Polygon
import numpy as np

def create_flowchart():
    """Create proper flowchart with standard shapes"""
    
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 15)
    ax.axis('off')
    
    # Title
    ax.text(5, 14.3, 'Spelling Correction System Flowchart', 
            ha='center', fontsize=16, fontweight='bold')
    
    # ============ START (Circle/Oval) ============
    start_circle = mpatches.Ellipse((5, 13.5), 1.5, 0.5, 
                                    facecolor='black', edgecolor='black', linewidth=2)
    ax.add_patch(start_circle)
    ax.text(5, 13.5, 'START', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    # Arrow down
    ax.annotate('', xy=(5, 12.8), xytext=(5, 13.25),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # ============ INPUT (Parallelogram) ============
    input_para = Polygon([(3.2, 12.8), (6.8, 12.8), (6.3, 12.1), (2.7, 12.1)],
                         facecolor='lightblue', edgecolor='black', linewidth=2, alpha=0.3)
    ax.add_patch(input_para)
    ax.text(4.75, 12.45, 'Input Text', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    # Arrow down
    ax.annotate('', xy=(5, 11.6), xytext=(5, 12.1),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # ============ PROCESS BOX 1: Text Preprocessing (Rectangle) ============
    rect1 = Rectangle((3.2, 11.0), 3.6, 0.6, facecolor='lightgreen', 
                     edgecolor='black', linewidth=2, alpha=0.3)
    ax.add_patch(rect1)
    ax.text(5, 11.3, 'Text Preprocessing Pipeline', ha='center', fontsize=10, fontweight='bold')
    
    # Arrow
    ax.annotate('', xy=(5, 10.4), xytext=(5, 11.0),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # ============ PROCESS BOX 2: Tokenization (Rectangle) ============
    rect2 = Rectangle((3.0, 9.8), 4.0, 0.6, facecolor='lightyellow', 
                     edgecolor='black', linewidth=2, alpha=0.4)
    ax.add_patch(rect2)
    ax.text(5, 10.1, 'Tokenization & Word Extraction', ha='center', fontsize=10, fontweight='bold')
    
    # Arrow
    ax.annotate('', xy=(5, 9.2), xytext=(5, 9.8),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # ============ PROCESS BOX 3: Parallel Execution (Rectangle) ============
    rect3 = Rectangle((3.0, 8.6), 4.0, 0.6, facecolor='lightcoral', 
                     edgecolor='black', linewidth=2, alpha=0.4)
    ax.add_patch(rect3)
    ax.text(5, 8.9, 'Parallel Algorithm Execution', ha='center', fontsize=10, fontweight='bold')
    
    # Arrow
    ax.annotate('', xy=(5, 7.6), xytext=(5, 8.6),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # ============ FOUR ALGORITHM BOXES (Rectangles) ============
    algorithms = [
        ('PySpellChecker', 1.5),
        ('Frequency-Based\nCorrection', 3.2),
        ('AutoCorrect ML', 5.0),
        ('Levenshtein\nDistance', 6.7)
    ]
    
    for name, x in algorithms:
        rect = Rectangle((x-0.65, 6.3), 1.3, 1.0, facecolor='wheat', 
                        edgecolor='black', linewidth=2, alpha=0.6)
        ax.add_patch(rect)
        lines = name.split('\n')
        if len(lines) == 1:
            ax.text(x, 6.8, lines[0], ha='center', va='center', fontsize=9, fontweight='bold')
        elif len(lines) == 2:
            ax.text(x, 6.95, lines[0], ha='center', va='center', fontsize=8, fontweight='bold')
            ax.text(x, 6.65, lines[1], ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Arrows from algorithms converging
    for _, x in algorithms:
        ax.annotate('', xy=(5, 5.5), xytext=(x, 6.3),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # ============ PROCESS BOX: Performance Comparison (Rectangle) ============
    rect_compare = Rectangle((2.8, 4.9), 4.4, 0.6, facecolor='lightpink', 
                             edgecolor='black', linewidth=2, alpha=0.4)
    ax.add_patch(rect_compare)
    ax.text(5, 5.2, 'Performance Comparison & Analysis', ha='center', fontsize=10, fontweight='bold')
    
    # Arrow
    ax.annotate('', xy=(5, 4.3), xytext=(5, 4.9),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # ============ PROCESS BOX: Individual Algorithm Outputs (Rectangle) ============
    rect_outputs = Rectangle((3.0, 3.7), 4.0, 0.6, facecolor='lightblue', 
                            edgecolor='black', linewidth=2, alpha=0.4)
    ax.add_patch(rect_outputs)
    ax.text(5, 4.0, 'Individual Algorithm Outputs', ha='center', fontsize=10, fontweight='bold')
    
    # Arrow down
    ax.annotate('', xy=(5, 2.9), xytext=(5, 3.7),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # ============ OUTPUT (Parallelogram) ============
    output_para = Polygon([(3.2, 2.9), (6.8, 2.9), (6.3, 2.1), (2.7, 2.1)],
                         facecolor='lightblue', edgecolor='black', linewidth=2, alpha=0.3)
    ax.add_patch(output_para)
    ax.text(4.75, 2.5, 'Output: Corrected Text', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    # Arrow to END
    ax.annotate('', xy=(5, 1.5), xytext=(5, 2.1),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # ============ END (Circle/Oval) ============
    end_circle = mpatches.Ellipse((5, 1.3), 1.5, 0.5, 
                                  facecolor='black', edgecolor='black', linewidth=2)
    ax.add_patch(end_circle)
    ax.text(5, 1.3, 'END', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    

    
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    print("Generating proper flowchart with standard shapes...")
    
    # Create flowchart
    fig = create_flowchart()
    
    # Save as both PDF (vector - best for LaTeX) and PNG (raster)
    output_pdf = 'spelling_correction_flowchart.pdf'
    output_png = 'spelling_correction_flowchart.png'
    
    fig.savefig(output_pdf, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"✓ Flowchart saved as '{output_pdf}' (vector format - best for LaTeX)")
    
    fig.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Flowchart saved as '{output_png}' (high resolution raster)")
    
    # Also save as SVG (alternative vector format)
    output_svg = 'spelling_correction_flowchart.svg'
    fig.savefig(output_svg, format='svg', bbox_inches='tight', facecolor='white')
    print(f"✓ Flowchart saved as '{output_svg}' (vector format - web friendly)")
    
    plt.close()
    print("\n✅ Flowchart generation complete!")
    print("\n📋 Flowchart structure:")
    print("  • START → Input Text")
    print("  • Text Preprocessing Pipeline")
    print("  • Tokenization & Word Extraction")
    print("  • Parallel Algorithm Execution")
    print("    - PySpellChecker")
    print("    - Frequency-Based Correction")
    print("    - AutoCorrect ML")
    print("    - Levenshtein Distance")
    print("  • Performance Comparison & Analysis")
    print("  • Individual Algorithm Outputs")
    print("  • Output: Corrected Text → END")
    print("\n📁 Saved in 3 formats: PDF (best for LaTeX), PNG, SVG")
