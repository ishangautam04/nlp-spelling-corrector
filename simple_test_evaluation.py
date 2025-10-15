#!/usr/bin/env python3
"""
Simple Test Evaluation on Large Dataset
Tests all 4 pre-trained algorithms on the entire dataset (no train-test split)
Shows how well each algorithm performs on your complete dataset
"""

import json
import time
import pandas as pd
from spelling_corrector import SpellingCorrector
from tqdm import tqdm
import numpy as np

class SimpleEvaluator:
    def __init__(self):
        """Initialize with spelling corrector"""
        print("Initializing Spelling Corrector...")
        self.corrector = SpellingCorrector()
        print("✓ Corrector initialized\n")
    
    def load_dataset(self, filepath):
        """Load test dataset"""
        print(f"Loading dataset: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ Loaded {len(data)} samples\n")
        return data
    
    def evaluate_method(self, dataset, method_name):
        """Evaluate a single correction method on entire dataset"""
        print(f"\n{'='*60}")
        print(f"Testing: {method_name.upper()}")
        print(f"{'='*60}")
        
        correct_corrections = 0
        total_samples = 0
        processing_times = []
        corrections_made = 0
        error_type_stats = {
            'keyboard': {'correct': 0, 'total': 0},
            'phonetic': {'correct': 0, 'total': 0},
            'character': {'correct': 0, 'total': 0},
            'simple_typo': {'correct': 0, 'total': 0}
        }
        
        for sample in tqdm(dataset, desc=f"{method_name:15}", ncols=80):
            error_word = sample.get('error', '')
            correct_word = sample.get('correct', '')
            error_type = sample.get('error_type', 'unknown')
            
            if not error_word or not correct_word:
                continue
            
            # Time the correction - correct individual word, not full text
            start_time = time.time()
            
            # Use the appropriate method to correct the single word
            if method_name == 'pyspellchecker':
                corrected_word = self.corrector.correct_with_pyspellchecker(error_word)
            elif method_name == 'autocorrect':
                corrected_word = self.corrector.correct_with_autocorrect(error_word)
            elif method_name == 'levenshtein':
                corrected_word = self.corrector.correct_with_levenshtein(error_word)
            elif method_name == 'frequency':
                corrected_word = self.corrector.correct_with_frequency(error_word)
            else:
                corrected_word = error_word
            
            elapsed = time.time() - start_time
            processing_times.append(elapsed)
            
            total_samples += 1
            
            # Normalize for comparison
            corrected_normalized = corrected_word.lower().strip()
            expected_normalized = correct_word.lower().strip()
            
            # Check if correction is accurate
            if corrected_normalized == expected_normalized:
                correct_corrections += 1
                
                # Track by error type
                if error_type in error_type_stats:
                    error_type_stats[error_type]['correct'] += 1
            
            # Track total by error type
            if error_type in error_type_stats:
                error_type_stats[error_type]['total'] += 1
            
            # Count corrections made (if word was changed)
            if corrected_word != error_word:
                corrections_made += 1
        
        # Calculate metrics
        accuracy = (correct_corrections / total_samples * 100) if total_samples > 0 else 0
        avg_time = np.mean(processing_times) * 1000 if processing_times else 0
        total_time = sum(processing_times)
        
        # Calculate per-error-type accuracy
        error_type_accuracy = {}
        for error_type, stats in error_type_stats.items():
            if stats['total'] > 0:
                error_type_accuracy[error_type] = (stats['correct'] / stats['total'] * 100)
            else:
                error_type_accuracy[error_type] = 0
        
        result = {
            'method': method_name,
            'total_samples': total_samples,
            'correct_corrections': correct_corrections,
            'corrections_made': corrections_made,
            'accuracy_%': round(accuracy, 2),
            'avg_time_ms': round(avg_time, 3),
            'total_time_sec': round(total_time, 2),
            'error_type_accuracy': error_type_accuracy,
            'error_type_stats': error_type_stats
        }
        
        # Print summary
        print(f"\n{'─'*60}")
        print(f"Results for {method_name.upper()}:")
        print(f"{'─'*60}")
        print(f"  Samples processed:     {total_samples}")
        print(f"  Correct corrections:   {correct_corrections}")
        print(f"  Corrections made:      {corrections_made}")
        print(f"  Accuracy:              {accuracy:.2f}%")
        print(f"  Avg time per sample:   {avg_time:.3f} ms")
        print(f"  Total time:            {total_time:.2f} seconds")
        print(f"\n  Per-error-type accuracy:")
        for error_type, acc in error_type_accuracy.items():
            print(f"    {error_type:15}: {acc:5.2f}%")
        
        return result
    
    def run_evaluation(self, dataset_path):
        """Run evaluation on all methods"""
        print("\n" + "="*60)
        print("SPELLING CORRECTION EVALUATION")
        print("No Training - Testing Pre-trained Algorithms")
        print("="*60)
        
        # Load dataset
        dataset = self.load_dataset(dataset_path)
        
        # Methods to test (all pre-trained, no training needed)
        methods = [
            'pyspellchecker',
            'autocorrect', 
            'frequency',
            'levenshtein'
        ]
        
        # Store results
        all_results = []
        detailed_corrections = {}
        
        # Test each method
        for method in methods:
            result = self.evaluate_method(dataset, method)
            all_results.append(result)
        
        return all_results
    
    def print_comparison(self, results):
        """Print comparison table of all methods"""
        print("\n" + "="*80)
        print("FINAL COMPARISON - ALL ALGORITHMS")
        print("="*80)
        print()
        
        # Create comparison table
        print(f"{'Algorithm':<20} {'Accuracy':<12} {'Avg Time (ms)':<15} {'Total Time (s)':<15}")
        print("─" * 80)
        
        for r in results:
            print(f"{r['method'].capitalize():<20} "
                  f"{r['accuracy_%']:>6.2f}%      "
                  f"{r['avg_time_ms']:>10.3f}      "
                  f"{r['total_time_sec']:>12.2f}")
        
        print("\n" + "="*80)
        
        # Find best performer
        best_accuracy = max(results, key=lambda x: x['accuracy_%'])
        fastest = min(results, key=lambda x: x['avg_time_ms'])
        
        print("\n🏆 WINNER - Highest Accuracy:")
        print(f"   {best_accuracy['method'].upper()}: {best_accuracy['accuracy_%']:.2f}%")
        
        print("\n⚡ FASTEST:")
        print(f"   {fastest['method'].upper()}: {fastest['avg_time_ms']:.3f} ms/sample")
        
    def generate_visualizations(self, results):
        """Generate charts showing performance"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        sns.set_style("whitegrid")
        
        # 1. Overall Accuracy Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        methods = [r['method'].title() for r in results]
        accuracies = [r['accuracy_%'] for r in results]
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
        
        bars = ax.bar(methods, accuracies, color=colors, alpha=0.8)
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Algorithm Accuracy Comparison\n(Complete Dataset - No Training)', 
                    fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('simple_evaluation_accuracy.png', dpi=300, bbox_inches='tight')
        print(f"✅ Chart saved: simple_evaluation_accuracy.png")
        plt.close()
        
        # 2. Per-Error-Type Performance
        fig, ax = plt.subplots(figsize=(12, 7))
        error_types = ['keyboard', 'phonetic', 'character', 'simple_typo']
        x = np.arange(len(error_types))
        width = 0.2
        
        for i, r in enumerate(results):
            error_acc = r['error_type_accuracy']
            accuracies = [error_acc.get(et, 0) for et in error_types]
            ax.bar(x + i*width, accuracies, width, label=r['method'].title(), 
                  color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Error Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Performance by Error Type\n(Complete Dataset - No Training)', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([et.replace('_', ' ').title() for et in error_types])
        ax.legend(fontsize=10)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('simple_evaluation_by_error_type.png', dpi=300, bbox_inches='tight')
        print(f"✅ Chart saved: simple_evaluation_by_error_type.png")
        plt.close()
    
    def save_results(self, results, output_file='evaluation_results.json'):
        """Save results to JSON file"""
        # Prepare data for saving (remove non-serializable objects)
        serializable_results = []
        for r in results:
            result_copy = r.copy()
            # Convert error_type_stats to simpler format
            if 'error_type_stats' in result_copy:
                result_copy['error_type_stats'] = {
                    k: {'correct': v['correct'], 'total': v['total']}
                    for k, v in result_copy['error_type_stats'].items()
                }
            serializable_results.append(result_copy)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_file}")
        
        # Also save CSV for easy viewing
        csv_file = output_file.replace('.json', '.csv')
        # Create simplified dataframe without nested objects
        df_data = []
        for r in results:
            df_data.append({
                'Method': r['method'],
                'Accuracy (%)': r['accuracy_%'],
                'Samples': r['total_samples'],
                'Correct': r['correct_corrections'],
                'Corrections Made': r['corrections_made'],
                'Avg Time (ms)': r['avg_time_ms'],
                'Total Time (s)': r['total_time_sec']
            })
        df = pd.DataFrame(df_data)
        df.to_csv(csv_file, index=False)
        print(f"✅ Summary saved to: {csv_file}")
        

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate spelling correction algorithms')
    parser.add_argument('--dataset', 
                       default='spelling_test_dataset_large.json',
                       help='Dataset file to test on')
    parser.add_argument('--output',
                       default='evaluation_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = SimpleEvaluator()
    
    # Run evaluation
    results = evaluator.run_evaluation(args.dataset)
    
    # Print comparison
    evaluator.print_comparison(results)
    
    # Generate visualizations
    print("\n" + "="*80)
    print("Generating visualizations...")
    print("="*80)
    evaluator.generate_visualizations(results)
    
    # Save results
    evaluator.save_results(results, args.output)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nDataset: {args.dataset}")
    print(f"Results: {args.output}")
    print(f"\n💡 All algorithms tested WITHOUT training (using pre-trained models)")
    print(f"📊 This shows performance on your complete dataset")


if __name__ == '__main__':
    main()
