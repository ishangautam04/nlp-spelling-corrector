#!/usr/bin/env python3
"""
Comprehensive Spelling Correction Evaluation Framework
Tests individual algorithms vs ensemble methods with detailed metrics
"""

import json
import time
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from spelling_corrector import SpellingCorrector
from ml_spelling_corrector import MLSpellingCorrector
import warnings
warnings.filterwarnings('ignore')

class SpellingCorrectionEvaluator:
    def __init__(self, model_path=None):
        """Initialize evaluator with correctors"""
        print("Initializing spelling correctors...")
        
        # Initialize basic corrector
        self.base_corrector = SpellingCorrector()
        
        # Initialize ML corrector if model exists
        self.ml_corrector = None
        if model_path:
            try:
                self.ml_corrector = MLSpellingCorrector(model_path)
                print(f"ML corrector loaded from {model_path}")
            except Exception as e:
                print(f"Could not load ML corrector: {e}")
        
        # Define individual correction methods
        self.individual_methods = {
            'pyspellchecker': self.base_corrector.correct_with_pyspellchecker,
            'autocorrect': self.base_corrector.correct_with_autocorrect,
            'levenshtein': self.base_corrector.correct_with_levenshtein,
            'frequency': self.base_corrector.correct_with_frequency
        }
        
        # No ensemble methods - removed as they don't improve performance
    
    def _standard_ensemble_wrapper(self, word, context=None):
        """Wrapper for standard ensemble correction"""
        result, _ = self.base_corrector.ensemble_correction(word)
        return result
    
    def _weighted_ensemble_wrapper(self, word, context=None):
        """Wrapper for weighted ensemble correction"""
        result, _ = self.base_corrector.ensemble_correction(word)
        return result
    
    def _ml_ensemble_wrapper(self, word, context=None):
        """Wrapper for ML-enhanced ensemble correction"""
        if self.ml_corrector:
            result, _ = self.ml_corrector.ensemble_correction_with_ml(word)
            return result
        return word
    
    def load_dataset(self, filename):
        """Load test dataset from JSON file"""
        print(f"Loading dataset from {filename}...")
        with open(filename, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        print(f"Loaded {len(dataset)} test samples")
        return dataset
    
    def evaluate_method(self, method_func, dataset, method_name="method"):
        """Evaluate a single correction method on dataset"""
        print(f"Evaluating {method_name}...")
        
        results = []
        correct_predictions = 0
        total_predictions = len(dataset)
        
        start_time = time.time()
        
        for i, sample in enumerate(dataset):
            if i % 100 == 0 and i > 0:
                print(f"  Processed {i}/{total_predictions} samples...")
            
            error_word = sample['error']
            correct_word = sample['correct']
            context = sample.get('context', '')
            error_type = sample['error_type']
            
            # Get prediction
            try:
                if 'context' in str(method_func):
                    predicted = method_func(error_word, context)
                else:
                    predicted = method_func(error_word)
            except Exception as e:
                predicted = error_word  # Fallback to original if error
            
            # Check if prediction is correct
            is_correct = predicted.lower() == correct_word.lower()
            if is_correct:
                correct_predictions += 1
            
            results.append({
                'error_word': error_word,
                'correct_word': correct_word,
                'predicted_word': predicted,
                'is_correct': is_correct,
                'error_type': error_type,
                'method': method_name
            })
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        accuracy = correct_predictions / total_predictions
        
        print(f"  {method_name} Accuracy: {accuracy:.3f} ({correct_predictions}/{total_predictions})")
        print(f"  Processing time: {processing_time:.2f} seconds")
        
        return results, accuracy, processing_time
    
    def evaluate_all_methods(self, dataset):
        """Evaluate all individual and ensemble methods"""
        all_results = []
        performance_summary = {}
        
        print("="*60)
        print("EVALUATING INDIVIDUAL METHODS")
        print("="*60)
        
        # Evaluate individual methods
        for method_name, method_func in self.individual_methods.items():
            results, accuracy, time_taken = self.evaluate_method(
                method_func, dataset, method_name
            )
            all_results.extend(results)
            performance_summary[method_name] = {
                'accuracy': accuracy,
                'time': time_taken,
                'type': 'individual'
            }
        
        return all_results, performance_summary
    
    def analyze_error_types(self, results_df):
        """Analyze performance by error type"""
        print("\n" + "="*60)
        print("ERROR TYPE ANALYSIS")
        print("="*60)
        
        error_type_analysis = {}
        
        for error_type in results_df['error_type'].unique():
            type_data = results_df[results_df['error_type'] == error_type]
            
            # Calculate accuracy by method for this error type
            method_accuracies = type_data.groupby('method')['is_correct'].mean()
            
            error_type_analysis[error_type] = method_accuracies.to_dict()
            
            print(f"\n{error_type.upper()} errors:")
            for method, accuracy in method_accuracies.sort_values(ascending=False).items():
                print(f"  {method}: {accuracy:.3f}")
        
        return error_type_analysis
    
    def analyze_algorithm_strengths(self, results_df):
        """Analyze where each algorithm performs best"""
        print("\n" + "="*60)
        print("ALGORITHM STRENGTHS ANALYSIS")
        print("="*60)
        
        individual_methods = ['pyspellchecker', 'autocorrect', 'levenshtein', 'frequency']
        
        strengths = {}
        
        for method in individual_methods:
            strengths[method] = {
                'best_error_types': [],
                'accuracy_by_type': {}
            }
        
        # Analyze each error type
        for error_type in results_df['error_type'].unique():
            type_data = results_df[results_df['error_type'] == error_type]
            method_accuracies = type_data.groupby('method')['is_correct'].mean()
            
            # Find best method for this error type
            best_method = method_accuracies.idxmax()
            best_accuracy = method_accuracies.max()
            
            if best_method in individual_methods:
                strengths[best_method]['best_error_types'].append((error_type, best_accuracy))
                strengths[best_method]['accuracy_by_type'][error_type] = best_accuracy
        
        # Print findings
        for method, data in strengths.items():
            if data['best_error_types']:
                print(f"\n{method.upper()} excels at:")
                for error_type, accuracy in sorted(data['best_error_types'], key=lambda x: x[1], reverse=True)[:5]:
                    print(f"  • {error_type}: {accuracy:.3f}")
        
        return strengths
    
    def create_performance_plots(self, performance_summary, error_type_analysis, save_plots=True):
        """Create visualization plots for performance analysis"""
        print("\nCreating performance visualizations...")
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Spelling Correction Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Overall accuracy comparison
        methods = list(performance_summary.keys())
        accuracies = [performance_summary[method]['accuracy'] for method in methods]
        colors = ['lightcoral' if performance_summary[method]['type'] == 'individual' else 'lightblue' for method in methods]
        
        bars = axes[0, 0].bar(methods, accuracies, color=colors)
        axes[0, 0].set_title('Overall Accuracy by Method')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{acc:.3f}', ha='center', va='bottom')
        
        # 2. Processing time comparison
        times = [performance_summary[method]['time'] for method in methods]
        
        bars = axes[0, 1].bar(methods, times, color=colors)
        axes[0, 1].set_title('Processing Time by Method')
        axes[0, 1].set_ylabel('Time (seconds)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        for bar, time_val in zip(bars, times):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{time_val:.1f}s', ha='center', va='bottom')
        
        # 3. Error type heatmap
        error_types = list(error_type_analysis.keys())
        methods_subset = [m for m in methods if m in error_type_analysis[error_types[0]]]
        
        heatmap_data = []
        for error_type in error_types:
            row = [error_type_analysis[error_type].get(method, 0) for method in methods_subset]
            heatmap_data.append(row)
        
        im = axes[1, 0].imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        axes[1, 0].set_title('Accuracy by Error Type and Method')
        axes[1, 0].set_xticks(range(len(methods_subset)))
        axes[1, 0].set_xticklabels(methods_subset, rotation=45)
        axes[1, 0].set_yticks(range(len(error_types)))
        axes[1, 0].set_yticklabels(error_types)
        
        # Add text annotations
        for i in range(len(error_types)):
            for j in range(len(methods_subset)):
                text = axes[1, 0].text(j, i, f'{heatmap_data[i][j]:.2f}',
                                     ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=axes[1, 0])
        
        # 4. Individual vs Ensemble comparison
        individual_acc = [acc for method, acc in zip(methods, accuracies) 
                         if performance_summary[method]['type'] == 'individual']
        ensemble_acc = [acc for method, acc in zip(methods, accuracies) 
                       if performance_summary[method]['type'] == 'ensemble']
        
        if individual_acc and ensemble_acc:
            axes[1, 1].boxplot([individual_acc, ensemble_acc], labels=['Individual', 'Ensemble'])
            axes[1, 1].set_title('Individual vs Ensemble Methods')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('spelling_correction_performance.png', dpi=300, bbox_inches='tight')
            print("Performance plots saved to 'spelling_correction_performance.png'")
        
        plt.show()
    
    def generate_detailed_report(self, performance_summary, error_type_analysis, strengths, dataset_info):
        """Generate comprehensive text report"""
        report = []
        report.append("="*80)
        report.append("COMPREHENSIVE SPELLING CORRECTION EVALUATION REPORT")
        report.append("="*80)
        
        report.append(f"\nDataset Information:")
        report.append(f"Total samples: {dataset_info['total_samples']}")
        report.append(f"Error types: {', '.join(dataset_info['error_types'])}")
        
        report.append(f"\n{'METHOD':<20} {'ACCURACY':<10} {'TIME (s)':<10} {'TYPE':<12}")
        report.append("-" * 60)
        
        # Sort by accuracy
        sorted_methods = sorted(performance_summary.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        for method, stats in sorted_methods:
            report.append(f"{method:<20} {stats['accuracy']:<10.3f} {stats['time']:<10.1f} {stats['type']:<12}")
        
        best_individual = sorted([m for m, s in sorted_methods if s['type'] == 'individual'], key=lambda x: performance_summary[x]['accuracy'], reverse=True)[0]
        report.append(f"\nBest Overall Method: {best_individual} ({performance_summary[best_individual]['accuracy']:.3f})")
        
        # Error type analysis
        report.append(f"\nERROR TYPE PERFORMANCE:")
        report.append("-" * 40)
        
        for error_type, method_accs in error_type_analysis.items():
            best_method = max(method_accs.items(), key=lambda x: x[1])
            report.append(f"{error_type}: Best = {best_method[0]} ({best_method[1]:.3f})")
        
        # Algorithm strengths
        report.append(f"\nALGORITHM STRENGTHS:")
        report.append("-" * 40)
        
        for method, data in strengths.items():
            if data['best_error_types']:
                report.append(f"\n{method.upper()} excels at:")
                for error_type, accuracy in sorted(data['best_error_types'], key=lambda x: x[1], reverse=True)[:3]:
                    report.append(f"  • {error_type}: {accuracy:.3f}")
        
        report_text = "\n".join(report)
        
        # Save to file
        with open('spelling_correction_evaluation_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\nDetailed report saved to 'spelling_correction_evaluation_report.txt'")
        
        return report_text

def main():
    """Run comprehensive evaluation"""
    print("Starting Comprehensive Spelling Correction Evaluation")
    print("="*60)
    
    # Initialize evaluator
    model_path = "spelling_correction_model.pkl"
    evaluator = SpellingCorrectionEvaluator(model_path)
    
    # Test with different dataset sizes
    dataset_files = [
        'spelling_test_dataset_small.json',
        'spelling_test_dataset_medium.json',
        'spelling_test_dataset_large.json'
    ]
    
    for dataset_file in dataset_files:
        try:
            print(f"\n{'='*60}")
            print(f"EVALUATING ON {dataset_file.upper()}")
            print(f"{'='*60}")
            
            # Load dataset
            dataset = evaluator.load_dataset(dataset_file)
            
            # Run evaluation
            all_results, performance_summary = evaluator.evaluate_all_methods(dataset)
            
            # Convert to DataFrame for analysis
            results_df = pd.DataFrame(all_results)
            
            # Analyze performance
            error_type_analysis = evaluator.analyze_error_types(results_df)
            strengths = evaluator.analyze_algorithm_strengths(results_df)
            
            # Create visualizations
            evaluator.create_performance_plots(performance_summary, error_type_analysis)
            
            # Generate report
            dataset_info = {
                'total_samples': len(dataset),
                'error_types': list(set(sample['error_type'] for sample in dataset))
            }
            
            evaluator.generate_detailed_report(
                performance_summary, error_type_analysis, strengths, dataset_info
            )
            
            # Save results
            results_df.to_csv(f'results_{dataset_file.replace(".json", ".csv")}', index=False)
            print(f"Detailed results saved to 'results_{dataset_file.replace('.json', '.csv')}'")
            
            break  # Use first available dataset
            
        except FileNotFoundError:
            print(f"Dataset {dataset_file} not found, skipping...")
            continue
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()