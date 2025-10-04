#!/usr/bin/env python3
"""
CLI Test Runner for ML Spelling Corrector
Allows interactive testing of different methods
"""

import argparse
import sys
from ml_spelling_corrector import MLSpellingCorrector

def test_single_word(corrector, word, method='all'):
    """Test correction of a single word"""
    print(f"\nTesting word: '{word}'")
    print("-" * 30)
    
    if method in ['all', 'pyspell']:
        result = corrector.correct_with_pyspellchecker(word)
        print(f"PySpellChecker: {result}")
    
    if method in ['all', 'autocorrect']:
        result = corrector.correct_with_autocorrect(word)
        print(f"AutoCorrect:    {result}")
    
    if method in ['all', 'ml']:
        result = corrector.correct_with_ml_model(word)
        print(f"ML Model:       {result}")
    
    if method in ['all', 'ensemble']:
        result = corrector.ensemble_correction(word)
        if isinstance(result, tuple):
            result = result[0]
        print(f"Ensemble:       {result}")
    
    if method in ['all', 'ml_ensemble']:
        result = corrector.ensemble_correction_with_ml(word)
        if isinstance(result, tuple):
            result = result[0]
        print(f"ML Ensemble:    {result}")

def test_sentence(corrector, sentence, method='all'):
    """Test correction of a sentence"""
    print(f"\nTesting sentence: '{sentence}'")
    print("-" * 50)
    
    if method in ['all', 'ensemble']:
        result = corrector.correct_text(sentence, method='ensemble')
        if isinstance(result, tuple):
            result = result[0]
        print(f"Standard Ensemble: {result}")
    
    if method in ['all', 'ml_only']:
        result = corrector.correct_text_with_ml(sentence, method='ml_only')
        print(f"ML Only:           {result}")
    
    if method in ['all', 'ml_ensemble']:
        result = corrector.correct_text_with_ml(sentence, method='ensemble_ml')
        if isinstance(result, tuple):
            corrected, log = result
            print(f"ML Enhanced:       {corrected}")
            if log:
                print("  Corrections made:")
                for correction in log:
                    print(f"    '{correction['original']}' → '{correction['corrected']}'")
        else:
            print(f"ML Enhanced:       {result}")

def interactive_mode(corrector):
    """Interactive testing mode"""
    print("\n🔤 INTERACTIVE ML SPELLING CORRECTOR TEST 🔤")
    print("="*50)
    print("Commands:")
    print("  Type a word or sentence to test")
    print("  'quit' or 'exit' to stop")
    print("  'help' for more options")
    print("="*50)
    
    while True:
        try:
            user_input = input("\nEnter text to correct: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if user_input.lower() == 'help':
                print("\nAvailable methods:")
                print("  - Standard ensemble (traditional algorithms)")
                print("  - ML-only (machine learning model only)")
                print("  - ML-enhanced (ML + traditional ensemble)")
                print("  - Individual algorithms (PySpellChecker, AutoCorrect)")
                continue
            
            if not user_input:
                continue
            
            # Determine if single word or sentence
            if len(user_input.split()) == 1:
                test_single_word(corrector, user_input)
            else:
                test_sentence(corrector, user_input)
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

def benchmark_test(corrector):
    """Run benchmark tests"""
    print("\n📊 BENCHMARK TESTS 📊")
    print("="*30)
    
    # Common misspellings test
    common_errors = [
        "teh", "thier", "recieve", "seperate", "definately",
        "beleive", "wierd", "freind", "occured", "begining"
    ]
    
    print("\nCommon Spelling Errors Test:")
    print("-" * 30)
    
    for word in common_errors:
        ml_result = corrector.correct_with_ml_model(word)
        ensemble_result = corrector.ensemble_correction(word)
        if isinstance(ensemble_result, tuple):
            ensemble_result = ensemble_result[0]
        
        print(f"{word:12} → ML: {ml_result:12} | Ensemble: {ensemble_result}")
    
    # Sentence tests
    print("\nSentence Correction Test:")
    print("-" * 25)
    
    sentences = [
        "Teh qick brown fox",
        "I recieve your mesage",
        "Please seperate thier items",
        "This is definately wierd"
    ]
    
    for sentence in sentences:
        standard = corrector.correct_text(sentence, method='ensemble')
        if isinstance(standard, tuple):
            standard = standard[0]
        
        ml_enhanced = corrector.correct_text_with_ml(sentence, method='ensemble_ml')
        if isinstance(ml_enhanced, tuple):
            ml_enhanced = ml_enhanced[0]
        
        print(f"\nOriginal:    {sentence}")
        print(f"Standard:    {standard}")
        print(f"ML Enhanced: {ml_enhanced}")

def main():
    parser = argparse.ArgumentParser(description='Test ML Spelling Corrector')
    parser.add_argument('-w', '--word', help='Test a single word')
    parser.add_argument('-s', '--sentence', help='Test a sentence')
    parser.add_argument('-m', '--method', 
                       choices=['all', 'pyspell', 'autocorrect', 'ml', 'ensemble', 'ml_ensemble'],
                       default='all', help='Correction method to test')
    parser.add_argument('-i', '--interactive', action='store_true', 
                       help='Start interactive mode')
    parser.add_argument('-b', '--benchmark', action='store_true',
                       help='Run benchmark tests')
    parser.add_argument('--model', default='spelling_correction_model.pkl',
                       help='ML model file to use')
    
    args = parser.parse_args()
    
    # Check if model exists
    import os
    if not os.path.exists(args.model):
        print(f"❌ Model file '{args.model}' not found!")
        print("Please run 'python train_spelling_model.py' first to create the model")
        sys.exit(1)
    
    # Load corrector
    try:
        corrector = MLSpellingCorrector(args.model)
        print(f"✅ Loaded ML model: {args.model}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # Execute based on arguments
    if args.interactive:
        interactive_mode(corrector)
    elif args.benchmark:
        benchmark_test(corrector)
    elif args.word:
        test_single_word(corrector, args.word, args.method)
    elif args.sentence:
        test_sentence(corrector, args.sentence, args.method)
    else:
        # Default: show usage examples
        print("\n🔤 ML SPELLING CORRECTOR TEST CLI 🔤")
        print("="*40)
        print("\nUsage examples:")
        print("  python test_cli.py -w 'teh'")
        print("  python test_cli.py -s 'Teh qick brown fox'")
        print("  python test_cli.py -i                    # Interactive mode")
        print("  python test_cli.py -b                    # Benchmark tests")
        print("  python test_cli.py -w 'recieve' -m ml    # Test ML only")
        print("\nFor more options, use --help")
        
        # Quick demo
        print("\n📋 Quick Demo:")
        test_single_word(corrector, "teh", "all")
        test_sentence(corrector, "I recieve your mesage", "all")

if __name__ == "__main__":
    main()