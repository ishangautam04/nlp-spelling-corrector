#!/usr/bin/env python3
"""
Comprehensive Testing Suite for ML-Enhanced Spelling Corrector
Tests various aspects of the ML training and correction system
"""

import time
from ml_spelling_corrector import MLSpellingCorrector
from train_spelling_model import SpellingCorrectionTrainer
import pandas as pd

def test_basic_ml_corrections():
    """Test basic ML model corrections"""
    print("=== Test 1: Basic ML Model Corrections ===")
    
    # Load the ML-enhanced corrector
    corrector = MLSpellingCorrector("spelling_correction_model.pkl")
    
    test_words = [
        "teh",           # Should be "the"
        "thier",         # Should be "their"
        "recieve",       # Should be "receive"
        "seperate",      # Should be "separate"
        "definately",    # Should be "definitely"
        "beleive",       # Should be "believe"
        "wierd",         # Should be "weird"
        "freind",        # Should be "friend"
        "occured",       # Should be "occurred"
        "begining"       # Should be "beginning"
    ]
    
    print("Testing individual word corrections:")
    for word in test_words:
        # Test ML-only correction
        ml_correction = corrector.correct_with_ml_model(word)
        
        # Test standard ensemble
        standard_correction = corrector.correct_with_pyspellchecker(word)
        
        print(f"'{word}' -> ML: '{ml_correction}' | PySpell: '{standard_correction}'")
    
    print("\n" + "="*60 + "\n")

def test_sentence_corrections():
    """Test full sentence corrections"""
    print("=== Test 2: Full Sentence Corrections ===")
    
    corrector = MLSpellingCorrector("spelling_correction_model.pkl")
    
    test_sentences = [
        "Teh qick brown fox jumps ovr teh lazy dog",
        "I recieve your mesage and it was grate",
        "Please seperate thier beleifs from the facts",
        "We definately need to begining this project",
        "Their freind is wierd but very intresting",
        "The occured error was begining to worry us",
        "I beleive we should recieve the documnt soon"
    ]
    
    print("Testing sentence corrections (Standard vs ML-Enhanced):\n")
    
    for sentence in test_sentences:
        print(f"Original: {sentence}")
        
        # Standard ensemble correction
        standard_result = corrector.correct_text(sentence, method='ensemble')
        if isinstance(standard_result, tuple):
            standard_corrected = standard_result[0]
        else:
            standard_corrected = standard_result
        
        # ML-enhanced correction
        ml_result = corrector.correct_text_with_ml(sentence, method='ensemble_ml')
        if isinstance(ml_result, tuple):
            ml_corrected, corrections_log = ml_result
            print(f"Standard: {standard_corrected}")
            print(f"ML-Enhanced: {ml_corrected}")
            
            if corrections_log:
                print("  Corrections made:")
                for correction in corrections_log:
                    print(f"    '{correction['original']}' → '{correction['corrected']}'")
        else:
            print(f"ML-Enhanced: {ml_result}")
        
        print("-" * 50)
    
    print("\n" + "="*60 + "\n")

def test_ml_only_vs_ensemble():
    """Compare ML-only vs ensemble approaches"""
    print("=== Test 3: ML-Only vs Ensemble Comparison ===")
    
    corrector = MLSpellingCorrector("spelling_correction_model.pkl")
    
    test_cases = [
        "teh studnet was vrey smart",
        "seperate definately recieve",
        "thier beleif is wierd",
        "begining occured freind"
    ]
    
    for text in test_cases:
        print(f"Original: {text}")
        
        # ML-only correction
        ml_only = corrector.correct_text_with_ml(text, method='ml_only')
        
        # ML-enhanced ensemble
        ml_ensemble = corrector.correct_text_with_ml(text, method='ensemble_ml')
        if isinstance(ml_ensemble, tuple):
            ml_ensemble = ml_ensemble[0]
        
        # Standard ensemble
        standard = corrector.correct_text(text, method='ensemble')
        if isinstance(standard, tuple):
            standard = standard[0]
        
        print(f"ML-Only:     {ml_only}")
        print(f"ML-Ensemble: {ml_ensemble}")
        print(f"Standard:    {standard}")
        print("-" * 50)
    
    print("\n" + "="*60 + "\n")

def test_performance_timing():
    """Test performance of different correction methods"""
    print("=== Test 4: Performance Timing ===")
    
    corrector = MLSpellingCorrector("spelling_correction_model.pkl")
    
    # Test text with multiple errors
    test_text = "Teh qick brown fox jumps ovr teh lazy dog and recieve many seperate gifts from thier freinds"
    
    methods = [
        ('Standard Ensemble', lambda: corrector.correct_text(test_text, method='ensemble')),
        ('ML-Enhanced Ensemble', lambda: corrector.correct_text_with_ml(test_text, method='ensemble_ml')),
        ('ML-Only', lambda: corrector.correct_text_with_ml(test_text, method='ml_only')),
        ('PySpellChecker Only', lambda: corrector.correct_text(test_text, method='pyspellchecker')),
        ('AutoCorrect Only', lambda: corrector.correct_text(test_text, method='autocorrect'))
    ]
    
    print(f"Test text: {test_text}\n")
    
    for method_name, method_func in methods:
        # Warm up
        method_func()
        
        # Time the method
        start_time = time.time()
        result = method_func()
        end_time = time.time()
        
        if isinstance(result, tuple):
            result = result[0]
        
        print(f"{method_name}:")
        print(f"  Result: {result}")
        print(f"  Time: {(end_time - start_time)*1000:.2f} ms")
        print()
    
    print("="*60 + "\n")

def test_custom_training():
    """Test training with custom dataset"""
    print("=== Test 5: Custom Training Test ===")
    
    # Create a custom dataset with domain-specific errors
    custom_data = [
        # Technical terms
        {'incorrect': 'algoritm', 'correct': 'algorithm'},
        {'incorrect': 'databse', 'correct': 'database'},
        {'incorrect': 'programing', 'correct': 'programming'},
        {'incorrect': 'variabl', 'correct': 'variable'},
        {'incorrect': 'functon', 'correct': 'function'},
        
        # Medical terms
        {'incorrect': 'medecine', 'correct': 'medicine'},
        {'incorrect': 'hosptial', 'correct': 'hospital'},
        {'incorrect': 'patinet', 'correct': 'patient'},
        {'incorrect': 'diagnois', 'correct': 'diagnosis'},
        {'incorrect': 'treatmnt', 'correct': 'treatment'},
        
        # Business terms
        {'incorrect': 'managment', 'correct': 'management'},
        {'incorrect': 'businss', 'correct': 'business'},
        {'incorrect': 'custmer', 'correct': 'customer'},
        {'incorrect': 'reveue', 'correct': 'revenue'},
        {'incorrect': 'stratgy', 'correct': 'strategy'}
    ]
    
    # Train new model
    trainer = SpellingCorrectionTrainer()
    df = pd.DataFrame(custom_data)
    
    print("Training custom model with domain-specific terms...")
    trainer.train_model(df)
    
    # Test the custom trained model
    print("\nTesting custom trained model:")
    test_words = ['algoritm', 'databse', 'medecine', 'managment', 'custmer']
    
    for word in test_words:
        correction = trainer.predict_correction(word)
        print(f"'{word}' -> '{correction}'")
    
    # Save custom model
    trainer.save_model("custom_spelling_model.pkl")
    print("\nCustom model saved as 'custom_spelling_model.pkl'")
    
    # Test loading custom model
    print("\nTesting custom model integration:")
    custom_corrector = MLSpellingCorrector("custom_spelling_model.pkl")
    
    test_sentence = "The algoritm uses a databse to help with programing tasks"
    custom_result = custom_corrector.correct_text_with_ml(test_sentence, method='ensemble_ml')
    if isinstance(custom_result, tuple):
        custom_result = custom_result[0]
    
    print(f"Original: {test_sentence}")
    print(f"Custom ML: {custom_result}")
    
    print("\n" + "="*60 + "\n")

def test_error_patterns():
    """Test different types of spelling errors"""
    print("=== Test 6: Different Error Pattern Types ===")
    
    corrector = MLSpellingCorrector("spelling_correction_model.pkl")
    
    error_types = {
        "Transposition": ["form", "jsut", "waht", "whcih"],  # Should be: from, just, what, which
        "Missing Letters": ["wich", "becaus", "befor", "wth"],  # Should be: which, because, before, with
        "Extra Letters": ["whith", "whenn", "thenn", "goood"],  # Should be: with, when, then, good
        "Wrong Letters": ["teh", "thier", "recieve", "seperate"],  # Should be: the, their, receive, separate
        "Phonetic": ["nite", "lite", "rite", "thru"]  # Should be: night, light, right, through
    }
    
    for error_type, words in error_types.items():
        print(f"{error_type} Errors:")
        for word in words:
            # Test with ML model
            ml_correction = corrector.correct_with_ml_model(word)
            
            # Test with standard ensemble
            standard_result = corrector.ensemble_correction(word)
            if isinstance(standard_result, tuple):
                standard_correction = standard_result[0]
            else:
                standard_correction = standard_result
            
            print(f"  '{word}' -> ML: '{ml_correction}' | Standard: '{standard_correction}'")
        print()
    
    print("="*60 + "\n")

def test_context_sensitivity():
    """Test context-sensitive corrections"""
    print("=== Test 7: Context Sensitivity Test ===")
    
    corrector = MLSpellingCorrector("spelling_correction_model.pkl")
    
    # Words that could have multiple corrections depending on context
    context_tests = [
        ("The qick brown fox", "qick should be 'quick' in this context"),
        ("Please pic up the phone", "pic should be 'pick' in this context"),
        ("I will pic a card", "pic should be 'pick' in this context"),
        ("Send me a pic", "pic might stay as 'pic' (informal for picture)"),
        ("The studnet was smart", "studnet should be 'student'"),
        ("I recieve your message", "recieve should be 'receive'")
    ]
    
    for text, explanation in context_tests:
        print(f"Text: {text}")
        print(f"Expected: {explanation}")
        
        # Test context-aware correction
        context_result = corrector.correct_text(text, method='context')
        
        # Test ML-enhanced ensemble
        ml_result = corrector.correct_text_with_ml(text, method='ensemble_ml')
        if isinstance(ml_result, tuple):
            ml_result = ml_result[0]
        
        print(f"Context Method: {context_result}")
        print(f"ML-Enhanced: {ml_result}")
        print("-" * 40)
    
    print("\n" + "="*60 + "\n")

def main():
    """Run all tests"""
    print("🔤 COMPREHENSIVE ML SPELLING CORRECTOR TESTING SUITE 🔤")
    print("="*70)
    print()
    
    # Check if ML model exists
    import os
    if not os.path.exists("spelling_correction_model.pkl"):
        print("❌ ML model not found!")
        print("Please run 'python train_spelling_model.py' first to create the model")
        return
    
    print("✅ ML model found. Starting comprehensive tests...\n")
    
    # Run all tests
    try:
        test_basic_ml_corrections()
        test_sentence_corrections()
        test_ml_only_vs_ensemble()
        test_performance_timing()
        test_custom_training()
        test_error_patterns()
        test_context_sensitivity()
        
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY! 🎉")
        print("\nSummary:")
        print("✅ Basic ML corrections tested")
        print("✅ Full sentence corrections tested")
        print("✅ Performance comparison completed")
        print("✅ Custom training demonstrated")
        print("✅ Error pattern analysis completed")
        print("✅ Context sensitivity evaluated")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()