#!/usr/bin/env python3
"""
Quick ML Testing - Simple tests to verify ML model performance
"""

from ml_spelling_corrector import MLSpellingCorrector

def quick_ml_test():
    """Quick test of ML model corrections"""
    print("🔤 QUICK ML SPELLING CORRECTOR TEST 🔤")
    print("="*50)
    
    # Load the ML corrector
    corrector = MLSpellingCorrector("spelling_correction_model.pkl")
    
    # Quick test cases
    test_cases = [
        # Single words that should be learned by ML
        ("teh", "Should correct to 'the'"),
        ("recieve", "Should correct to 'receive'"),
        ("seperate", "Should correct to 'separate'"),
        ("definately", "Should correct to 'definitely'"),
        ("thier", "Should correct to 'their'"),
        
        # Sentences
        ("Teh qick brown fox", "Multiple corrections needed"),
        ("I recieve your mesage", "Common typos"),
        ("Please seperate thier items", "Multiple word errors"),
    ]
    
    print("\n1. Testing ML-only corrections:")
    print("-" * 30)
    for text, description in test_cases:
        ml_result = corrector.correct_with_ml_model(text) if len(text.split()) == 1 else corrector.correct_text_with_ml(text, method='ml_only')
        print(f"'{text}' → '{ml_result}'")
    
    print("\n2. Testing ML-Enhanced Ensemble vs Standard:")
    print("-" * 45)
    
    sentence_tests = [
        "Teh qick brown fox",
        "I recieve your mesage", 
        "Please seperate thier items"
    ]
    
    for text in sentence_tests:
        # Standard ensemble
        standard = corrector.correct_text(text, method='ensemble')
        if isinstance(standard, tuple):
            standard = standard[0]
        
        # ML-enhanced ensemble
        ml_enhanced = corrector.correct_text_with_ml(text, method='ensemble_ml')
        if isinstance(ml_enhanced, tuple):
            ml_enhanced = ml_enhanced[0]
        
        print(f"Original:     {text}")
        print(f"Standard:     {standard}")
        print(f"ML-Enhanced:  {ml_enhanced}")
        print()
    
    print("3. Algorithm Performance Comparison:")
    print("-" * 35)
    
    test_word = "recieve"
    methods = [
        ('PySpellChecker', lambda: corrector.correct_with_pyspellchecker(test_word)),
        ('AutoCorrect', lambda: corrector.correct_with_autocorrect(test_word)),
        ('ML Model', lambda: corrector.correct_with_ml_model(test_word)),
        ('Standard Ensemble', lambda: corrector.ensemble_correction(test_word)),
        ('ML Ensemble', lambda: corrector.ensemble_correction_with_ml(test_word))
    ]
    
    print(f"Testing word: '{test_word}'")
    for name, method in methods:
        try:
            result = method()
            if isinstance(result, tuple):
                result = result[0]
            print(f"{name:18}: {result}")
        except Exception as e:
            print(f"{name:18}: Error - {e}")
    
    print("\n✅ Quick test completed!")

if __name__ == "__main__":
    quick_ml_test()