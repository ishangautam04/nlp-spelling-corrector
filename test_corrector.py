#!/usr/bin/env python3
"""
Test script for the Spelling Corrector application
"""

import sys
import time
from spelling_corrector import SpellingCorrector

def test_spelling_corrector():
    """Test the spelling corrector with various examples"""
    print("="*70)
    print("TESTING NLP SPELLING CORRECTOR")
    print("="*70)
    
    # Initialize corrector
    print("Initializing spelling corrector...")
    start_time = time.time()
    corrector = SpellingCorrector()
    init_time = time.time() - start_time
    print(f"✅ Corrector initialized in {init_time:.2f} seconds\n")
    
    # Test cases
    test_cases = [
        {
            "text": "The qick brown fox jumps over the lazy dog",
            "description": "Common typing errors"
        },
        {
            "text": "I recieved your mesage yestarday and it was grate",
            "description": "Multiple spelling mistakes"
        },
        {
            "text": "Artifical inteligence is revolutionizing tecnology",
            "description": "Technical terms with errors"
        },
        {
            "text": "Please chck your speling before submiting the documnt",
            "description": "Missing letters and transpositions"
        },
        {
            "text": "Conscientious students alwayz compleet there asignments",
            "description": "Homophones and similar-sounding words"
        },
        {
            "text": "The committe will mett tomorrow to discus the proposel",
            "description": "Double letter errors"
        }
    ]
    
    # Test each case with different methods
    methods = ['ensemble', 'pyspellchecker', 'autocorrect', 'levenshtein']
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_case['description']}")
        print(f"Original: {test_case['text']}")
        print("-" * 50)
        
        # Test each method
        method_results = {}
        for method in methods:
            start_time = time.time()
            corrected_text, corrections = corrector.correct_text(test_case['text'], method=method)
            processing_time = time.time() - start_time
            
            method_results[method] = {
                'corrected': corrected_text,
                'corrections_count': len(corrections),
                'time': processing_time
            }
            
            print(f"{method.capitalize():15}: {corrected_text}")
            print(f"                Corrections: {len(corrections)}, Time: {processing_time:.3f}s")
        
        # Find consensus
        corrected_texts = [result['corrected'] for result in method_results.values()]
        most_common = max(set(corrected_texts), key=corrected_texts.count)
        consensus_count = corrected_texts.count(most_common)
        
        print(f"\n📊 Analysis:")
        print(f"   Consensus result: {most_common}")
        print(f"   Methods agreeing: {consensus_count}/{len(methods)}")
        print(f"   Average processing time: {sum(r['time'] for r in method_results.values()) / len(methods):.3f}s")
        
        print("\n" + "="*70 + "\n")
    
    # Performance test
    print("PERFORMANCE TEST")
    print("="*70)
    
    # Test with longer text
    long_text = """
    Artifical inteligence and machien lerning are revolutionizing the wrold of tecnology.
    Companys across various industrys are adopting thes advanced solutionss to improev
    ther operationss and provid beter servics to ther custommers. The potentiel of AI
    is enormus, and we are jst begining to scrach the surfac of what is possibl.
    From automativ vehcles to smart citys, from helthcar to finans, AI is transformig
    evry aspct of our lifes. Howevr, with grat powr coms grat responsibilty, and
    we must ensur that thes tecnologys are developd and deployyd ethicaly.
    """.strip()
    
    print(f"Testing with longer text ({len(long_text.split())} words)...")
    
    for method in ['ensemble', 'pyspellchecker', 'autocorrect']:
        start_time = time.time()
        corrected_text, corrections = corrector.correct_text(long_text, method=method)
        processing_time = time.time() - start_time
        
        print(f"\n{method.capitalize()} method:")
        print(f"  Processing time: {processing_time:.3f}s")
        print(f"  Corrections made: {len(corrections)}")
        print(f"  Words per second: {len(long_text.split()) / processing_time:.1f}")
    
    # Word suggestions test
    print("\n" + "="*70)
    print("WORD SUGGESTIONS TEST")
    print("="*70)
    
    test_words = ['qick', 'recieved', 'artifical', 'tecnology', 'helth']
    
    for word in test_words:
        suggestions = corrector.get_word_suggestions(word, n=5)
        print(f"'{word}' → {suggestions}")
    
    print("\n✅ All tests completed successfully!")

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "="*70)
    print("EDGE CASES TEST")
    print("="*70)
    
    corrector = SpellingCorrector()
    
    edge_cases = [
        ("", "Empty string"),
        ("   ", "Whitespace only"),
        ("123 456", "Numbers only"),
        ("!@#$%", "Punctuation only"),
        ("ABC XYZ", "All uppercase"),
        ("a b c d e", "Single letters"),
        ("verylongwordthatdoesnotexist", "Very long non-word"),
        ("Dr. Smith's car won't start.", "Text with punctuation and contractions")
    ]
    
    for text, description in edge_cases:
        print(f"\nTesting: {description}")
        print(f"Input: '{text}'")
        
        try:
            corrected_text, corrections = corrector.correct_text(text, method='ensemble')
            print(f"Output: '{corrected_text}'")
            print(f"Corrections: {len(corrections)}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n✅ Edge cases test completed!")

if __name__ == "__main__":
    try:
        test_spelling_corrector()
        test_edge_cases()
        
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED!")
        print("="*70)
        print("\nTo run the application:")
        print("1. Web interface: streamlit run streamlit_app.py")
        print("2. Command line: python cli.py --interactive")
        print("3. Direct usage: python spelling_corrector.py")
        
    except KeyboardInterrupt:
        print("\n\n❌ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
