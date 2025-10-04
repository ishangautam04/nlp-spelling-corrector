#!/usr/bin/env python3

from spelling_corrector import SpellingCorrector

def test_context_correction():
    print("Initializing spelling corrector...")
    sc = SpellingCorrector()
    
    test_cases = [
        "The qick brown fox",
        "I recieved the package",
        "Please pick up the phone",
        "She felt relieved after the test"
    ]
    
    print("\n=== Context-Aware Correction ===")
    for text in test_cases:
        try:
            result = sc.correct_with_context(text)
            print(f"'{text}' -> '{result}'")
        except Exception as e:
            print(f"Error with '{text}': {e}")
    
    print("\n=== Enhanced Ensemble ===")
    for text in test_cases:
        try:
            result = sc.correct_text(text, 'ensemble')
            print(f"'{text}' -> '{result}'")
        except Exception as e:
            print(f"Error with '{text}': {e}")
    
    print("\n=== Individual Word Tests ===")
    words = ["qick", "recieved", "pick", "relieved"]
    for word in words:
        try:
            pyspell = sc.correct_with_pyspellchecker(word)
            autocorrect = sc.correct_with_autocorrect(word)
            levenshtein = sc.correct_with_levenshtein(word)
            print(f"'{word}': PySpell={pyspell}, Auto={autocorrect}, Lev={levenshtein}")
        except Exception as e:
            print(f"Error with '{word}': {e}")

if __name__ == "__main__":
    test_context_correction()