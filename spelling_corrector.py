import nltk
import re
import string
from collections import Counter
from textdistance import levenshtein
from spellchecker import SpellChecker
from autocorrect import Speller
import requests
import os

class SpellingCorrector:
    def __init__(self):
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize different spell checkers
        self.pyspell_checker = SpellChecker()
        self.autocorrect_speller = Speller(lang='en')
        
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab')
    
    def preprocess_text(self, text):
        """Preprocess text by removing punctuation and converting to lowercase"""
        # Remove punctuation and convert to lowercase
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.lower()
        return text
    
    def tokenize_text(self, text):
        """Tokenize text into words"""
        return nltk.word_tokenize(text)
    
    def correct_with_pyspellchecker(self, word):
        """Correct spelling using PySpellChecker"""
        if word in self.pyspell_checker:
            return word
        
        # Use the correction method which returns the most likely candidate
        correction = self.pyspell_checker.correction(word)
        return correction if correction else word
    
    def correct_with_autocorrect(self, word):
        """Correct spelling using autocorrect library"""
        return self.autocorrect_speller(word)
    
    def correct_with_levenshtein(self, word, threshold=0.8):
        """Correct spelling using Levenshtein distance"""
        if word in self.pyspell_checker:
            return word
        
        best_match = None
        best_score = 0
        
        # Get candidates from PySpellChecker
        candidates = self.pyspell_checker.candidates(word)
        if not candidates:
            return word
        
        # Find best match using Levenshtein distance
        for candidate in candidates:
            distance = levenshtein.normalized_similarity(word, candidate)
            if distance > best_score and distance >= threshold:
                best_score = distance
                best_match = candidate
        
        return best_match if best_match else word
    
    def correct_with_frequency(self, word):
        """Correct spelling using word frequency and common patterns"""
        if word in self.pyspell_checker:
            return word
        
        # Common misspelling patterns for frequent words
        common_fixes = {
            # Determiners and articles
            'teh': 'the', 'a': 'a', 'an': 'an',
            'tehir': 'their', 'thier': 'their', 'there': 'there', 'their': 'their',
            'youre': 'you are', 'your': 'your', 'yoru': 'your',
            
            # Common pronouns
            'i': 'I', 'me': 'me', 'we': 'we', 'us': 'us', 'you': 'you',
            'he': 'he', 'she': 'she', 'it': 'it', 'they': 'they', 'them': 'them',
            
            # Prepositions
            'of': 'of', 'to': 'to', 'for': 'for', 'with': 'with', 'by': 'by',
            'from': 'from', 'in': 'in', 'on': 'on', 'at': 'at',
            
            # Common words
            'and': 'and', 'or': 'or', 'but': 'but', 'if': 'if', 'when': 'when',
            'where': 'where', 'what': 'what', 'who': 'who', 'how': 'how', 'why': 'why',
            
            # Common misspellings
            'recieve': 'receive', 'beleive': 'believe', 'achieve': 'achieve',
            'wierd': 'weird', 'freind': 'friend', 'seperate': 'separate',
            'definately': 'definitely', 'occured': 'occurred'
        }
        
        word_lower = word.lower()
        if word_lower in common_fixes:
            return common_fixes[word_lower]
        
        # If not in common fixes, use frequency-based selection from candidates
        candidates = self.pyspell_checker.candidates(word)
        if not candidates:
            return word
        
        # Get most frequent word from candidates
        # PySpellChecker already returns candidates sorted by frequency
        return list(candidates)[0]
    
    def correct_text(self, text, method='pyspellchecker'):
        """Correct spelling in entire text using specified method"""
        # Preprocess and tokenize
        processed_text = self.preprocess_text(text)
        words = self.tokenize_text(processed_text)
        
        corrected_words = []
        corrections_made = []
        
        for word in words:
            if method == 'pyspellchecker':
                corrected_word = self.correct_with_pyspellchecker(word)
            elif method == 'autocorrect':
                corrected_word = self.correct_with_autocorrect(word)
            elif method == 'levenshtein':
                corrected_word = self.correct_with_levenshtein(word)
            elif method == 'frequency':
                corrected_word = self.correct_with_frequency(word)
            else:
                corrected_word = word
            
            corrected_words.append(corrected_word)
            
            # Track corrections made
            if word != corrected_word:
                correction_info = {
                    'original': word,
                    'corrected': corrected_word,
                    'all_methods': None  # No ensemble, so no multiple methods
                }
                corrections_made.append(correction_info)
        
        return ' '.join(corrected_words), corrections_made
    
    def get_word_suggestions(self, word, n=5):
        """Get multiple suggestions for a misspelled word"""
        suggestions = set()
        
        # Get candidates from PySpellChecker
        pyspell_candidates = self.pyspell_checker.candidates(word)
        if pyspell_candidates:
            suggestions.update(list(pyspell_candidates)[:n])
        
        # Convert to list and return top n
        suggestions = list(suggestions)
        return suggestions[:n]


def main():
    """Main function to demonstrate the spelling corrector"""
    print("Initializing Spelling Corrector...")
    corrector = SpellingCorrector()
    print("Spelling Corrector initialized successfully!")
    
    # Test examples
    test_texts = [
        "The qick brown fox jumps over the lazy dog",
        "I recieved your mesage yestarday",
        "Artifical inteligence is revolutionizing tecnology",
        "Please chck your speling before submiting the documnt"
    ]
    
    print("\n" + "="*60)
    print("SPELLING CORRECTOR DEMONSTRATION")
    print("="*60)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nExample {i}:")
        print(f"Original: {text}")
        
        # Correct using PySpellChecker (best performing method)
        corrected_text, corrections = corrector.correct_text(text, method='pyspellchecker')
        print(f"Corrected: {corrected_text}")
        
        if corrections:
            print("Corrections made:")
            for correction in corrections:
                print(f"  '{correction['original']}' → '{correction['corrected']}'")
        else:
            print("No corrections needed.")
        
        print("-" * 40)
    
    # Interactive mode
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("Enter text to correct (or 'quit' to exit):")
    
    while True:
        user_input = input("\nEnter text: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if not user_input:
            continue
        
        # Show corrections for different methods
        methods = ['pyspellchecker', 'autocorrect', 'frequency', 'levenshtein']
        
        print(f"\nOriginal: {user_input}")
        print("\nCorrections by different methods:")
        
        for method in methods:
            corrected, corrections = corrector.correct_text(user_input, method=method)
            print(f"{method.capitalize():15}: {corrected}")
        
        # Show word suggestions for individual words
        words = corrector.tokenize_text(corrector.preprocess_text(user_input))
        misspelled_words = []
        
        for word in words:
            if word not in corrector.pyspell_checker and len(word) > 2:
                suggestions = corrector.get_word_suggestions(word)
                if suggestions:
                    misspelled_words.append((word, suggestions))
        
        if misspelled_words:
            print("\nWord suggestions:")
            for word, suggestions in misspelled_words:
                print(f"  '{word}': {', '.join(suggestions)}")
    
    print("\nThank you for using the Spelling Corrector!")


if __name__ == "__main__":
    main()
