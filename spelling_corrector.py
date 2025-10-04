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
    
    def ensemble_correction(self, word, prev_word="", next_word=""):
        """High-performance ensemble with only the best algorithms"""
        # Only use the top-performing algorithms
        corrections = {
            'pyspellchecker': self.correct_with_pyspellchecker(word),
            'frequency': self.correct_with_frequency(word),
            'autocorrect': self.correct_with_autocorrect(word),
            'levenshtein': self.correct_with_levenshtein(word)
        }
        
        # Optimized weights based on performance testing
        weights = {
            'pyspellchecker': 4.0,    # Best overall statistical accuracy
            'frequency': 3.5,         # Excellent for common words like "teh"
            'autocorrect': 2.5,       # Good machine learning approach
            'levenshtein': 2.0        # Good for character-level similarity
        }
        
        # Calculate weighted votes
        weighted_votes = {}
        for method, correction in corrections.items():
            weight = weights.get(method, 1.0)
            if correction in weighted_votes:
                weighted_votes[correction] += weight
            else:
                weighted_votes[correction] = weight
        
        # Get the correction with highest weighted score
        best_correction = max(weighted_votes.items(), key=lambda x: x[1])
        majority_choice = best_correction[0]
        
        # Apply context-based override if we have context and there's competition
        if (prev_word or next_word) and len(weighted_votes) > 1:
            # Get top 2 candidates
            sorted_votes = sorted(weighted_votes.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_votes) >= 2:
                top_candidate = sorted_votes[0][0]
                second_candidate = sorted_votes[1][0]
                
                # If the scores are close (within 1.0), use context to decide
                if sorted_votes[0][1] - sorted_votes[1][1] <= 1.0:
                    candidates = [top_candidate, second_candidate]
                    context_choice = self._score_with_context(word, candidates, prev_word.lower(), next_word.lower())
                    if context_choice != top_candidate:
                        majority_choice = context_choice
        
        return majority_choice, corrections
    
    def correct_text(self, text, method='ensemble'):
        """Correct entire text with specified method"""
        if method == 'context':
            return self.correct_with_context(text)
        elif method == 'ensemble':
            # Enhanced ensemble with context awareness
            words = text.split()
            corrected_words = []
            corrections_log = []
            
            for i, word in enumerate(words):
                # Get context
                prev_word = words[i-1] if i > 0 else ""
                next_word = words[i+1] if i < len(words)-1 else ""
                
                # Check if word needs correction
                if word not in self.pyspell_checker:
                    corrected_word, method_results = self.ensemble_correction(word, prev_word, next_word)
                    corrected_words.append(corrected_word)
                    
                    if corrected_word != word:
                        corrections_log.append({
                            'original': word,
                            'corrected': corrected_word,
                            'all_methods': method_results
                        })
                else:
                    corrected_words.append(word)
            
            return " ".join(corrected_words), corrections_log
        else:
            # Single method correction word by word
            words = self.tokenize(text)
            corrected_words = []
            
            for word in words:
                if method == 'pyspellchecker':
                    corrected_words.append(self.correct_with_pyspellchecker(word))
                elif method == 'autocorrect':
                    corrected_words.append(self.correct_with_autocorrect(word))
                elif method == 'levenshtein':
                    corrected_words.append(self.correct_with_levenshtein(word))
                elif method == 'frequency':
                    corrected_words.append(self.correct_with_frequency(word))
                else:
                    corrected_words.append(word)
            
            return " ".join(corrected_words)
    
    def correct_text(self, text, method='ensemble'):
        """Correct spelling in entire text"""
        # Preprocess and tokenize
        processed_text = self.preprocess_text(text)
        words = self.tokenize_text(processed_text)
        
        corrected_words = []
        corrections_made = []
        
        for word in words:
            if method == 'ensemble':
                corrected_word, all_corrections = self.ensemble_correction(word)
            elif method == 'pyspellchecker':
                corrected_word = self.correct_with_pyspellchecker(word)
                all_corrections = None
            elif method == 'autocorrect':
                corrected_word = self.correct_with_autocorrect(word)
                all_corrections = None
            elif method == 'levenshtein':
                corrected_word = self.correct_with_levenshtein(word)
                all_corrections = None
            elif method == 'frequency':
                corrected_word = self.correct_with_frequency(word)
                all_corrections = None
            else:
                corrected_word = word
                all_corrections = None
            
            corrected_words.append(corrected_word)
            
            # Track corrections made
            if word != corrected_word:
                correction_info = {
                    'original': word,
                    'corrected': corrected_word,
                    'all_methods': all_corrections
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
        
        # Correct using ensemble method
        corrected_text, corrections = corrector.correct_text(text, method='ensemble')
        print(f"Corrected: {corrected_text}")
        
        if corrections:
            print("Corrections made:")
            for correction in corrections:
                print(f"  '{correction['original']}' → '{correction['corrected']}'")
                if correction['all_methods']:
                    print(f"    Methods used: {correction['all_methods']}")
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
        methods = ['ensemble', 'pyspellchecker', 'autocorrect', 'levenshtein']
        
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
