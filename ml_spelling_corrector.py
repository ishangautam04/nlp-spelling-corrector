#!/usr/bin/env python3
"""
ML-Enhanced Spelling Corrector
Integrates trained machine learning models with traditional algorithms
"""

import pickle
import os
from spelling_corrector import SpellingCorrector

class MLSpellingCorrector(SpellingCorrector):
    """Enhanced spelling corrector with ML model integration"""
    
    def __init__(self, model_path=None):
        """Initialize with optional trained model"""
        super().__init__()
        self.ml_model = None
        self.ml_trained = False
        
        # Load ML model if provided
        if model_path and os.path.exists(model_path):
            self.load_ml_model(model_path)
    
    def load_ml_model(self, model_path):
        """Load trained ML model"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.ml_model = model_data
            self.ml_trained = model_data.get('trained', False)
            print(f"ML model loaded from {model_path}")
            
        except Exception as e:
            print(f"Failed to load ML model: {e}")
            self.ml_model = None
            self.ml_trained = False
    
    def correct_with_ml_model(self, word):
        """Correct spelling using trained ML model"""
        if not self.ml_model:
            return word
        
        word_lower = word.lower().strip()
        
        # Check pattern-based corrections first (fastest)
        pattern_model = self.ml_model.get('pattern_model', {})
        if word_lower in pattern_model:
            corrections = pattern_model[word_lower]
            # Return most frequent correction
            return max(corrections.items(), key=lambda x: x[1])[0]
        
        # Use ML model if available and trained
        if self.ml_trained and self.ml_model.get('vectorizer') and self.ml_model.get('ml_model'):
            try:
                vectorizer = self.ml_model['vectorizer']
                ml_model = self.ml_model['ml_model']
                
                word_vectorized = vectorizer.transform([word_lower])
                prediction = ml_model.predict(word_vectorized)[0]
                return prediction
                
            except Exception as e:
                # Fallback to original if ML fails
                pass
        
        return word
    
    def ensemble_correction_with_ml(self, word, prev_word="", next_word=""):
        """Enhanced ensemble that includes ML model"""
        # Get base ensemble result
        base_correction, base_corrections = super().ensemble_correction(word, prev_word, next_word)
        
        # If ML model is available, include its prediction
        if self.ml_model:
            ml_correction = self.correct_with_ml_model(word)
            
            # Add ML prediction to the mix
            all_corrections = dict(base_corrections)
            all_corrections['ml_model'] = ml_correction
            
            # Enhanced weighted voting including ML
            weights = {
                'ml_model': 4.5,          # High weight for trained model
                'pyspellchecker': 4.0,    # Best overall statistical accuracy
                'frequency': 3.5,         # Excellent for common words
                'autocorrect': 2.5,       # Good machine learning approach
                'levenshtein': 2.0        # Good for character-level similarity
            }
            
            # Calculate weighted votes
            weighted_votes = {}
            for method, correction in all_corrections.items():
                weight = weights.get(method, 1.0)
                if correction in weighted_votes:
                    weighted_votes[correction] += weight
                else:
                    weighted_votes[correction] = weight
            
            # Get the correction with highest weighted score
            best_correction = max(weighted_votes.items(), key=lambda x: x[1])
            
            return best_correction[0], all_corrections
        
        return base_correction, base_corrections
    
    def correct_text_with_ml(self, text, method='ensemble_ml'):
        """Correct text using ML-enhanced methods"""
        if method == 'ensemble_ml':
            # Use ML-enhanced ensemble
            words = text.split()
            corrected_words = []
            corrections_log = []
            
            for i, word in enumerate(words):
                # Get context
                prev_word = words[i-1] if i > 0 else ""
                next_word = words[i+1] if i < len(words)-1 else ""
                
                # Check if word needs correction
                if word not in self.pyspell_checker:
                    corrected_word, method_results = self.ensemble_correction_with_ml(word, prev_word, next_word)
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
            
        elif method == 'ml_only':
            # Use only ML model
            words = text.split()
            corrected_words = []
            
            for word in words:
                corrected_words.append(self.correct_with_ml_model(word))
            
            return " ".join(corrected_words)
        
        else:
            # Fall back to parent methods
            return super().correct_text(text, method)

def main():
    """Demonstration of ML-enhanced spelling corrector"""
    print("=== ML-Enhanced Spelling Corrector ===")
    
    # Check if ML model exists
    model_path = "spelling_correction_model.pkl"
    
    if not os.path.exists(model_path):
        print(f"ML model not found at {model_path}")
        print("Please run 'python train_spelling_model.py' first to create the model")
        print("Using standard spelling corrector...")
        corrector = MLSpellingCorrector()
    else:
        print(f"Loading ML model from {model_path}")
        corrector = MLSpellingCorrector(model_path)
    
    # Test examples
    test_cases = [
        "teh qick brown fox",
        "I recieve your mesage",
        "definately seperate the groups",
        "thier beleif is wierd"
    ]
    
    print("\n=== Testing ML-Enhanced Correction ===")
    
    for text in test_cases:
        print(f"\nOriginal: {text}")
        
        # Standard ensemble
        standard_result = corrector.correct_text(text, method='ensemble')
        if isinstance(standard_result, tuple):
            standard_corrected = standard_result[0]
        else:
            standard_corrected = standard_result
        print(f"Standard: {standard_corrected}")
        
        # ML-enhanced ensemble
        if corrector.ml_model:
            ml_result = corrector.correct_text_with_ml(text, method='ensemble_ml')
            if isinstance(ml_result, tuple):
                ml_corrected = ml_result[0]
            else:
                ml_corrected = ml_result
            print(f"ML-Enhanced: {ml_corrected}")
        
        print("-" * 50)
    
    print("\n=== Performance Comparison ===")
    print("The ML-enhanced corrector combines:")
    print("1. Traditional algorithms (PySpellChecker, AutoCorrect, etc.)")
    print("2. Custom trained patterns from datasets")
    print("3. Machine learning predictions")
    print("4. Weighted ensemble voting")

if __name__ == "__main__":
    main()