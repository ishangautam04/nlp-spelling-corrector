#!/usr/bin/env python3
"""
ML Training System for Spelling Correction
Uses datasets to train machine learning models for improved spell checking
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import nltk
from collections import defaultdict, Counter
import re
import os

class SpellingCorrectionTrainer:
    def __init__(self):
        """Initialize the spelling correction trainer"""
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.correction_patterns = defaultdict(Counter)
        self.context_patterns = defaultdict(list)
        self.trained = False
        
    def load_dataset(self, dataset_path, format_type="csv"):
        """
        Load spelling correction dataset
        
        Expected formats:
        CSV: columns ['incorrect', 'correct'] or ['mistake', 'correction']
        JSON: [{"incorrect": "teh", "correct": "the"}, ...]
        TXT: Each line: "incorrect,correct" or "incorrect -> correct"
        """
        print(f"Loading dataset from {dataset_path}...")
        
        if format_type == "csv":
            df = pd.read_csv(dataset_path)
            # Try different common column names
            if 'incorrect' in df.columns and 'correct' in df.columns:
                return df[['incorrect', 'correct']].dropna()
            elif 'mistake' in df.columns and 'correction' in df.columns:
                return df[['mistake', 'correction']].dropna().rename(columns={'mistake': 'incorrect', 'correction': 'correct'})
            elif 'wrong' in df.columns and 'right' in df.columns:
                return df[['wrong', 'right']].dropna().rename(columns={'wrong': 'incorrect', 'right': 'correct'})
            else:
                print(f"Unknown CSV format. Columns: {df.columns.tolist()}")
                print("Expected columns: 'incorrect' and 'correct' (or similar)")
                return None
                
        elif format_type == "json":
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            return df[['incorrect', 'correct']].dropna()
            
        elif format_type == "txt":
            data = []
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if '->' in line:
                        incorrect, correct = line.split('->', 1)
                        data.append({'incorrect': incorrect.strip(), 'correct': correct.strip()})
                    elif ',' in line:
                        parts = line.split(',', 1)
                        if len(parts) == 2:
                            data.append({'incorrect': parts[0].strip(), 'correct': parts[1].strip()})
            return pd.DataFrame(data)
        
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def create_sample_dataset(self):
        """Create a sample dataset for testing"""
        sample_data = [
            # Common typos
            {'incorrect': 'teh', 'correct': 'the'},
            {'incorrect': 'thier', 'correct': 'their'},
            {'incorrect': 'recieve', 'correct': 'receive'},
            {'incorrect': 'seperate', 'correct': 'separate'},
            {'incorrect': 'definately', 'correct': 'definitely'},
            {'incorrect': 'occured', 'correct': 'occurred'},
            {'incorrect': 'begining', 'correct': 'beginning'},
            {'incorrect': 'beleive', 'correct': 'believe'},
            {'incorrect': 'wierd', 'correct': 'weird'},
            {'incorrect': 'freind', 'correct': 'friend'},
            
            # Transpositions
            {'incorrect': 'form', 'correct': 'from'},
            {'incorrect': 'mose', 'correct': 'most'},
            {'incorrect': 'jsut', 'correct': 'just'},
            {'incorrect': 'waht', 'correct': 'what'},
            {'incorrect': 'whcih', 'correct': 'which'},
            
            # Missing letters
            {'incorrect': 'wich', 'correct': 'which'},
            {'incorrect': 'becaus', 'correct': 'because'},
            {'incorrect': 'befor', 'correct': 'before'},
            {'incorrect': 'wth', 'correct': 'with'},
            {'incorrect': 'hav', 'correct': 'have'},
            
            # Extra letters
            {'incorrect': 'whith', 'correct': 'with'},
            {'incorrect': 'whenn', 'correct': 'when'},
            {'incorrect': 'thenn', 'correct': 'then'},
            {'incorrect': 'seee', 'correct': 'see'},
            {'incorrect': 'goood', 'correct': 'good'},
            
            # Common word confusions
            {'incorrect': 'youre', 'correct': 'your'},
            {'incorrect': 'its', 'correct': "it's"},
            {'incorrect': 'alot', 'correct': 'a lot'},
            {'incorrect': 'loose', 'correct': 'lose'},
            {'incorrect': 'affect', 'correct': 'effect'},
        ]
        
        return pd.DataFrame(sample_data)
    
    def extract_features(self, incorrect_word, correct_word=None):
        """Extract features from word pairs for ML training"""
        features = {
            'length': len(incorrect_word),
            'length_diff': len(incorrect_word) - len(correct_word) if correct_word else 0,
            'starts_with': incorrect_word[0] if incorrect_word else '',
            'ends_with': incorrect_word[-1] if incorrect_word else '',
            'has_double_letters': bool(re.search(r'(.)\1', incorrect_word)),
            'vowel_count': len(re.findall(r'[aeiouAEIOU]', incorrect_word)),
            'consonant_count': len(re.findall(r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]', incorrect_word)),
        }
        
        # Character-level features
        if correct_word:
            # Edit distance approximation
            if len(incorrect_word) == len(correct_word):
                features['char_substitutions'] = sum(1 for a, b in zip(incorrect_word, correct_word) if a != b)
            else:
                features['char_substitutions'] = abs(len(incorrect_word) - len(correct_word))
                
        return features
    
    def train_model(self, df):
        """Train the ML model on the dataset"""
        print("Training spelling correction model...")
        
        # Prepare training data
        X_text = []
        y = []
        
        for _, row in df.iterrows():
            incorrect = str(row['incorrect']).lower().strip()
            correct = str(row['correct']).lower().strip()
            
            if incorrect and correct and incorrect != correct:
                # Store correction patterns
                self.correction_patterns[incorrect][correct] += 1
                
                # Prepare text features for vectorization
                X_text.append(incorrect)
                y.append(correct)
        
        print(f"Training on {len(X_text)} correction pairs...")
        
        # Train pattern-based model
        self.pattern_model = dict(self.correction_patterns)
        
        # If we have enough data, train ML model
        if len(set(y)) > 10:  # At least 10 unique corrections
            try:
                X_vectorized = self.vectorizer.fit_transform(X_text)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_vectorized, y, test_size=0.2, random_state=42
                )
                
                # Train model
                self.model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = self.model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"Model accuracy: {accuracy:.3f}")
                
                self.trained = True
                
            except Exception as e:
                print(f"ML training failed: {e}")
                print("Using pattern-based approach only")
                self.trained = False
        else:
            print("Not enough data for ML training. Using pattern-based approach only")
            self.trained = False
    
    def predict_correction(self, word):
        """Predict correction for a misspelled word"""
        word = word.lower().strip()
        
        # First check direct pattern matching
        if word in self.pattern_model:
            # Return most common correction
            return self.pattern_model[word].most_common(1)[0][0]
        
        # If trained, use ML model
        if self.trained:
            try:
                word_vectorized = self.vectorizer.transform([word])
                prediction = self.model.predict(word_vectorized)[0]
                return prediction
            except:
                pass
        
        # Fallback: return original word
        return word
    
    def save_model(self, model_path):
        """Save the trained model"""
        model_data = {
            'pattern_model': dict(self.pattern_model),
            'vectorizer': self.vectorizer if self.trained else None,
            'ml_model': self.model if self.trained else None,
            'trained': self.trained
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """Load a trained model"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.pattern_model = model_data['pattern_model']
        self.vectorizer = model_data.get('vectorizer')
        self.model = model_data.get('ml_model')
        self.trained = model_data.get('trained', False)
        
        print(f"Model loaded from {model_path}")

def main():
    """Main training function"""
    trainer = SpellingCorrectionTrainer()
    
    print("=== Spelling Correction Model Trainer ===")
    print()
    
    # Option 1: Use sample dataset
    print("1. Creating sample dataset for demonstration...")
    df = trainer.create_sample_dataset()
    print(f"Sample dataset created with {len(df)} examples")
    
    # Train on sample data
    trainer.train_model(df)
    
    # Test the trained model
    print("\n=== Testing Trained Model ===")
    test_words = ['teh', 'thier', 'definately', 'seperate', 'recieve']
    
    for word in test_words:
        correction = trainer.predict_correction(word)
        print(f"'{word}' -> '{correction}'")
    
    # Save model
    model_path = "spelling_correction_model.pkl"
    trainer.save_model(model_path)
    
    print(f"\n=== Model saved to {model_path} ===")
    print("You can now use this model in your spelling corrector!")
    
    # Instructions for using custom datasets
    print("\n=== Using Custom Datasets ===")
    print("To train on your own dataset:")
    print("1. Prepare CSV with columns 'incorrect' and 'correct'")
    print("2. Use: trainer.load_dataset('your_dataset.csv')")
    print("3. Call: trainer.train_model(df)")
    print()
    print("Popular datasets to try:")
    print("- Wikipedia edit history")
    print("- Google Web 1T 5-gram spelling corrections")
    print("- Peter Norvig's spell correction dataset")
    print("- Aspell dictionary corrections")

if __name__ == "__main__":
    main()