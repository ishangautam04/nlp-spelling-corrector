#!/usr/bin/env python3
"""
Synthetic Dataset Generator for Spelling Correction Testing
Creates diverse error types where ensemble methods should outperform individual algorithms
"""

import random
import string
import json
from collections import defaultdict

class SpellingErrorGenerator:
    def __init__(self):
        """Initialize error generation patterns"""
        
        # Keyboard layout for realistic typos
        self.keyboard_layout = {
            'a': ['q', 'w', 's', 'x', 'z'],
            'b': ['v', 'g', 'h', 'n'],
            'c': ['x', 'd', 'f', 'v'],
            'd': ['s', 'e', 'r', 'f', 'c', 'x'],
            'e': ['w', 's', 'd', 'r'],
            'f': ['d', 'r', 't', 'g', 'v', 'c'],
            'g': ['f', 't', 'y', 'h', 'b', 'v'],
            'h': ['g', 'y', 'u', 'j', 'n', 'b'],
            'i': ['u', 'j', 'k', 'o'],
            'j': ['h', 'u', 'i', 'k', 'm', 'n'],
            'k': ['j', 'i', 'o', 'l', 'm'],
            'l': ['k', 'o', 'p'],
            'm': ['n', 'j', 'k'],
            'n': ['b', 'h', 'j', 'm'],
            'o': ['i', 'k', 'l', 'p'],
            'p': ['o', 'l'],
            'q': ['a', 'w'],
            'r': ['e', 'd', 'f', 't'],
            's': ['a', 'w', 'e', 'd', 'x', 'z'],
            't': ['r', 'f', 'g', 'y'],
            'u': ['y', 'h', 'j', 'i'],
            'v': ['c', 'f', 'g', 'b'],
            'w': ['q', 'a', 's', 'e'],
            'x': ['z', 'a', 's', 'd', 'c'],
            'y': ['t', 'g', 'h', 'u'],
            'z': ['a', 's', 'x']
        }
        
        # Phonetic confusion patterns
        self.phonetic_patterns = {
            'ph': 'f', 'f': 'ph',
            'ck': 'k', 'k': 'ck',
            'c': 'k', 'k': 'c',
            'sh': 'ch', 'ch': 'sh',
            'th': 'f', 'f': 'th',
            'tion': 'shun', 'sion': 'shun',
            'ough': 'uff', 'augh': 'af',
            'ei': 'ie', 'ie': 'ei',
            'ou': 'ow', 'ow': 'ou'
        }
        
        # OCR confusion patterns (visually similar characters)
        self.ocr_patterns = {
            'l': ['1', 'I', '|'],
            'I': ['l', '1', '|'],
            '1': ['l', 'I', '|'],
            'o': ['0', 'O'],
            'O': ['0', 'o'],
            '0': ['O', 'o'],
            'rn': 'm',
            'm': 'rn',
            'cl': 'd',
            'd': 'cl',
            'vv': 'w',
            'w': 'vv',
            'u': 'n',
            'n': 'u'
        }
        
        # Context-dependent words (homophone confusion)
        self.context_confusion = {
            'their': ['there', 'they\'re'],
            'there': ['their', 'they\'re'],
            'they\'re': ['their', 'there'],
            'your': ['you\'re'],
            'you\'re': ['your'],
            'its': ['it\'s'],
            'it\'s': ['its'],
            'accept': ['except'],
            'except': ['accept'],
            'affect': ['effect'],
            'effect': ['affect'],
            'then': ['than'],
            'than': ['then'],
            'loose': ['lose'],
            'lose': ['loose']
        }
        
        # Common words for testing - expanded with words that challenge different algorithms
        self.test_vocabulary = [
            # Common simple words (good for PySpellChecker and Frequency)
            'the', 'and', 'you', 'that', 'was', 'for', 'are', 'with', 'his', 'they',
            'have', 'this', 'will', 'your', 'from', 'they', 'know', 'want', 'been',
            'good', 'much', 'some', 'time', 'very', 'when', 'come', 'here', 'just',
            'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them',
            'well', 'were', 'work', 'first', 'also', 'after', 'back', 'other',
            'people', 'because', 'before', 'between', 'different', 'example',
            
            # Commonly misspelled words (good for AutoCorrect)
            'beautiful', 'definitely', 'separate', 'receive', 'believe', 'weird',
            'achievement', 'knowledge', 'embarrass', 'occurrence', 'maintenance',
            'necessary', 'beginning', 'restaurant', 'tomorrow', 'February',
            'accommodate', 'argument', 'calendar', 'colonel', 'conscience',
            'discipline', 'existence', 'foreign', 'government', 'harass',
            'independent', 'jewelry', 'liaison', 'millennium', 'noticeable',
            
            # Words good for Levenshtein (similar character patterns)
            'algorithm', 'analysis', 'application', 'architecture', 'authentication',
            'background', 'bandwidth', 'bootstrap', 'brightness', 'calculate',
            'certificate', 'challenge', 'character', 'collaboration', 'communication',
            'comparison', 'complexity', 'component', 'compression', 'configuration',
            
            # Words good for Frequency (very common words)
            'about', 'above', 'across', 'actually', 'again', 'against', 'almost',
            'already', 'always', 'among', 'another', 'anyone', 'anything', 'around',
            'because', 'become', 'being', 'below', 'better', 'business',
            
            # Technical and longer words (challenging for all)
            'development', 'experience', 'management', 'international', 'technology',
            'information', 'education', 'environment', 'professional', 'organization',
            'implementation', 'infrastructure', 'investigation', 'recommendation',
            'responsibility', 'understanding', 'characteristic', 'administration'
        ]
    
    def generate_keyboard_error(self, word):
        """Generate keyboard-adjacent character substitution errors"""
        if len(word) < 2:
            return word
        
        # Choose random position (avoid first/last for readability)
        pos = random.randint(1, len(word) - 2)
        char = word[pos].lower()
        
        if char in self.keyboard_layout:
            # Replace with adjacent key
            replacement = random.choice(self.keyboard_layout[char])
            return word[:pos] + replacement + word[pos+1:]
        
        return word
    
    def generate_phonetic_error(self, word):
        """Generate phonetic confusion errors"""
        word_lower = word.lower()
        
        for pattern, replacement in self.phonetic_patterns.items():
            if pattern in word_lower:
                # Replace first occurrence
                new_word = word_lower.replace(pattern, replacement, 1)
                # Preserve original capitalization pattern
                if word[0].isupper():
                    new_word = new_word.capitalize()
                return new_word
        
        return word
    
    def generate_ocr_error(self, word):
        """Generate OCR-like visual confusion errors"""
        if len(word) < 2:
            return word
        
        word_list = list(word)
        pos = random.randint(0, len(word) - 1)
        char = word[pos].lower()
        
        if char in self.ocr_patterns:
            replacements = self.ocr_patterns[char]
            if isinstance(replacements, list):
                replacement = random.choice(replacements)
            else:
                replacement = replacements
            
            if len(replacement) == 1:
                word_list[pos] = replacement
            else:
                # Handle multi-character replacements like 'rn' -> 'm'
                if char == 'm' and pos > 0:
                    word_list[pos-1:pos+1] = list(replacement)
                elif char in ['r', 'n'] and pos < len(word) - 1:
                    word_list[pos:pos+2] = [replacement]
        
        return ''.join(word_list)
    
    def generate_context_error(self, word):
        """Generate context-dependent word confusion"""
        word_lower = word.lower()
        
        if word_lower in self.context_confusion:
            confusions = self.context_confusion[word_lower]
            replacement = random.choice(confusions)
            
            # Preserve capitalization
            if word[0].isupper():
                replacement = replacement.capitalize()
            
            return replacement
        
        return word
    
    def generate_character_level_errors(self, word):
        """Generate various character-level errors"""
        if len(word) < 3:
            return word
        
        error_type = random.choice(['substitution', 'deletion', 'insertion', 'transposition'])
        
        if error_type == 'substitution':
            pos = random.randint(1, len(word) - 2)
            char = random.choice(string.ascii_lowercase)
            return word[:pos] + char + word[pos+1:]
        
        elif error_type == 'deletion':
            pos = random.randint(1, len(word) - 2)
            return word[:pos] + word[pos+1:]
        
        elif error_type == 'insertion':
            pos = random.randint(1, len(word) - 1)
            char = random.choice(string.ascii_lowercase)
            return word[:pos] + char + word[pos:]
        
        elif error_type == 'transposition':
            if len(word) >= 4:
                pos = random.randint(1, len(word) - 3)
                return word[:pos] + word[pos+1] + word[pos] + word[pos+2:]
        
        return word
    
    def generate_test_dataset(self, num_samples=1000):
        """Generate comprehensive test dataset with various error types"""
        dataset = []
        
        # Ensure we have examples of each error type
        # Each error type tests different algorithm strengths
        error_types = [
            ('keyboard', self.generate_keyboard_error),      # Good for Levenshtein
            ('phonetic', self.generate_phonetic_error),      # Good for AutoCorrect
            ('character', self.generate_character_level_errors), # Good for PySpellChecker
            ('simple_typo', self.generate_keyboard_error),   # Good for Frequency
        ]
        
        samples_per_type = num_samples // len(error_types)
        
        for error_type, error_function in error_types:
            for _ in range(samples_per_type):
                # Choose random word
                correct_word = random.choice(self.test_vocabulary)
                
                # Generate error
                error_word = error_function(correct_word)
                
                # Ensure we actually created an error
                if error_word != correct_word:
                    dataset.append({
                        'correct': correct_word,
                        'error': error_word,
                        'error_type': error_type,
                        'context': self._generate_context(correct_word)
                    })
        
        # No mixed error types - keep it simple to show each algorithm's strength
        return dataset
    
    def _generate_context(self, word):
        """Generate realistic context sentences for words"""
        context_templates = [
            f"The {word} is very important in this situation.",
            f"We need to consider the {word} carefully.",
            f"This {word} has been discussed many times.",
            f"The {word} shows interesting patterns.",
            f"I believe the {word} is correct.",
            f"The {word} appears frequently in texts.",
            f"We should focus on the {word} first.",
            f"The {word} demonstrates the concept well."
        ]
        
        return random.choice(context_templates)
    
    def save_dataset(self, dataset, filename):
        """Save dataset to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset saved to {filename}")
        print(f"Total samples: {len(dataset)}")
        
        # Print statistics
        error_counts = defaultdict(int)
        for sample in dataset:
            error_counts[sample['error_type']] += 1
        
        print("\nError type distribution:")
        for error_type, count in sorted(error_counts.items()):
            print(f"  {error_type}: {count}")

def main():
    """Generate and save synthetic dataset"""
    generator = SpellingErrorGenerator()
    
    print("Generating synthetic spelling error dataset...")
    
    # Generate different sizes for testing
    datasets = {
        'small': 500,
        'medium': 1500,
        'large': 3000
    }
    
    for size_name, num_samples in datasets.items():
        print(f"\nGenerating {size_name} dataset ({num_samples} samples)...")
        dataset = generator.generate_test_dataset(num_samples)
        filename = f"spelling_test_dataset_{size_name}.json"
        generator.save_dataset(dataset, filename)
    
    print("\n" + "="*50)
    print("Dataset generation complete!")
    print("Use these datasets to test ensemble vs individual algorithms.")

if __name__ == "__main__":
    main()