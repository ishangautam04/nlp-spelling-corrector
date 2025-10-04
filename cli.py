#!/usr/bin/env python3
"""
Command Line Interface for S    # Method selection
    parser.add_argument('-m', '--met        print("Comparing all correction methods:")
        methods = ['ensemble', 'context', 'pyspellchecker', 'autocorrect', 'levenshtein', 'frequency']d', 
                       choices=['ensemble', 'context', 'pyspe            elif user_input.lower() == 'methods':
                methods = ['ensemble', 'context', 'pyspellchecker', 'autocorrect', 'levenshtein', 'frequency']
                print(f"\nAvailable methods: {', '.join(methods)}")
                print(f"Current method: {current_method}")
                continue
            
            elif user_input.startswith('method '):
                new_method = user_input[7:].strip()
                valid_methods = ['ensemble', 'context', 'pyspellchecker', 'autocorrect', 'levenshtein', 'frequency']', 
                               'autocorrect', 'levenshtein'], 
                       default='ensemble',
                       help='Spelling correction method (default: ensemble)')g Corrector
Usage: python cli.py [options]
"""

import argparse
import sys
import os
from spelling_corrector import SpellingCorrector

def read_file(file_path):
    """Read text from a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def write_file(file_path, content):
    """Write text to a file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        return True
    except Exception as e:
        print(f"Error writing file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="NLP Spelling Corrector - Correct spelling errors in text using various NLP methods",
        epilog="Examples:\n"
               "  python cli.py -t 'The qick brown fox'\n"
               "  python cli.py -f input.txt -o output.txt\n"
               "  python cli.py --interactive\n"
               "  python cli.py -t 'hello wrold' -m ensemble --verbose",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-t', '--text', 
                           help='Text to correct')
    input_group.add_argument('-f', '--file', 
                           help='Input file path')
    input_group.add_argument('--interactive', action='store_true',
                           help='Run in interactive mode')
    
    # Output options
    parser.add_argument('-o', '--output',
                       help='Output file path (default: print to console)')
    
    # Method selection
    parser.add_argument('-m', '--method', 
                       choices=['ensemble', 'pyspellchecker', 
                               'autocorrect', 'levenshtein', 'frequency'],
                       default='ensemble',
                       help='Correction method to use (default: ensemble)')
    
    # Options
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed correction information')
    parser.add_argument('--suggestions', action='store_true',
                       help='Show alternative suggestions for corrections')
    parser.add_argument('--compare', action='store_true',
                       help='Compare results from all methods')
    
    args = parser.parse_args()
    
    # Initialize corrector
    print("Initializing spelling corrector...")
    try:
        corrector = SpellingCorrector()
        print("Spelling corrector loaded successfully!\n")
    except Exception as e:
        print(f"Error initializing corrector: {e}")
        sys.exit(1)
    
    if args.interactive:
        run_interactive_mode(corrector)
    else:
        # Get input text
        if args.text:
            input_text = args.text
        elif args.file:
            input_text = read_file(args.file)
            if input_text is None:
                sys.exit(1)
        
        # Process text
        process_text(corrector, input_text, args)

def process_text(corrector, text, args):
    """Process text with the specified options"""
    print(f"Original text: {text}")
    print("-" * 50)
    
    if args.compare:
        # Compare all methods
        print("Comparing all correction methods:")
        methods = ['ensemble', 'pyspellchecker', 'autocorrect', 'levenshtein', 'frequency']
        
        for method in methods:
            try:
                result = corrector.correct_text(text, method=method)
                if isinstance(result, tuple):
                    corrected, corrections = result
                else:
                    corrected = result
                    corrections = []
                print(f"\n{method.capitalize():15}: {corrected}")
                if args.verbose and corrections:
                    for correction in corrections:
                        print(f"  '{correction['original']}' → '{correction['corrected']}'")
            except Exception as e:
                print(f"\n{method.capitalize():15}: Error - {e}")
    else:
        # Single method correction
        corrected_text, corrections = corrector.correct_text(text, method=args.method)
        print(f"Corrected text: {corrected_text}")
        
        if args.verbose and corrections:
            print(f"\nCorrections made ({len(corrections)}):")
            for correction in corrections:
                print(f"  '{correction['original']}' → '{correction['corrected']}'")
                if correction.get('all_methods') and args.method == 'ensemble':
                    print(f"    All methods: {correction['all_methods']}")
        
        if args.suggestions:
            words = corrector.tokenize_text(corrector.preprocess_text(text))
            print(f"\nWord suggestions:")
            for word in set(words):
                if word not in corrector.word_freq and len(word) > 2:
                    suggestions = corrector.get_word_suggestions(word)
                    if suggestions:
                        print(f"  '{word}': {', '.join(suggestions)}")
        
        # Save to file if specified
        if args.output:
            if write_file(args.output, corrected_text):
                print(f"\nCorrected text saved to: {args.output}")
            else:
                sys.exit(1)

def run_interactive_mode(corrector):
    """Run the corrector in interactive mode"""
    print("="*60)
    print("INTERACTIVE SPELLING CORRECTOR")
    print("="*60)
    print("Commands:")
    print("  'quit' or 'exit' - Exit the program")
    print("  'help' - Show this help message")
    print("  'methods' - List available correction methods")
    print("  'method <name>' - Change correction method")
    print("  'suggestions <word>' - Get suggestions for a word")
    print("-" * 60)
    
    current_method = 'ensemble'
    
    while True:
        try:
            user_input = input(f"\n[{current_method}] Enter text: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            elif user_input.lower() == 'help':
                print("\nCommands:")
                print("  'quit' or 'exit' - Exit the program")
                print("  'help' - Show this help message")
                print("  'methods' - List available correction methods")
                print("  'method <name>' - Change correction method")
                print("  'suggestions <word>' - Get suggestions for a word")
                continue
            
            elif user_input.lower() == 'methods':
                methods = ['ensemble', 'pyspellchecker', 'autocorrect', 'levenshtein']
                print(f"\nAvailable methods: {', '.join(methods)}")
                print(f"Current method: {current_method}")
                continue
            
            elif user_input.lower().startswith('method '):
                new_method = user_input[7:].strip()
                valid_methods = ['ensemble', 'pyspellchecker', 'autocorrect', 'levenshtein']
                if new_method in valid_methods:
                    current_method = new_method
                    print(f"Method changed to: {current_method}")
                else:
                    print(f"Invalid method. Available: {', '.join(valid_methods)}")
                continue
            
            elif user_input.lower().startswith('suggestions '):
                word = user_input[12:].strip()
                suggestions = corrector.get_word_suggestions(word)
                if suggestions:
                    print(f"Suggestions for '{word}': {', '.join(suggestions)}")
                else:
                    print(f"No suggestions found for '{word}'")
                continue
            
            elif not user_input:
                continue
            
            # Correct the text
            corrected_text, corrections = corrector.correct_text(user_input, method=current_method)
            
            print(f"Original:  {user_input}")
            print(f"Corrected: {corrected_text}")
            
            if corrections:
                print(f"Corrections made:")
                for correction in corrections:
                    print(f"  '{correction['original']}' → '{correction['corrected']}'")
            else:
                print("No corrections needed.")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            print("\n\nGoodbye!")
            break

if __name__ == "__main__":
    main()
