#!/usr/bin/env python3
"""
NLP Spelling Corrector - Comparison-Focused Streamlit App
Compares 4 individual algorithms without ensemble approach
Uses the existing SpellingCorrector class for all algorithms
"""

import streamlit as st
import pandas as pd
import time
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from spelling_corrector import SpellingCorrector

# Page Configuration
st.set_page_config(
    page_title="Spelling Correction Comparison",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .algorithm-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    .correction-highlight {
        background-color: #fff3cd;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: bold;
    }
    .winner-badge {
        background-color: #28a745;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .error-type-badge {
        background-color: #dc3545;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        margin: 2px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Initialize spelling corrector
@st.cache_resource
def initialize_corrector():
    """Initialize the spelling corrector with all 4 algorithms"""
    return SpellingCorrector()

def correct_text_with_method(text, corrector, method_name):
    """Correct text using specified method from SpellingCorrector"""
    words = corrector.tokenize_text(corrector.preprocess_text(text))
    corrections = []
    
    if method_name == 'pyspellchecker':
        corrected_words = []
        for word in words:
            corrected = corrector.correct_with_pyspellchecker(word)
            if corrected != word:
                corrections.append({'original': word, 'corrected': corrected})
            corrected_words.append(corrected)
        return ' '.join(corrected_words), corrections
    
    elif method_name == 'autocorrect':
        corrected_words = []
        for word in words:
            corrected = corrector.correct_with_autocorrect(word)
            if corrected != word:
                corrections.append({'original': word, 'corrected': corrected})
            corrected_words.append(corrected)
        return ' '.join(corrected_words), corrections
    
    elif method_name == 'frequency':
        corrected_words = []
        for word in words:
            corrected = corrector.correct_with_frequency(word)
            if corrected != word:
                corrections.append({'original': word, 'corrected': corrected})
            corrected_words.append(corrected)
        return ' '.join(corrected_words), corrections
    
    elif method_name == 'levenshtein':
        corrected_words = []
        for word in words:
            corrected = corrector.correct_with_levenshtein(word)
            if corrected != word:
                corrections.append({'original': word, 'corrected': corrected})
            corrected_words.append(corrected)
        return ' '.join(corrected_words), corrections
    
    else:
        return text, []

def run_all_algorithms(text, corrector):
    """Run all 4 algorithms and return results"""
    results = {}
    
    # PySpellChecker
    start = time.time()
    corrected, corrections = correct_text_with_method(text, corrector, 'pyspellchecker')
    results['PySpellChecker'] = {
        'corrected': corrected,
        'corrections': corrections,
        'time': (time.time() - start) * 1000,
        'accuracy_score': 92.6  # From evaluation
    }
    
    # AutoCorrect
    start = time.time()
    corrected, corrections = correct_text_with_method(text, corrector, 'autocorrect')
    results['AutoCorrect'] = {
        'corrected': corrected,
        'corrections': corrections,
        'time': (time.time() - start) * 1000,
        'accuracy_score': 91.2  # From evaluation
    }
    
    # Frequency-Based
    start = time.time()
    corrected, corrections = correct_text_with_method(text, corrector, 'frequency')
    results['Frequency-Based'] = {
        'corrected': corrected,
        'corrections': corrections,
        'time': (time.time() - start) * 1000,
        'accuracy_score': 75.3  # From evaluation
    }
    
    # Levenshtein
    start = time.time()
    corrected, corrections = correct_text_with_method(text, corrector, 'levenshtein')
    results['Levenshtein'] = {
        'corrected': corrected,
        'corrections': corrections,
        'time': (time.time() - start) * 1000,
        'accuracy_score': 64.2  # From evaluation
    }
    
    return results

def display_comparison_results(results, original_text):
    """Display comparison results in an organized way"""
    
    st.markdown("### 📊 Algorithm Comparison Results")
    
    # Create 4 columns for each algorithm
    cols = st.columns(4)
    
    for idx, (algo_name, result) in enumerate(results.items()):
        with cols[idx]:
            st.markdown(f"**{algo_name}**")
            st.metric("Corrections", len(result['corrections']))
            st.metric("Time (ms)", f"{result['time']:.2f}")
            st.metric("Accuracy", f"{result['accuracy_score']}%")
            
            # Show corrected text preview
            with st.expander("View Output"):
                st.text_area("", value=result['corrected'], height=100, key=f"output_{idx}")
    
    # Detailed comparison table
    st.markdown("### 📝 Detailed Comparison")
    
    comparison_data = []
    for algo_name, result in results.items():
        comparison_data.append({
            'Algorithm': algo_name,
            'Corrected Text': result['corrected'],
            'Corrections Made': len(result['corrections']),
            'Processing Time (ms)': f"{result['time']:.2f}",
            'Known Accuracy (%)': result['accuracy_score']
        })
    
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Find the "winner" (most corrections with highest accuracy)
    if any(len(r['corrections']) > 0 for r in results.values()):
        best_algo = max(results.items(), 
                       key=lambda x: (len(x[1]['corrections']), x[1]['accuracy_score']))
        st.success(f"🏆 **Best Performance**: {best_algo[0]} (Made {len(best_algo[1]['corrections'])} corrections with {best_algo[1]['accuracy_score']}% accuracy)")

def show_performance_benchmarks():
    """Show performance benchmarks from evaluation"""
    st.markdown("## 📈 Performance Benchmarks")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Accuracy Comparison")
        
        # Accuracy data from evaluation
        accuracy_data = {
            'Algorithm': ['PySpellChecker', 'AutoCorrect', 'Frequency-Based', 'Levenshtein'],
            'Accuracy (%)': [92.6, 91.2, 75.3, 64.2]
        }
        
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(accuracy_data['Algorithm'], accuracy_data['Accuracy (%)'], 
                      color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c'])
        ax.set_xlabel('Accuracy (%)', fontweight='bold')
        ax.set_title('Algorithm Accuracy on Test Dataset (433 samples)', fontweight='bold')
        ax.set_xlim(0, 100)
        
        # Add value labels
        for i, (algo, acc) in enumerate(zip(accuracy_data['Algorithm'], accuracy_data['Accuracy (%)'])):
            ax.text(acc + 1, i, f'{acc}%', va='center', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown("### Error Type Performance")
        
        # Error type data
        error_types = ['Keyboard Errors', 'Phonetic Errors', 'Character Errors', 'Simple Typos']
        pyspell_scores = [93.5, 91.8, 92.4, 92.7]
        autocorrect_scores = [91.0, 90.5, 91.8, 91.5]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        x = range(len(error_types))
        width = 0.35
        
        ax.bar([i - width/2 for i in x], pyspell_scores, width, 
               label='PySpellChecker', color='#2ecc71')
        ax.bar([i + width/2 for i in x], autocorrect_scores, width, 
               label='AutoCorrect', color='#3498db')
        
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.set_title('Performance by Error Type', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(error_types, rotation=15, ha='right')
        ax.legend()
        ax.set_ylim(85, 95)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Dataset information
    st.markdown("### 📊 Test Dataset Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", "433")
    
    with col2:
        st.metric("Error Types", "4")
    
    with col3:
        st.metric("Best Algorithm", "PySpellChecker")
    
    with col4:
        st.metric("Best Accuracy", "92.6%")
    
    with st.expander("📖 View Dataset Composition"):
        dataset_info = pd.DataFrame({
            'Error Type': ['Keyboard Errors', 'Phonetic Errors', 'Character Errors', 'Simple Typos'],
            'Examples': [
                'teh → the, fpr → for',
                'nite → night, fone → phone',
                'recieve → receive, definately → definitely',
                'tiem → time, freind → friend'
            ],
            'Sample Count': ['~108', '~108', '~108', '~109']
        })
        st.dataframe(dataset_info, use_container_width=True, hide_index=True)

def show_algorithm_details():
    """Show detailed information about each algorithm"""
    st.markdown("## 🔬 Algorithm Details")
    
    algorithms_info = {
        'PySpellChecker': {
            'type': 'Statistical Dictionary-Based',
            'accuracy': '92.6%',
            'speed': 'Fast',
            'description': 'Uses word frequency data from large text corpus. Finds candidates by edit distance and selects most frequent word.',
            'strengths': ['High accuracy', 'Fast performance', 'Good with common words'],
            'weaknesses': ['Limited context awareness', 'May struggle with rare words'],
            'use_cases': ['General text correction', 'Real-time correction', 'Common typos']
        },
        'AutoCorrect': {
            'type': 'Pattern-Based ML',
            'accuracy': '91.2%',
            'speed': 'Fast',
            'description': 'Machine learning model trained on common misspelling patterns. Uses statistical models for prediction.',
            'strengths': ['Pattern recognition', 'Good with phonetic errors', 'Context-aware'],
            'weaknesses': ['Slightly lower accuracy', 'May overfit to training data'],
            'use_cases': ['Mobile keyboards', 'Predictive text', 'Phonetic errors']
        },
        'Frequency-Based': {
            'type': 'Statistical Frequency Analysis',
            'accuracy': '75.3%',
            'speed': 'Medium',
            'description': 'Combines word frequency from corpus with edit distance. Prefers more common words as corrections.',
            'strengths': ['Simple approach', 'Explainable results', 'No training needed'],
            'weaknesses': ['Lower accuracy', 'Sensitive to corpus quality', 'May miss context'],
            'use_cases': ['Research', 'Educational purposes', 'Baseline comparison']
        },
        'Levenshtein': {
            'type': 'Edit Distance Algorithm',
            'accuracy': '64.2%',
            'speed': 'Slow',
            'description': 'Pure edit distance (character operations: insert, delete, substitute). Finds closest dictionary word.',
            'strengths': ['Simple concept', 'No training required', 'Language agnostic'],
            'weaknesses': ['Lowest accuracy', 'Slow on large vocabularies', 'No context'],
            'use_cases': ['String matching', 'Fuzzy search', 'Academic study']
        }
    }
    
    for algo_name, info in algorithms_info.items():
        with st.expander(f"🔍 {algo_name} - {info['type']}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Accuracy", info['accuracy'])
            with col2:
                st.metric("Speed", info['speed'])
            with col3:
                st.metric("Type", info['type'])
            
            st.markdown(f"**Description:** {info['description']}")
            
            col_str, col_weak = st.columns(2)
            
            with col_str:
                st.markdown("**✅ Strengths:**")
                for strength in info['strengths']:
                    st.markdown(f"• {strength}")
            
            with col_weak:
                st.markdown("**⚠️ Weaknesses:**")
                for weakness in info['weaknesses']:
                    st.markdown(f"• {weakness}")
            
            st.markdown("**💡 Best Use Cases:**")
            st.markdown(", ".join(info['use_cases']))

def main():
    # Header
    st.markdown('<div class="main-title">🔤 Spelling Correction Comparison Tool</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Compare 4 different spelling correction algorithms side-by-side</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page:",
        ["🏠 Live Comparison", "📈 Performance Benchmarks", "🔬 Algorithm Details", "📚 About"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Stats")
    st.sidebar.metric("Algorithms Compared", "4")
    st.sidebar.metric("Best Accuracy", "92.6%")
    st.sidebar.metric("Test Dataset Size", "433 samples")
    
    # Initialize corrector
    if 'corrector' not in st.session_state:
        with st.spinner("Initializing spelling corrector..."):
            st.session_state.corrector = initialize_corrector()
    
    # Page routing
    if page == "🏠 Live Comparison":
        show_live_comparison(st.session_state.corrector)
    elif page == "📈 Performance Benchmarks":
        show_performance_benchmarks()
    elif page == "🔬 Algorithm Details":
        show_algorithm_details()
    elif page == "📚 About":
        show_about()

def show_live_comparison(corrector):
    """Main comparison interface"""
    st.markdown("## 🧪 Live Algorithm Comparison")
    
    # Input methods
    input_method = st.radio(
        "Choose input method:",
        ["✍️ Type Text", "📄 Use Example", "📁 Load Test Cases"],
        horizontal=True
    )
    
    user_text = ""
    
    if input_method == "✍️ Type Text":
        user_text = st.text_area(
            "Enter text with spelling errors:",
            height=150,
            placeholder="The qick brown fox jumps over the lazy dog..."
        )
    
    elif input_method == "📄 Use Example":
        examples = {
            "Simple Typos": "The qick brown fox jumps over the lazy dog",
            "Multiple Errors": "I recieved your mesage yestarday and it was grate",
            "Technical Text": "Artifical inteligence is revolutionizing tecnology and machne lerning",
            "Common Mistakes": "Please chck your speling before submiting the documnt to recieve feedback"
        }
        
        selected = st.selectbox("Select an example:", list(examples.keys()))
        user_text = examples[selected]
        st.info(f"**Selected:** {user_text}")
    
    elif input_method == "📁 Load Test Cases":
        # Load from test dataset if available
        if os.path.exists("spelling_test_dataset_small.json"):
            with open("spelling_test_dataset_small.json", 'r') as f:
                test_data = json.load(f)
            
            test_idx = st.slider("Select test case:", 0, len(test_data)-1, 0)
            test_case = test_data[test_idx]
            user_text = test_case['misspelled']
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Misspelled:** {test_case['misspelled']}")
            with col2:
                st.success(f"**Correct:** {test_case['correct']}")
            
            st.caption(f"Error Type: {test_case.get('error_type', 'Unknown')}")
        else:
            st.warning("Test dataset not found. Please use other input methods.")
            user_text = ""
    
    # Run comparison button
    if user_text and st.button("🚀 Compare All Algorithms", type="primary"):
        with st.spinner("Running all algorithms..."):
            results = run_all_algorithms(user_text, corrector)
        
        # Display original text
        st.markdown("### 📝 Original Text")
        st.info(user_text)
        
        # Display results
        display_comparison_results(results, user_text)
        
        # Individual corrections details
        st.markdown("### 🔍 Individual Corrections by Algorithm")
        
        for algo_name, result in results.items():
            with st.expander(f"{algo_name} - {len(result['corrections'])} corrections"):
                if result['corrections']:
                    corrections_df = pd.DataFrame(result['corrections'])
                    st.dataframe(corrections_df, use_container_width=True, hide_index=True)
                    
                    st.markdown("**Full Corrected Text:**")
                    st.success(result['corrected'])
                else:
                    st.info("No corrections needed or no errors found.")

def show_about():
    """About page with project information"""
    st.markdown("## 📚 About This Project")
    
    st.markdown("""
    ### Overview
    This is a comprehensive spelling correction comparison tool that evaluates 4 different NLP algorithms:
    
    1. **PySpellChecker** - Statistical dictionary-based approach
    2. **AutoCorrect** - Pattern-based machine learning
    3. **Frequency-Based** - Corpus frequency with edit distance
    4. **Levenshtein Distance** - Pure edit distance algorithm
    
    ### Evaluation Methodology
    - **Dataset**: 433 samples with 4 error types (keyboard, phonetic, character, simple typos)
    - **Metrics**: Accuracy, processing speed, error-type performance
    - **Approach**: Individual algorithm comparison (not ensemble)
    
    ### Key Findings
    - **PySpellChecker** achieved the highest accuracy (92.6%)
    - **AutoCorrect** came close second (91.2%)
    - **Frequency-Based** and **Levenshtein** showed lower accuracy but remain useful for specific use cases
    
    ### Technology Stack
    - Python 3.x
    - Streamlit for web interface
    - PySpellChecker, AutoCorrect, TextDistance libraries
    - NLTK for corpus data
    - Matplotlib/Seaborn for visualizations
    
    ### Project Structure
    ```
    📁 Project Files
    ├── streamlit_app_new.py (This app)
    ├── generate_test_dataset.py (Dataset generation)
    ├── evaluate_spelling_methods.py (Evaluation script)
    ├── spelling_test_dataset_small.json (Test data)
    └── Results & Reports
    ```
    
    ### Future Enhancements
    - [ ] Add more algorithms (SymSpell, BK-Tree)
    - [ ] Context-aware correction using transformers
    - [ ] Real-time performance monitoring
    - [ ] Export comparison results
    - [ ] Batch file processing
    
    ---
    
    **Built with ❤️ for NLP Research and Education**
    """)
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Algorithms", "4")
    with col2:
        st.metric("Test Samples", "433")
    with col3:
        st.metric("Error Types", "4")

if __name__ == "__main__":
    main()
