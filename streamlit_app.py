import streamlit as st
import pandas as pd
from spelling_corrector import SpellingCorrector
from ml_spelling_corrector import MLSpellingCorrector
import time
import os

# Configure the page
st.set_page_config(
    page_title="NLP Spelling Corrector",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .method-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .correction-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .original-text {
        color: #d62728;
        font-weight: bold;
    }
    .corrected-text {
        color: #2ca02c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_corrector():
    """Load the spelling corrector (cached for performance)"""
    return SpellingCorrector()

@st.cache_resource
def load_ml_corrector():
    """Load the ML-enhanced spelling corrector (cached for performance)"""
    if os.path.exists("spelling_correction_model.pkl"):
        return MLSpellingCorrector("spelling_correction_model.pkl")
    return None

def show_training_data():
    """Display the training data used for the ML model"""
    st.subheader("🎯 ML Model Training Data")
    
    # The exact training data from the model
    training_data = [
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
    
    df = pd.DataFrame(training_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Training Examples", len(df))
    
    with col2:
        st.metric("Categories", "5 types: Common typos, Transpositions, Missing letters, Extra letters, Word confusions")
    
    # Show the data in categories
    categories = {
        "Common Typos": training_data[0:10],
        "Transpositions": training_data[10:15],
        "Missing Letters": training_data[15:20],
        "Extra Letters": training_data[20:25],
        "Word Confusions": training_data[25:30]
    }
    
    for category, data in categories.items():
        with st.expander(f"📂 {category} ({len(data)} examples)"):
            category_df = pd.DataFrame(data)
            st.dataframe(category_df, hide_index=True)
    
    st.info("💡 **Model Details:** The ML model uses TF-IDF vectorization with Random Forest classifier, trained on character patterns and n-grams to learn spelling correction patterns.")

def ml_comparison_section():
    """Show ML vs Standard algorithm comparison"""
    st.subheader("🤖 ML vs Standard Algorithm Comparison")
    
    ml_corrector = load_ml_corrector()
    standard_corrector = load_corrector()
    
    if ml_corrector is None:
        st.error("❌ ML model not found! Please run 'python train_spelling_model.py' first.")
        return
    
    # Test cases specifically designed to show ML advantages
    test_cases = [
        "teh qick brown fox",
        "I recieve your mesage",
        "Please seperate thier items",
        "This is definately wierd",
        "We beleive the freind is coming",
        "The begining was vrey intresting"
    ]
    
    st.write("**Pre-loaded test cases showing ML improvements:**")
    
    for i, test_text in enumerate(test_cases):
        with st.expander(f"Test {i+1}: '{test_text}'"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🔧 Standard Ensemble:**")
                standard_result = standard_corrector.correct_text(test_text, method='ensemble')
                if isinstance(standard_result, tuple):
                    standard_result = standard_result[0]
                st.write(f"➡️ {standard_result}")
            
            with col2:
                st.write("**🤖 ML-Enhanced:**")
                ml_result = ml_corrector.correct_text_with_ml(test_text, method='ensemble_ml')
                if isinstance(ml_result, tuple):
                    corrected, corrections = ml_result
                    st.write(f"➡️ {corrected}")
                    if corrections:
                        st.write("**Corrections made:**")
                        for correction in corrections:
                            st.write(f"• '{correction['original']}' → '{correction['corrected']}'")
                else:
                    st.write(f"➡️ {ml_result}")
    
    # Custom testing
    st.write("**🧪 Test your own text:**")
    custom_text = st.text_input("Enter text to compare ML vs Standard correction:")
    
    if custom_text:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🔧 Standard Result:**")
            start_time = time.time()
            standard_result = standard_corrector.correct_text(custom_text, method='ensemble')
            standard_time = time.time() - start_time
            if isinstance(standard_result, tuple):
                standard_result = standard_result[0]
            st.success(standard_result)
            st.caption(f"⏱️ Time: {standard_time*1000:.2f}ms")
        
        with col2:
            st.write("**🤖 ML-Enhanced Result:**")
            start_time = time.time()
            ml_result = ml_corrector.correct_text_with_ml(custom_text, method='ensemble_ml')
            ml_time = time.time() - start_time
            if isinstance(ml_result, tuple):
                corrected, corrections = ml_result
                st.success(corrected)
                if corrections:
                    st.write("**Corrections:**")
                    for correction in corrections:
                        st.write(f"• '{correction['original']}' → '{correction['corrected']}'")
            else:
                st.success(ml_result)
            st.caption(f"⏱️ Time: {ml_time*1000:.2f}ms")

def main():
    # Header
    st.markdown('<h1 class="main-header">📝 NLP Spelling Corrector</h1>', unsafe_allow_html=True)
    
    # Sidebar for method selection and information
    st.sidebar.title("Configuration")
    st.sidebar.markdown("---")
    
    # Method selection
    correction_method = st.sidebar.selectbox(
        "Select Correction Method:",
        ['ensemble', 'pyspellchecker', 'autocorrect', 'levenshtein', 'frequency'],
        index=0,
        help="Choose the NLP method for spelling correction"
    )
    
    st.sidebar.markdown("---")
    
    # Method descriptions
    st.sidebar.markdown("### Method Descriptions")
    method_descriptions = {
        'ensemble': 'Combines multiple methods with weighted voting (PySpell, Frequency, AutoCorrect, Levenshtein)',
        'pyspellchecker': 'Statistical spell checker based on word frequency',
        'autocorrect': 'Machine learning-based autocorrection',
        'levenshtein': 'Edit distance-based correction',
        'frequency': 'Frequency-based with common misspelling patterns'
    }
    
    for method, description in method_descriptions.items():
        if method == correction_method:
            st.sidebar.markdown(f"**{method.capitalize()}**: {description}")
        else:
            st.sidebar.markdown(f"{method.capitalize()}: {description}")
    
    # Load the corrector
    if 'corrector' not in st.session_state:
        with st.spinner('Initializing spelling corrector...'):
            st.session_state.corrector = load_corrector()
        st.success('Spelling corrector loaded successfully!')
    
    corrector = st.session_state.corrector
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="method-header">Input Text</h2>', unsafe_allow_html=True)
        
        # Text input options
        input_method = st.radio(
            "Choose input method:",
            ["Type text", "Upload file", "Use example"],
            horizontal=True
        )
        
        user_text = ""
        
        if input_method == "Type text":
            user_text = st.text_area(
                "Enter text to correct:",
                height=200,
                placeholder="Type your text here..."
            )
        
        elif input_method == "Upload file":
            uploaded_file = st.file_uploader(
                "Choose a text file",
                type=['txt'],
                help="Upload a .txt file to correct spelling"
            )
            
            if uploaded_file is not None:
                user_text = str(uploaded_file.read(), "utf-8")
                st.text_area("File content:", value=user_text, height=200, disabled=True)
        
        elif input_method == "Use example":
            examples = [
                "The qick brown fox jumps over the lazy dog",
                "I recieved your mesage yestarday and it was grate",
                "Artifical inteligence is revolutionizing tecnology",
                "Please chck your speling before submiting the documnt"
            ]
            
            selected_example = st.selectbox("Choose an example:", examples)
            user_text = selected_example
            st.text_area("Selected example:", value=user_text, height=100, disabled=True)
    
    with col2:
        st.markdown('<h2 class="method-header">Corrected Text</h2>', unsafe_allow_html=True)
        
        if user_text.strip():
            # Correct the text
            with st.spinner(f'Correcting text using {correction_method} method...'):
                start_time = time.time()
                
                # Handle different return types from correction methods
                if correction_method in ['ensemble', 'context']:
                    result = corrector.correct_text(user_text, method=correction_method)
                    if isinstance(result, tuple) and len(result) == 2:
                        corrected_text, corrections = result
                    else:
                        corrected_text = result
                        corrections = []
                else:
                    corrected_text = corrector.correct_text(user_text, method=correction_method)
                    corrections = []
                
                end_time = time.time()
            
            # Display corrected text
            st.text_area(
                "Corrected text:",
                value=corrected_text,
                height=200,
                disabled=True
            )
            
            # Performance metrics
            processing_time = end_time - start_time
            col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
            
            with col_metrics1:
                st.metric("Processing Time", f"{processing_time:.3f}s")
            
            with col_metrics2:
                st.metric("Corrections Made", len(corrections))
            
            with col_metrics3:
                original_words = len(user_text.split())
                st.metric("Total Words", original_words)
            
        else:
            st.info("Enter some text to see the correction results.")
    
    # Detailed corrections section
    if user_text.strip():
        st.markdown("---")
        st.markdown('<h2 class="method-header">Detailed Analysis</h2>', unsafe_allow_html=True)
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["Corrections Made", "Method Comparison", "Word Suggestions"])
        
        with tab1:
            if corrections:
                st.markdown("### Corrections Made")
                
                corrections_df = pd.DataFrame([
                    {
                        'Original': correction['original'],
                        'Corrected': correction['corrected'],
                        'Methods (if ensemble)': str(correction.get('all_methods', 'N/A'))
                    }
                    for correction in corrections
                ])
                
                st.dataframe(corrections_df, use_container_width=True)
                
                # Highlight differences
                st.markdown("### Text Comparison")
                col_comp1, col_comp2 = st.columns(2)
                
                with col_comp1:
                    st.markdown("**Original Text:**")
                    st.markdown(f'<div class="correction-box original-text">{user_text}</div>', unsafe_allow_html=True)
                
                with col_comp2:
                    st.markdown("**Corrected Text:**")
                    st.markdown(f'<div class="correction-box corrected-text">{corrected_text}</div>', unsafe_allow_html=True)
            
            else:
                st.success("No spelling errors found! ✅")
        
        with tab2:
            st.markdown("### Method Comparison")
            st.info("Compare corrections from different methods")
            
            methods = ['ensemble', 'pyspellchecker', 'autocorrect', 'levenshtein']
            comparison_data = []
            
            with st.spinner('Comparing methods...'):
                for method in methods:
                    method_corrected, method_corrections = corrector.correct_text(user_text, method=method)
                    comparison_data.append({
                        'Method': method.capitalize(),
                        'Corrected Text': method_corrected,
                        'Corrections Count': len(method_corrections)
                    })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
        
        with tab3:
            st.markdown("### Word Suggestions")
            
            # Get individual word suggestions
            words = corrector.tokenize_text(corrector.preprocess_text(user_text))
            suggestion_data = []
            
            for word in set(words):  # Use set to avoid duplicates
                if word not in corrector.pyspell_checker and len(word) > 2:
                    suggestions = corrector.get_word_suggestions(word, n=5)
                    if suggestions:
                        suggestion_data.append({
                            'Original Word': word,
                            'Suggestions': ', '.join(suggestions[:3])  # Show top 3
                        })
            
            if suggestion_data:
                suggestions_df = pd.DataFrame(suggestion_data)
                st.dataframe(suggestions_df, use_container_width=True)
            else:
                st.info("No alternative suggestions available for the words in your text.")
    
    # Footer with information
    st.markdown("---")
    st.markdown("### About this Application")
    st.markdown("""
    This spelling corrector uses multiple NLP techniques:
    - **Statistical approaches** using word frequency data (PySpellChecker)
    - **Edit distance algorithms** (Levenshtein)
    - **Machine learning models** for context-aware correction (AutoCorrect)
    - **Ensemble methods** combining multiple approaches
    
    Built with Python, NLTK, and various NLP libraries.
    """)

if __name__ == "__main__":
    main()
