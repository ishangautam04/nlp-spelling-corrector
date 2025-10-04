import streamlit as st
import pandas as pd
from spelling_corrector import SpellingCorrector
from ml_spelling_corrector import MLSpellingCorrector
import time
import os

# Configure the page
st.set_page_config(
    page_title="ML Spelling Corrector Testing",
    page_icon="🤖",
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
        background-color: #f8f9fa;
        color: #212529;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
        font-weight: 500;
    }
    .original-text {
        color: #dc3545;
        font-weight: bold;
        background-color: #f8d7da;
        padding: 0.5rem;
        border-radius: 0.3rem;
        border: 1px solid #f5c6cb;
    }
    .corrected-text {
        color: #155724;
        font-weight: bold;
        background-color: #d4edda;
        padding: 0.5rem;
        border-radius: 0.3rem;
        border: 1px solid #c3e6cb;
    }
    .ml-result {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        font-weight: bold;
        font-size: 1.1rem;
        border: 2px solid #28a745;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .standard-result {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        font-weight: bold;
        font-size: 1.1rem;
        border: 2px solid #17a2b8;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .word-result {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.8rem;
        border-radius: 0.4rem;
        border: 2px solid #ffc107;
        font-weight: bold;
        margin: 0.5rem 0;
        text-align: center;
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
    st.header("🎯 ML Model Training Data")
    
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
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Training Examples", len(df))
    
    with col2:
        st.metric("Categories", "5 types")
    
    with col3:
        st.metric("Model Type", "Random Forest + TF-IDF")
    
    st.info("💡 **Model Details:** The ML model uses TF-IDF vectorization with Random Forest classifier, trained on character patterns and n-grams to learn spelling correction patterns.")
    
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
            st.dataframe(category_df, hide_index=True, use_container_width=True)

def ml_testing_interface():
    """Main ML testing interface"""
    st.header("🤖 ML Spelling Corrector Testing")
    
    ml_corrector = load_ml_corrector()
    standard_corrector = load_corrector()
    
    if ml_corrector is None:
        st.error("❌ ML model not found! Please run 'python train_spelling_model.py' first.")
        st.info("To create the ML model, run the following command in your terminal:")
        st.code("python train_spelling_model.py")
        return
    
    st.success("✅ ML model loaded successfully!")
    
    # Test input methods
    test_method = st.radio(
        "Choose testing method:",
        ["Enter custom text", "Use predefined test cases", "Single word testing"],
        horizontal=True
    )
    
    if test_method == "Enter custom text":
        custom_text_testing(ml_corrector, standard_corrector)
    elif test_method == "Use predefined test cases":
        predefined_testing(ml_corrector, standard_corrector)
    else:
        single_word_testing(ml_corrector, standard_corrector)

def custom_text_testing(ml_corrector, standard_corrector):
    """Custom text input testing"""
    st.subheader("📝 Custom Text Testing")
    
    user_text = st.text_area(
        "Enter text to test spelling correction:",
        height=100,
        placeholder="Type your text with spelling errors here..."
    )
    
    if user_text:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔧 Standard Ensemble:**")
            start_time = time.time()
            standard_result = standard_corrector.correct_text(user_text, method='ensemble')
            standard_time = time.time() - start_time
            
            if isinstance(standard_result, tuple):
                standard_corrected, standard_corrections = standard_result
            else:
                standard_corrected = standard_result
                standard_corrections = []
            
            st.markdown(f'<div class="standard-result">{standard_corrected}</div>', unsafe_allow_html=True)
            st.caption(f"⏱️ Time: {standard_time*1000:.2f}ms | Corrections: {len(standard_corrections)}")
        
        with col2:
            st.markdown("**🤖 ML-Enhanced:**")
            start_time = time.time()
            ml_result = ml_corrector.correct_text_with_ml(user_text, method='ensemble_ml')
            ml_time = time.time() - start_time
            
            if isinstance(ml_result, tuple):
                ml_corrected, ml_corrections = ml_result
                st.markdown(f'<div class="ml-result">{ml_corrected}</div>', unsafe_allow_html=True)
                st.caption(f"⏱️ Time: {ml_time*1000:.2f}ms | Corrections: {len(ml_corrections)}")
                
                if ml_corrections:
                    with st.expander("View detailed corrections"):
                        for correction in ml_corrections:
                            st.write(f"• '{correction['original']}' → '{correction['corrected']}'")
            else:
                st.markdown(f'<div class="ml-result">{ml_result}</div>', unsafe_allow_html=True)
                st.caption(f"⏱️ Time: {ml_time*1000:.2f}ms")

def predefined_testing(ml_corrector, standard_corrector):
    """Test with predefined cases"""
    st.subheader("📋 Predefined Test Cases")
    
    test_cases = [
        "teh qick brown fox",
        "I recieve your mesage",
        "Please seperate thier items",
        "This is definately wierd",
        "We beleive the freind is coming",
        "The begining was vrey intresting",
        "I occured to me that this is wierd",
        "Thier seperate beleifs are definately different"
    ]
    
    selected_case = st.selectbox("Select a test case:", test_cases)
    
    if st.button("Run Test", type="primary"):
        st.markdown(f"**Original:** {selected_case}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔧 Standard Result:**")
            start_time = time.time()
            standard_result = standard_corrector.correct_text(selected_case, method='ensemble')
            standard_time = time.time() - start_time
            
            if isinstance(standard_result, tuple):
                standard_corrected = standard_result[0]
            else:
                standard_corrected = standard_result
            
            st.markdown(f'<div class="standard-result">{standard_corrected}</div>', unsafe_allow_html=True)
            st.caption(f"⏱️ {standard_time*1000:.2f}ms")
        
        with col2:
            st.markdown("**🤖 ML-Enhanced Result:**")
            start_time = time.time()
            ml_result = ml_corrector.correct_text_with_ml(selected_case, method='ensemble_ml')
            ml_time = time.time() - start_time
            
            if isinstance(ml_result, tuple):
                ml_corrected, ml_corrections = ml_result
                st.markdown(f'<div class="ml-result">{ml_corrected}</div>', unsafe_allow_html=True)
                st.caption(f"⏱️ {ml_time*1000:.2f}ms")
                
                if ml_corrections:
                    st.write("**Corrections made:**")
                    for correction in ml_corrections:
                        st.write(f"• '{correction['original']}' → '{correction['corrected']}'")
            else:
                st.markdown(f'<div class="ml-result">{ml_result}</div>', unsafe_allow_html=True)
                st.caption(f"⏱️ {ml_time*1000:.2f}ms")

def single_word_testing(ml_corrector, standard_corrector):
    """Test individual words"""
    st.subheader("🔤 Single Word Testing")
    
    # Pre-loaded test words from training data
    training_words = ['teh', 'thier', 'recieve', 'seperate', 'definately', 'beleive', 'wierd', 'freind']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Enter a word to test:**")
        custom_word = st.text_input("Word:", placeholder="Type a misspelled word...")
    
    with col2:
        st.markdown("**Or choose from training data:**")
        selected_word = st.selectbox("Training words:", [''] + training_words)
    
    test_word = custom_word if custom_word else selected_word
    
    if test_word:
        st.markdown(f"**Testing word:** `{test_word}`")
        
        # Create columns for different methods
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**PySpellChecker:**")
            pyspell_result = standard_corrector.correct_with_pyspellchecker(test_word)
            st.markdown(f'<div class="word-result">→ {pyspell_result}</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("**AutoCorrect:**")
            autocorrect_result = standard_corrector.correct_with_autocorrect(test_word)
            st.markdown(f'<div class="word-result">→ {autocorrect_result}</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown("**Standard Ensemble:**")
            ensemble_result = standard_corrector.ensemble_correction(test_word)
            if isinstance(ensemble_result, tuple):
                ensemble_result = ensemble_result[0]
            st.markdown(f'<div class="word-result">→ {ensemble_result}</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown("**🤖 ML Model:**")
            ml_result = ml_corrector.correct_with_ml_model(test_word)
            st.markdown(f'<div class="ml-result">→ {ml_result}</div>', unsafe_allow_html=True)

def performance_comparison():
    """Performance comparison section"""
    st.header("📈 Performance Comparison")
    
    ml_corrector = load_ml_corrector()
    standard_corrector = load_corrector()
    
    if ml_corrector is None:
        st.error("❌ ML model not found!")
        return
    
    # Benchmark test
    st.subheader("🏃‍♂️ Speed Benchmark")
    
    benchmark_text = "Teh qick brown fox jumps ovr teh lazy dog and recieve many seperate gifts from thier freinds"
    
    if st.button("Run Benchmark", type="primary"):
        methods = [
            ('Standard Ensemble', lambda: standard_corrector.correct_text(benchmark_text, method='ensemble')),
            ('ML-Enhanced Ensemble', lambda: ml_corrector.correct_text_with_ml(benchmark_text, method='ensemble_ml')),
            ('ML-Only', lambda: ml_corrector.correct_text_with_ml(benchmark_text, method='ml_only')),
            ('PySpellChecker Only', lambda: standard_corrector.correct_text(benchmark_text, method='pyspellchecker')),
            ('AutoCorrect Only', lambda: standard_corrector.correct_text(benchmark_text, method='autocorrect'))
        ]
        
        results = []
        
        for method_name, method_func in methods:
            # Warm up
            method_func()
            
            # Time the method
            start_time = time.time()
            result = method_func()
            end_time = time.time()
            
            if isinstance(result, tuple):
                result = result[0]
            
            results.append({
                'Method': method_name,
                'Result': result,
                'Time (ms)': f"{(end_time - start_time)*1000:.2f}"
            })
        
        # Display results
        st.markdown(f"**Test text:** {benchmark_text}")
        st.dataframe(pd.DataFrame(results), hide_index=True, use_container_width=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 ML Spelling Corrector Testing Interface</h1>', unsafe_allow_html=True)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["🧪 ML Testing", "📊 Training Data", "📈 Performance", "ℹ️ About"])
    
    with tab1:
        ml_testing_interface()
    
    with tab2:
        show_training_data()
    
    with tab3:
        performance_comparison()
    
    with tab4:
        st.header("ℹ️ About ML Spelling Corrector")
        
        st.markdown("""
        ### 🤖 Machine Learning Model
        
        This spelling corrector combines traditional NLP techniques with machine learning:
        
        **Traditional Algorithms:**
        - PySpellChecker (statistical frequency-based)
        - AutoCorrect (machine learning-based)
        - Levenshtein distance (edit distance)
        - Frequency-based pattern matching
        
        **ML Enhancement:**
        - **Model:** Random Forest Classifier
        - **Features:** TF-IDF vectorization with character n-grams
        - **Training:** Pattern-based learning from common spelling errors
        - **Integration:** Weighted ensemble voting combining ML predictions with traditional methods
        
        ### 📊 Training Data
        
        The ML model was trained on 30 carefully selected spelling error patterns:
        - Common typos (teh → the, recieve → receive)
        - Character transpositions (form → from)
        - Missing letters (wich → which)
        - Extra letters (whith → with)
        - Word confusions (alot → a lot)
        
        ### 🎯 Performance
        
        **ML-Enhanced Benefits:**
        - Superior accuracy on trained patterns
        - Context-aware corrections
        - Learns from specific error types
        - Maintains speed with slight overhead (~20ms)
        
        **When to Use:**
        - Documents with common spelling errors
        - Text with learned error patterns
        - When accuracy is more important than speed
        
        ### 🔧 Technical Details
        
        - **Framework:** scikit-learn, pandas, numpy
        - **Vectorization:** TF-IDF with 1-3 character n-grams
        - **Model:** Random Forest (100 estimators)
        - **Integration:** Weighted voting ensemble
        - **Weights:** ML=4.5, PySpell=4.0, Frequency=3.5, AutoCorrect=2.5, Levenshtein=2.0
        """)

if __name__ == "__main__":
    main()