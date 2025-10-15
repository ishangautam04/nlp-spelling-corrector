# 🔤 NLP Spelling Corrector

A comprehensive spelling correction system comparing 4 different algorithms with interactive web interface and detailed performance analysis.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🌟 Features

- **4 Spelling Correction Algorithms**
  - PySpellChecker (91.76% accuracy)
  - AutoCorrect (88.13% accuracy)
  - Frequency-Based (69.34% accuracy)
  - Levenshtein Distance (60.76% accuracy)

- **Interactive Web UI** (Streamlit)
  - Live correction comparison
  - Performance benchmarks
  - Algorithm details
  - Multiple input methods

- **Comprehensive Evaluation Framework**
  - 2,671 test samples
  - 4 error types (keyboard, phonetic, character, simple typo)
  - Detailed performance metrics
  - Visualization generation

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ishangautam04/nlp-spelling-corrector.git
cd nlp-spelling-corrector

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Web App

**Windows:**
```bash
start_web_app_new.bat
```

**Or manually:**
```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

## 📊 Performance Results

Tested on 2,671 samples across 4 error types:

| Algorithm | Accuracy | Speed (ms/word) | Best For |
|-----------|----------|-----------------|----------|
| **PySpellChecker** | **91.76%** | 32.24 | Keyboard errors (97.2%) |
| **AutoCorrect** | 88.13% | **4.28** ⚡ | Simple typos (93.2%) |
| **Frequency-Based** | 69.34% | 34.06 | General corrections |
| **Levenshtein** | 60.76% | 37.54 | Edit distance |

### Performance by Error Type

- **Keyboard Errors**: PySpellChecker (97.2%)
- **Simple Typos**: PySpellChecker (96.9%)
- **Character Errors**: PySpellChecker (92.0%)
- **Phonetic Errors**: PySpellChecker (73.7%)

## 📁 Project Structure

```
nlp-spelling-corrector/
├── streamlit_app.py              # Main web interface
├── spelling_corrector.py         # Core correction algorithms
├── evaluate_spelling_methods.py  # Comprehensive evaluation
├── simple_test_evaluation.py     # Quick evaluation script
├── generate_test_dataset.py      # Synthetic dataset generator
├── generate_flowchart.py         # Flowchart generation
├── requirements.txt              # Python dependencies
├── evaluation_results.json       # Latest test results
├── spelling_test_dataset_*.json  # Test datasets (small/medium/large)
└── *.bat                         # Windows batch files
```

## 🔧 Usage

### Web Interface

```bash
streamlit run streamlit_app.py
```

**4 Pages:**
1. **Live Comparison** - Test corrections in real-time
2. **Performance Benchmarks** - View accuracy metrics
3. **Algorithm Details** - Learn how each works
4. **About** - Project information

### Command Line

```python
from spelling_corrector import SpellingCorrector

corrector = SpellingCorrector()

# Correct text with different algorithms
text = "The qick brown fox jumps"

corrected1, _ = corrector.correct_text(text, method='pyspellchecker')
corrected2, _ = corrector.correct_text(text, method='autocorrect')
corrected3, _ = corrector.correct_text(text, method='frequency')
corrected4, _ = corrector.correct_text(text, method='levenshtein')

print(corrected1)  # "The quick brown fox jumps"
```

### Run Evaluation

```bash
# Quick evaluation
python simple_test_evaluation.py

# Comprehensive evaluation
python evaluate_spelling_methods.py
```

## 📈 Visualizations

The project generates:
- `spelling_correction_flowchart.pdf` - Process flow diagram
- `simple_evaluation_accuracy.png` - Accuracy comparison
- `simple_evaluation_by_error_type.png` - Per-error-type performance
- `spelling_correction_performance.png` - Detailed benchmarks

## 🧪 Testing

Generate synthetic test datasets:
```bash
python generate_test_dataset.py
```

Creates 3 datasets:
- Small: 500 samples
- Medium: 1,500 samples
- Large: 3,000 samples (2,671 after filtering)

## 📚 Algorithms

### 1. PySpellChecker
- Dictionary-based with edit distance
- Best overall accuracy (91.76%)
- Pre-trained on English dictionary

### 2. AutoCorrect
- Statistical pattern matching
- Fastest algorithm (4.28 ms/word)
- Good accuracy-speed balance

### 3. Frequency-Based
- Uses NLTK Brown corpus
- Frequency-weighted candidate selection
- Common misspelling patterns

### 4. Levenshtein Distance
- Pure edit distance calculation
- No linguistic knowledge
- Baseline comparison

## 🔬 Research

This project evaluates word-level spelling correction algorithms without context awareness. Suitable for:
- Academic research papers
- Algorithm comparison studies
- NLP coursework
- Baseline benchmarking

**Limitations:**
- No context awareness (homophones not disambiguated)
- Word-by-word processing
- No grammar correction

## 🛠️ Technologies

- **Python 3.8+**
- **Streamlit** - Web interface
- **pyspellchecker** - Dictionary-based correction
- **autocorrect** - Statistical correction
- **textdistance** - Levenshtein distance
- **nltk** - Natural language toolkit
- **matplotlib/seaborn** - Visualizations
- **pandas** - Data analysis

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Ishan Gautam**
- GitHub: [@ishangautam04](https://github.com/ishangautam04)
- Repository: [nlp-spelling-corrector](https://github.com/ishangautam04/nlp-spelling-corrector)

## 🙏 Acknowledgments

- PySpellChecker library
- AutoCorrect library
- NLTK Brown Corpus
- Streamlit framework

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@software{gautam2025spelling,
  author = {Gautam, Ishan},
  title = {NLP Spelling Corrector: Comparative Analysis of Correction Algorithms},
  year = {2025},
  url = {https://github.com/ishangautam04/nlp-spelling-corrector}
}
```

## 🚀 Future Work

- Add context-aware correction (BERT/GPT)
- Implement n-gram language models
- Add grammar correction
- Multi-language support
- Real-time typing correction
- API endpoint deployment

---

**⭐ Star this repo if you find it useful!**
