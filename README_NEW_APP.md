# 🔤 Spelling Correction Comparison Tool

## Overview

This is a **NEW comparison-focused** Streamlit web application that compares 4 different spelling correction algorithms side-by-side, **without using ensemble methods**. Perfect for research, education, and understanding algorithm performance.

## ✨ Key Features

### 🏠 Live Comparison Page
- **Side-by-side algorithm comparison**: See how all 4 algorithms perform on the same input
- **Multiple input methods**:
  - ✍️ Type your own text
  - 📄 Use pre-loaded examples
  - 📁 Load test cases from dataset
- **Real-time metrics**: Processing time, correction count, accuracy scores
- **Winner detection**: Automatically highlights the best-performing algorithm
- **Detailed corrections**: View exactly what each algorithm changed

### 📈 Performance Benchmarks Page
- **Accuracy comparison charts**: Visual bar charts showing algorithm performance
- **Error-type analysis**: How each algorithm performs on different error types
- **Dataset statistics**: Complete information about the test dataset (433 samples)
- **Interactive visualizations**: Matplotlib/Seaborn charts

### 🔬 Algorithm Details Page
- **In-depth explanations**: How each algorithm works
- **Strengths & weaknesses**: Honest assessment of each method
- **Use case recommendations**: When to use which algorithm
- **Technical specifications**: Algorithm type, speed, accuracy

### 📚 About Page
- **Project overview**: Complete project description
- **Methodology**: How algorithms were evaluated
- **Key findings**: Research results and insights
- **Technology stack**: All libraries and tools used

## 🎯 Algorithms Compared

| Algorithm | Type | Accuracy | Speed | Best For |
|-----------|------|----------|-------|----------|
| **PySpellChecker** | Statistical Dictionary | 92.6% | Fast | General text, common typos |
| **AutoCorrect** | Pattern-based ML | 91.2% | Fast | Phonetic errors, mobile keyboards |
| **Frequency-Based** | Statistical Frequency | 75.3% | Medium | Research, baseline comparison |
| **Levenshtein** | Edit Distance | 64.2% | Slow | String matching, academic study |

## 🚀 Getting Started

### Prerequisites
```bash
python 3.x
pip install streamlit spellchecker autocorrect textdistance nltk pandas matplotlib seaborn
```

### Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download NLTK data** (first time only):
   ```python
   import nltk
   nltk.download('words')
   nltk.download('brown')
   ```

### Running the App

#### Option 1: Using the Launcher (Recommended)
```bash
launch_app.bat
```
Then select option `[1] NEW: Algorithm Comparison Tool`

#### Option 2: Direct Launch
```bash
start_web_app_new.bat
```

#### Option 3: Manual Launch
```bash
.venv\Scripts\activate
streamlit run streamlit_app_new.py --server.port 8504
```

The app will open in your browser at `http://localhost:8504`

## 📊 Test Dataset

- **Size**: 433 samples
- **Error Types**: 4 categories
  - Keyboard Errors (e.g., teh → the)
  - Phonetic Errors (e.g., nite → night)
  - Character Errors (e.g., recieve → receive)
  - Simple Typos (e.g., freind → friend)

## 🆚 Differences from Original App

### New App (streamlit_app_new.py)
✅ **Focus**: Algorithm comparison  
✅ **Approach**: Individual algorithms only  
✅ **Features**: Side-by-side comparison, benchmarks, detailed analysis  
✅ **UI**: Clean, comparison-focused interface  
✅ **Best for**: Research, education, algorithm evaluation  

### Original App (streamlit_app.py)
✅ **Focus**: Full-featured correction tool  
✅ **Approach**: Includes ensemble methods and ML models  
✅ **Features**: Training data, ML comparison, comprehensive correction  
✅ **UI**: Feature-rich interface  
✅ **Best for**: Production use, end-users  

## 🎨 Screenshots

### Live Comparison
- Input text with errors
- See 4 algorithm outputs side-by-side
- Compare corrections, time, and accuracy

### Performance Benchmarks
- Bar charts showing accuracy comparison
- Error-type performance analysis
- Dataset composition breakdown

### Algorithm Details
- Technical specifications
- Strengths and weaknesses
- Use case recommendations

## 📈 Performance Results

Based on 433-sample test dataset:

```
PySpellChecker:    92.6% accuracy ⭐ WINNER
AutoCorrect:       91.2% accuracy
Frequency-Based:   75.3% accuracy
Levenshtein:       64.2% accuracy
```

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Algorithms**: 
  - PySpellChecker
  - AutoCorrect
  - TextDistance (Levenshtein)
  - NLTK (Corpus)
- **Visualization**: Matplotlib, Seaborn
- **Data**: Pandas, NumPy

## 📝 Usage Examples

### Example 1: Simple Typo Correction
```
Input:  "The qick brown fox jumps over the lazy dog"
Output: See how each algorithm handles "qick" → "quick"
```

### Example 2: Multiple Errors
```
Input:  "I recieved your mesage yestarday"
Output: Compare correction quality across algorithms
```

### Example 3: Technical Text
```
Input:  "Artifical inteligence is revolutionizing tecnology"
Output: Evaluate performance on technical terms
```

## 🔧 Customization

### Adding New Test Examples
Edit the `examples` dictionary in `show_live_comparison()`:
```python
examples = {
    "Your Category": "Your test text here",
    ...
}
```

### Adjusting Algorithms
Modify the `run_all_algorithms()` function to:
- Add new algorithms
- Change evaluation metrics
- Adjust correction parameters

### Changing UI Theme
Modify the custom CSS in the `st.markdown()` section at the top of the file.

## 📚 Documentation

### Project Files
```
📁 nlp-spelling-corrector/
├── streamlit_app_new.py          ← This app
├── streamlit_app.py               ← Original app
├── launch_app.bat                 ← App launcher
├── start_web_app_new.bat          ← Direct launcher
├── generate_test_dataset.py       ← Dataset generation
├── evaluate_spelling_methods.py   ← Evaluation script
├── spelling_test_dataset_small.json
├── README_NEW_APP.md              ← This file
└── requirements.txt
```

## 🐛 Troubleshooting

### App won't start
- Check virtual environment is activated
- Install missing dependencies: `pip install -r requirements.txt`
- Verify Python version: `python --version` (should be 3.x)

### NLTK errors
```python
import nltk
nltk.download('words')
nltk.download('brown')
```

### Port already in use
Change the port in the launch command:
```bash
streamlit run streamlit_app_new.py --server.port 8505
```

## 🤝 Contributing

Suggestions for improvements:
- Add more algorithms (SymSpell, BK-Tree)
- Implement context-aware correction
- Add batch file processing
- Export comparison results to CSV/Excel
- Add performance profiling

## 📄 License

This project is part of the NLP Spelling Corrector research project.

## 👤 Author

Built with ❤️ for NLP Research and Education

---

## 🚀 Quick Start Commands

```bash
# Activate virtual environment
.venv\Scripts\activate

# Run the new comparison app
streamlit run streamlit_app_new.py --server.port 8504

# Or use the launcher
launch_app.bat
```

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review algorithm documentation
3. Check project reports in the repository

---

**Last Updated**: October 2025  
**Version**: 2.0 (Comparison-Focused)
