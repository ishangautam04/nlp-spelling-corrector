
# LaTeX Compilation Instructions

## Prerequisites
1. Install a LaTeX distribution:
   - **Windows**: MiKTeX or TeX Live
   - **macOS**: MacTeX
   - **Linux**: TeX Live

## Compilation Commands

### Method 1: Basic compilation
```bash
pdflatex NLP_Spelling_Corrector_Project_Report.tex
pdflatex NLP_Spelling_Corrector_Project_Report.tex  # Run twice for references
```

### Method 2: Full compilation with bibliography (if needed)
```bash
pdflatex NLP_Spelling_Corrector_Project_Report.tex
bibtex NLP_Spelling_Corrector_Project_Report
pdflatex NLP_Spelling_Corrector_Project_Report.tex
pdflatex NLP_Spelling_Corrector_Project_Report.tex
```

### Method 3: Using latexmk (recommended)
```bash
latexmk -pdf NLP_Spelling_Corrector_Project_Report.tex
```

## Online LaTeX Editors
If you don't have LaTeX installed locally, you can use:
- **Overleaf**: https://www.overleaf.com/
- **ShareLaTeX**: Integrated into Overleaf
- **TeXmaker Online**: Various online options

Simply upload the .tex file to any of these platforms and compile online.

## Output
After successful compilation, you'll get:
- `NLP_Spelling_Corrector_Project_Report.pdf` - Your final document
- Various auxiliary files (.aux, .log, .toc, etc.) - Can be deleted

## Troubleshooting
- If you get package errors, install missing packages through your LaTeX distribution
- Run compilation twice to resolve cross-references
- Check the .log file for detailed error information

## Customization
The LaTeX file includes:
- Professional academic formatting
- Syntax-highlighted code blocks
- Proper table formatting
- Cross-references and hyperlinks
- Academic title page
