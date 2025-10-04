#!/usr/bin/env python3
"""
LaTeX Report Generator for NLP Spelling Corrector Project
Converts the Markdown project report to a well-formatted LaTeX document.
"""

import re
import os
from pathlib import Path
from datetime import datetime

def markdown_to_latex(md_content):
    """
    Convert markdown content to LaTeX format with proper academic styling.
    """
    
    # Start with LaTeX document structure
    latex_content = r"""\documentclass[12pt,a4paper]{article}

% Packages
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{geometry}
\usepackage{fancyhdr}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{multirow}
\usepackage{multicol}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage{tocloft}
\usepackage{algorithm}
\usepackage{algorithmic}

% Page setup
\geometry{
    left=1in,
    right=1in,
    top=1in,
    bottom=1in,
    headheight=14pt
}

% Header and footer
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{NLP Spelling Corrector Project}
\fancyhead[R]{\thepage}
\fancyfoot[C]{Academic Project Report - 2025}

% Code listing style
\lstdefinestyle{pythonstyle}{
    language=Python,
    basicstyle=\ttfamily\footnotesize,
    keywordstyle=\color{blue}\bfseries,
    commentstyle=\color{green!60!black},
    stringstyle=\color{red},
    numberstyle=\tiny\color{gray},
    breaklines=true,
    showstringspaces=false,
    frame=single,
    backgroundcolor=\color{gray!10},
    captionpos=b,
    numbers=left,
    numbersep=5pt,
    tabsize=4
}

\lstset{style=pythonstyle}

% Custom colors
\definecolor{headercolor}{RGB}{44, 62, 80}
\definecolor{subheadercolor}{RGB}{52, 73, 94}
\definecolor{highlightcolor}{RGB}{52, 152, 219}

% Title formatting
\titleformat{\section}
  {\normalfont\Large\bfseries\color{headercolor}}
  {\thesection}{1em}{}

\titleformat{\subsection}
  {\normalfont\large\bfseries\color{subheadercolor}}
  {\thesubsection}{1em}{}

\titleformat{\subsubsection}
  {\normalfont\normalsize\bfseries\color{subheadercolor}}
  {\thesubsubsection}{1em}{}

% Hyperlink setup
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    urlcolor=blue,
    citecolor=blue,
    pdfauthor={Student},
    pdftitle={NLP Spelling Corrector Project Report},
    pdfsubject={Natural Language Processing},
    pdfkeywords={NLP, Spelling Correction, Machine Learning, Python}
}

% Document start
\begin{document}

% Title page
\begin{titlepage}
    \centering
    \vspace*{2cm}
    
    {\Huge\bfseries\color{headercolor} NLP Spelling Corrector Project Report\par}
    \vspace{1cm}
    {\Large\itshape Advanced Natural Language Processing Implementation\par}
    \vspace{2cm}
    
    {\Large
    \textbf{Course:} Natural Language Processing\\[0.5cm]
    \textbf{Academic Year:} 2024-2025\\[0.5cm]
    \textbf{Date:} """ + datetime.now().strftime("%B %d, %Y") + r"""\\[2cm]
    }
    
    {\large
    \textbf{Project Overview:}\\[0.5cm]
    Advanced implementation of a multi-algorithm ensemble spelling corrector\\
    incorporating machine learning techniques and comprehensive evaluation\\[1cm]
    }
    
    \vfill
    
    {\large
    \textbf{Key Technologies:} Python, NLTK, Scikit-learn, Streamlit\\
    \textbf{Algorithms:} Ensemble Learning, Edit Distance, Statistical Models\\
    \textbf{Accuracy Achieved:} 93.4\% on test datasets
    }
    
    \vspace{1cm}
\end{titlepage}

% Table of contents
\tableofcontents
\newpage

"""
    
    # Process the markdown content
    lines = md_content.split('\n')
    in_code_block = False
    code_language = ""
    table_mode = False
    
    for line in lines:
        # Skip title page content (first few lines)
        if line.startswith('# NLP Spelling Corrector Project Report'):
            continue
        if 'Academic Project Report' in line or 'Course: Natural Language Processing' in line:
            continue
        if line.startswith('**Date:**') or line.startswith('**Academic Year:**'):
            continue
        if line.strip() == '---':
            continue
            
        # Handle code blocks
        if line.startswith('```'):
            if not in_code_block:
                in_code_block = True
                code_language = line[3:].strip() or 'text'
                latex_content += f"\\begin{{lstlisting}}[language={code_language}]\n"
            else:
                in_code_block = False
                latex_content += "\\end{lstlisting}\n\n"
            continue
        
        if in_code_block:
            # Escape special LaTeX characters in code
            escaped_line = line.replace('\\', '\\textbackslash{}')
            escaped_line = escaped_line.replace('{', '\\{').replace('}', '\\}')
            escaped_line = escaped_line.replace('_', '\\_')
            latex_content += escaped_line + "\n"
            continue
        
        # Handle headers
        if line.startswith('# '):
            title = line[2:].strip()
            latex_content += f"\\section{{{escape_latex(title)}}}\n\n"
        elif line.startswith('## '):
            title = line[3:].strip()
            latex_content += f"\\subsection{{{escape_latex(title)}}}\n\n"
        elif line.startswith('### '):
            title = line[4:].strip()
            latex_content += f"\\subsubsection{{{escape_latex(title)}}}\n\n"
        elif line.startswith('#### '):
            title = line[5:].strip()
            latex_content += f"\\paragraph{{{escape_latex(title)}}}\n\n"
        
        # Handle tables
        elif '|' in line and line.strip():
            if not table_mode:
                table_mode = True
                # Count columns
                cols = len([x for x in line.split('|') if x.strip()])
                col_spec = 'l' * cols
                latex_content += f"\\begin{{longtable}}{{{col_spec}}}\n"
                latex_content += "\\toprule\n"
            
            if line.strip().startswith('|---') or line.strip().startswith(':---'):
                latex_content += "\\midrule\n"
            else:
                cells = [escape_latex(cell.strip()) for cell in line.split('|') if cell.strip()]
                if cells:
                    latex_content += " & ".join(cells) + " \\\\\n"
        
        # Handle lists
        elif line.strip().startswith('- ') or line.strip().startswith('* '):
            if not latex_content.endswith('\\begin{itemize}\n'):
                latex_content += "\\begin{itemize}\n"
            item_text = line.strip()[2:].strip()
            latex_content += f"\\item {escape_latex(item_text)}\n"
        elif line.strip().startswith(('1. ', '2. ', '3. ', '4. ', '5. ', '6. ', '7. ', '8. ', '9. ')):
            if not latex_content.endswith('\\begin{enumerate}\n'):
                latex_content += "\\begin{enumerate}\n"
            item_text = re.sub(r'^\d+\.\s*', '', line.strip())
            latex_content += f"\\item {escape_latex(item_text)}\n"
        
        # Handle emphasis and inline code
        elif line.strip():
            if table_mode and '|' not in line:
                table_mode = False
                latex_content += "\\bottomrule\n\\end{longtable}\n\n"
            
            # End lists if needed
            if not line.strip().startswith(('- ', '* ', '1. ', '2. ', '3. ', '4. ', '5. ', '6. ', '7. ', '8. ', '9. ')):
                if latex_content.endswith('\\begin{itemize}\n') or '\\item ' in latex_content.split('\n')[-2:]:
                    if 'itemize' in latex_content and not latex_content.endswith('\\end{itemize}\n'):
                        latex_content += "\\end{itemize}\n\n"
                if latex_content.endswith('\\begin{enumerate}\n') or ('\\item ' in latex_content.split('\n')[-2:] and 'enumerate' in latex_content):
                    if 'enumerate' in latex_content and not latex_content.endswith('\\end{enumerate}\n'):
                        latex_content += "\\end{enumerate}\n\n"
            
            # Process inline formatting
            processed_line = escape_latex(line)
            
            # Bold text
            processed_line = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', processed_line)
            
            # Italic text
            processed_line = re.sub(r'\*(.*?)\*', r'\\textit{\1}', processed_line)
            
            # Inline code
            processed_line = re.sub(r'`(.*?)`', r'\\texttt{\1}', processed_line)
            
            latex_content += processed_line + "\n\n"
        
        else:
            # Empty line
            if table_mode:
                table_mode = False
                latex_content += "\\bottomrule\n\\end{longtable}\n\n"
            latex_content += "\n"
    
    # Close any remaining environments
    if table_mode:
        latex_content += "\\bottomrule\n\\end{longtable}\n\n"
    
    # End document
    latex_content += "\\end{document}\n"
    
    return latex_content

def escape_latex(text):
    """
    Escape special LaTeX characters in text.
    """
    # Dictionary of replacements
    replacements = {
        '&': '\\&',
        '%': '\\%',
        '$': '\\$',
        '#': '\\#',
        '^': '\\textasciicircum{}',
        '_': '\\_',
        '{': '\\{',
        '}': '\\}',
        '~': '\\textasciitilde{}',
        '\\': '\\textbackslash{}'
    }
    
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    return text

def create_latex_report():
    """
    Convert the PROJECT_REPORT.md file to a professional LaTeX document.
    """
    
    # File paths
    md_file = "PROJECT_REPORT.md"
    tex_file = "NLP_Spelling_Corrector_Project_Report.tex"
    
    print("🔄 Converting Markdown to LaTeX...")
    
    try:
        # Check if markdown file exists
        if not os.path.exists(md_file):
            print(f"❌ Error: {md_file} not found!")
            return False
        
        # Read markdown content
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        print("📝 Processing markdown content...")
        
        # Convert to LaTeX
        latex_content = markdown_to_latex(md_content)
        
        # Write LaTeX file
        with open(tex_file, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        print(f"✅ LaTeX file created: {tex_file}")
        
        # Get file size
        tex_size = os.path.getsize(tex_file) / 1024  # Convert to KB
        print(f"📄 LaTeX file size: {tex_size:.2f} KB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting to LaTeX: {e}")
        return False

def create_compilation_instructions():
    """
    Create instructions for compiling the LaTeX document.
    """
    
    instructions = """
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
"""
    
    with open("LATEX_COMPILATION_GUIDE.md", 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    print("📋 Compilation guide created: LATEX_COMPILATION_GUIDE.md")

def main():
    """
    Main function to generate LaTeX report.
    """
    print("=" * 60)
    print("📝 NLP Spelling Corrector - LaTeX Report Generator")
    print("=" * 60)
    
    # Check if running in correct directory
    if not os.path.exists("PROJECT_REPORT.md"):
        print("❌ Error: PROJECT_REPORT.md not found in current directory!")
        print("💡 Please run this script from the project directory containing PROJECT_REPORT.md")
        return
    
    try:
        # Generate LaTeX
        success = create_latex_report()
        
        if success:
            # Create compilation instructions
            create_compilation_instructions()
            
            print("\n" + "=" * 60)
            print("🎉 LaTeX Report Generation Complete!")
            print("=" * 60)
            print("📄 Files created:")
            print("   - PROJECT_REPORT.md (Original Markdown)")
            print("   - NLP_Spelling_Corrector_Project_Report.tex (LaTeX source)")
            print("   - LATEX_COMPILATION_GUIDE.md (Compilation instructions)")
            print("\n💡 Next steps:")
            print("   1. Install LaTeX distribution (MiKTeX, TeX Live, or MacTeX)")
            print("   2. Run: pdflatex NLP_Spelling_Corrector_Project_Report.tex")
            print("   3. Or use Overleaf for online compilation")
            print("\n🎯 The LaTeX file includes:")
            print("   - Professional academic formatting")
            print("   - Syntax-highlighted code blocks")
            print("   - Proper tables and cross-references")
            print("   - Academic title page")
        else:
            print("\n" + "=" * 60)
            print("⚠️  LaTeX Generation Failed")
            print("=" * 60)
            print("📄 Markdown report is still available: PROJECT_REPORT.md")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()