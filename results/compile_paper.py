#!/usr/bin/env python3
"""
Simple LaTeX to PDF compiler for the research paper.
"""

import os
import subprocess
import sys

def compile_latex_to_pdf(tex_file: str, output_dir: str = None) -> bool:
    """
    Compile LaTeX file to PDF using pdflatex.
    
    Args:
        tex_file: Path to the .tex file
        output_dir: Output directory for the PDF (optional)
    
    Returns:
        bool: True if compilation successful, False otherwise
    """
    if not os.path.exists(tex_file):
        print(f"Error: LaTeX file '{tex_file}' not found!")
        return False
    
    # Get the directory containing the tex file
    tex_dir = os.path.dirname(os.path.abspath(tex_file))
    tex_filename = os.path.basename(tex_file)
    
    # Change to the tex file directory
    original_dir = os.getcwd()
    os.chdir(tex_dir)
    
    try:
        print(f"Compiling {tex_filename} to PDF...")
        
        # First pass
        result1 = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', tex_filename],
            capture_output=True,
            text=True
        )
        
        if result1.returncode != 0:
            print("Error in first pdflatex pass:")
            print(result1.stdout)
            print(result1.stderr)
            return False
        
        # Second pass (for references and citations)
        result2 = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', tex_filename],
            capture_output=True,
            text=True
        )
        
        if result2.returncode != 0:
            print("Error in second pdflatex pass:")
            print(result2.stdout)
            print(result2.stderr)
            return False
        
        # Clean up auxiliary files
        aux_extensions = ['.aux', '.log', '.out', '.toc', '.lof', '.lot', '.bbl', '.blg']
        base_name = tex_filename.replace('.tex', '')
        
        for ext in aux_extensions:
            aux_file = base_name + ext
            if os.path.exists(aux_file):
                os.remove(aux_file)
        
        pdf_file = base_name + '.pdf'
        if os.path.exists(pdf_file):
            print(f"✅ PDF successfully generated: {pdf_file}")
            
            # Move to output directory if specified
            if output_dir and output_dir != tex_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, pdf_file)
                os.rename(pdf_file, output_path)
                print(f"✅ PDF moved to: {output_path}")
            
            return True
        else:
            print("❌ PDF file was not generated!")
            return False
            
    except FileNotFoundError:
        print("❌ Error: pdflatex not found! Please install LaTeX distribution.")
        print("   - macOS: brew install --cask mactex")
        print("   - Ubuntu: sudo apt-get install texlive-full")
        print("   - Windows: Install MiKTeX or TeX Live")
        return False
    
    except Exception as e:
        print(f"❌ Unexpected error during compilation: {e}")
        return False
    
    finally:
        # Return to original directory
        os.chdir(original_dir)

def main():
    """Main function to compile the research paper."""
    
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tex_file = os.path.join(script_dir, "paper.tex")
    
    print("🔬 Research Paper PDF Compiler")
    print("=" * 50)
    print(f"LaTeX file: {tex_file}")
    
    if not os.path.exists(tex_file):
        print(f"❌ Error: {tex_file} not found!")
        sys.exit(1)
    
    # Compile the paper
    success = compile_latex_to_pdf(tex_file)
    
    if success:
        print("\n🎉 Paper compilation completed successfully!")
        print(f"📄 PDF location: {os.path.join(script_dir, 'paper.pdf')}")
        print("\n📊 Paper Statistics:")
        
        # Get basic statistics about the paper
        with open(tex_file, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            sections = len([line for line in lines if line.strip().startswith('\\section{')])
            tables = len([line for line in lines if '\\begin{table}' in line])
            figures = len([line for line in lines if '\\begin{figure}' in line])
            references = len([line for line in lines if '\\bibitem{' in line])
            
        print(f"   - Sections: {sections}")
        print(f"   - Tables: {tables}")  
        print(f"   - Figures: {figures}")
        print(f"   - References: {references}")
        print(f"   - Total lines: {len(lines)}")
        
    else:
        print("\n❌ Paper compilation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()