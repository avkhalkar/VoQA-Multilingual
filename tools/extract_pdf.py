import os
import sys

def extract():
    pdf_path = "Detailed_VoQA.pdf"
    out_path = "pdf_extracted.txt"
    
    try:
        import fitz
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text() for page in doc])
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print("Extracted via PyMuPDF")
        return
    except Exception as e:
        print("PyMuPDF failed:", e)
        
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(pdf_path)
        text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print("Extracted via PyPDF2")
    except Exception as e:
        print("PyPDF2 failed:", e)

if __name__ == "__main__":
    extract()
