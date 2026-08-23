import sys
try:
    from pypdf import PdfReader
    reader = PdfReader('d:/Main/Rnd_Projects/VoQA-Multilingual/Detailed_VoQA.pdf')
    text = '\n'.join([page.extract_text() for page in reader.pages[10:] if page.extract_text()])
    with open('d:/Main/Rnd_Projects/VoQA-Multilingual/pdf_extracted2.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    print("Success with pypdf")
except Exception as e:
    try:
        import fitz
        doc = fitz.open('d:/Main/Rnd_Projects/VoQA-Multilingual/Detailed_VoQA.pdf')
        text = '\n'.join([doc[i].get_text() for i in range(10, len(doc))])
        with open('d:/Main/Rnd_Projects/VoQA-Multilingual/pdf_extracted2.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        print("Success with fitz")
    except Exception as e2:
        print(f"Failed completely: {e} | {e2}")
