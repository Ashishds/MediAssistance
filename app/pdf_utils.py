from pypdf import PdfReader

def read_pdf_text(file) -> str:
    """Extract raw text from a PDF file"""
    reader = PdfReader(file)
    content = ''
    for pg in reader.pages:
        content += pg.extract_text() or ''
    return content
