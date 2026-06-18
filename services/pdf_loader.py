import re
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []
    
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append({
                "page": page_num,
                "text": text
            })
    
    return pages


def chunk_text(pages, chunk_size=900, overlap=150):
    chunks = []

    for page in pages:
        words = page["text"].split()
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append({
                "page": page["page"],
                "text": chunk
            })
            start += chunk_size - overlap

    return chunks
