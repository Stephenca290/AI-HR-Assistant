import PyPDF2

def extract_text_from_pdf(uploaded_files):
    texts = []
    for file in uploaded_files:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        texts.append(text)
    return texts
