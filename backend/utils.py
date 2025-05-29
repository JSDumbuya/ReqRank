import pandas as pd
from PyPDF2 import PdfReader
from docx import Document

def create_csv(text, file_name):
    if isinstance(text, pd.DataFrame):
        df = text
    else:
        df = pd.DataFrame(text)
    df.to_csv(file_name, index=False, header=True)

def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def read_docx(file_path):
    doc = Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def read_csv(file_path):
    return pd.read_csv(file_path)

def read_excel(file_path):
    file_extension = file_path.split('.')[-1].lower()
    
    if file_extension == "xlsx":
        return pd.read_excel(file_path, engine="openpyxl")
    elif file_extension == "xls":
        return pd.read_excel(file_path, engine="xlrd")
    else:
        raise ValueError("Unsupported Excel file format.")

def read_file(file_path):
    file_extension = file_path.split('.')[-1].lower()

    if file_extension == 'txt':
        return read_txt(file_path)
    elif file_extension == 'pdf':
        return read_pdf(file_path)
    elif file_extension == 'docx':
        return read_docx(file_path)
    elif file_extension == 'csv':
        return read_csv(file_path)
    elif file_extension in ['xlsx', 'xls']:
        return read_excel(file_path)
    else:
        raise ValueError("Unsupported file type")



