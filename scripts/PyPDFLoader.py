from langchain_community.document_loaders import PyPDFLoader
import os

pdf_dir = "data/pdfs"
pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

all_docs = []

for file in pdf_files:
    file_path = os.path.join(pdf_dir, file)
    loader = PyPDFLoader(file_path)
    docs = loader.load()  # Devuelve una lista de documentos, uno por página
    for i, doc in enumerate(docs):
        doc.metadata["source"] = file
        doc.metadata["page"] = i + 1  # empezamos en 1 para humanos
    all_docs.extend(docs)
