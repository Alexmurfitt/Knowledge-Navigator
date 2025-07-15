import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Base path del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ruta absoluta a la carpeta de PDFs
PDF_DIR = os.path.abspath(os.path.join(BASE_DIR, "../data/pdfs"))

# Ruta donde guardar la base de datos vectorial
VECTOR_DIR = os.path.abspath(os.path.join(BASE_DIR, "../data/vectorstore"))

# Cargar y fragmentar los PDFs
documents = []
for filename in os.listdir(PDF_DIR):
    if filename.endswith(".pdf"):
        file_path = os.path.join(PDF_DIR, filename)
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(documents)

# Embeddings y base vectorial FAISS
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(split_docs, embedding_model)
db.save_local(VECTOR_DIR)

print("✅ PDFs vectorizados y guardados exitosamente en:", VECTOR_DIR)
