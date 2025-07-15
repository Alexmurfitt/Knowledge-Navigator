import os
from pathlib import Path
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.vectorstores import Qdrant

# Configuración
PDF_DIR = "../data/pdfs"
VECTOR_DB_DIR = "../vector_db"
COLLECTION_NAME = "knowledge_navigator"

# Paso 1: Cargar PDFs
pdf_files = [str(Path(PDF_DIR) / f) for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
docs = []
for pdf in pdf_files:
    loader = PyMuPDFLoader(pdf)
    docs.extend(loader.load())

# Paso 2: Dividir texto en fragmentos
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Paso 3: Crear embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Paso 4: Inicializar Qdrant
client = QdrantClient(path=VECTOR_DB_DIR)
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# Paso 5: Almacenar chunks en Qdrant
qdrant = Qdrant(client=client, collection_name=COLLECTION_NAME, embeddings=embeddings)
qdrant.add_documents(chunks)

print("✅ Vectorización completada. Documentos cargados en Qdrant.")
