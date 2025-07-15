import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

# 0. Cargar variables de entorno
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "knowledge_navigator"
PDF_DIR = "data/pdfs"

# 1. Cargar documentos PDF con metadatos
all_docs = []
print("📥 Cargando documentos...")
for filename in os.listdir(PDF_DIR):
    if filename.endswith(".pdf"):
        path = os.path.join(PDF_DIR, filename)
        loader = PyPDFLoader(path)
        docs = loader.load()  # documentos por página
        for i, doc in enumerate(docs):
            doc.metadata["source"] = filename
            doc.metadata["page"] = i + 1
        all_docs.extend(docs)
print(f"✅ Se han cargado {len(all_docs)} páginas de documentos.")

# 2. Segmentar en fragmentos
print("✂️ Dividiendo en fragmentos...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " "]
)
chunks = splitter.split_documents(all_docs)
print(f"✅ Fragmentos generados: {len(chunks)}")

# 3. Embeddings con modelo local Ollama
print("🧠 Generando embeddings...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")


# 4. Indexar en Qdrant Cloud
print("📡 Conectando con Qdrant Cloud...")
print(f"📦 Guardando en colección '{COLLECTION_NAME}'...")

vectorstore = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    url=QDRANT_URL,
    api_key=API_KEY,
    collection_name=COLLECTION_NAME,
    prefer_grpc=False
)

print("✅ ¡Vector store cargado exitosamente en Qdrant!")
