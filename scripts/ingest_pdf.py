# ingest_pdf_qdrant.py
# ───────────────────────────────────────────────
# Script de carga e indexación de documentos PDF
# Proyecto: Knowledge Navigator
# Función: Procesar PDFs → dividir en fragmentos → generar embeddings → indexar en Qdrant Cloud
# ───────────────────────────────────────────────

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
# 0. 🔐 Cargar variables de entorno desde .env
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "Knowledge-Navigator"
PDF_DIR = "data/pdfs"

# 1. 📥 Cargar documentos PDF con metadatos por página
def cargar_documentos(pdf_dir):
    documentos = []
    paginas_ignoradas = 0

    print("📥 Cargando documentos...")

    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            path = os.path.join(pdf_dir, filename)
            loader = PyPDFLoader(path)
            docs_por_pagina = loader.load()
            
            for i, doc in enumerate(docs_por_pagina):
                # Ignorar páginas vacías
                if not doc.page_content.strip():
                    print(f"⚠️ Página vacía ignorada: {filename}, página {i+1}")
                    paginas_ignoradas += 1
                    continue

                doc.metadata["source"] = filename
                doc.metadata["page"] = i + 1
                documentos.append(doc)
    
    print(f"✅ Se han cargado {len(documentos)} páginas válidas de documentos.")
    return documentos

# 2. ✂️ Fragmentar documentos en chunks con solapamiento
def dividir_en_fragmentos(documentos):
    print("✂️ Dividiendo en fragmentos...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documentos)
    print(f"✅ Fragmentos generados: {len(chunks)}")
    return chunks


# 3. 📡 Conectar con Qdrant y almacenar los fragmentos
def indexar_en_qdrant(chunks):
    embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
    print("📡 Conectando con Qdrant Cloud...")
    print(f"📦 Guardando en colección '{COLLECTION_NAME}'...")
    vectorstore = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    url= QDRANT_URL,
    api_key=QDRANT_API_KEY,
    collection_name=COLLECTION_NAME,
    # prefer_grpc=False,
    force_recreate=True  # 🟢 Añade esto para evitar conflicto dimensional
)

    print("✅ ¡Vector store cargado exitosamente en Qdrant!")

# 🏁 Ejecución principal
if __name__ == "__main__":
    documentos = cargar_documentos(PDF_DIR)
    chunks = dividir_en_fragmentos(documentos)
    # genembeddings = generar_embeddings()
    indexar_en_qdrant(chunks)
