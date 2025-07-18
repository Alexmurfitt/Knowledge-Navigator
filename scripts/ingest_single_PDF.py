# ingest_pdf_qdrant.py
# ───────────────────────────────────────────────
# Subida de un único PDF a Qdrant
# Proyecto: Knowledge Navigator
# Función: Procesar un PDF → dividir en fragmentos → generar embeddings → indexar en Qdrant Cloud
# ───────────────────────────────────────────────

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

# 0. 🔐 Cargar variables de entorno
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "knowledge_navigator"
PDF_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "pdfs")
PDF_DIR = os.path.abspath(PDF_DIR)


# 📄 Nombre del PDF que deseas subir (¡ajústalo aquí!)
PDF_NAME = "8_Tablas_Y_Texto.pdf"

# 1. 📥 Cargar el PDF con metadatos por página
def cargar_un_pdf(pdf_dir, nombre_pdf):
    documentos = []
    path = os.path.join(pdf_dir, nombre_pdf)

    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ El archivo '{nombre_pdf}' no existe en {pdf_dir}")

    print(f"📥 Cargando documento: {nombre_pdf}")
    loader = PyPDFLoader(path)
    docs_por_pagina = loader.load()

    paginas_ignoradas = 0
    for i, doc in enumerate(docs_por_pagina):
        if not doc.page_content.strip():
            print(f"⚠️ Página vacía ignorada: {nombre_pdf}, página {i+1}")
            paginas_ignoradas += 1
            continue

        doc.metadata["source"] = nombre_pdf
        doc.metadata["page"] = i + 1
        documentos.append(doc)

    print(f"✅ Se han cargado {len(documentos)} páginas válidas de '{nombre_pdf}'")
    return documentos

# 2. ✂️ Fragmentar en chunks con solapamiento
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

# 3. 🧠 Generar embeddings con Ollama
def generar_embeddings():
    print("🧠 Generando embeddings...")
    return OllamaEmbeddings(model="nomic-embed-text")

# 4. 📡 Conectar e indexar en Qdrant
def indexar_en_qdrant(chunks, embeddings):
    print("📡 Conectando con Qdrant Cloud...")
    print(f"📦 Guardando en colección '{COLLECTION_NAME}'...")
    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
        prefer_grpc=False
    )
    print("✅ ¡Vector store cargado exitosamente en Qdrant!")

# 🏁 Ejecución principal
if __name__ == "__main__":
    documentos = cargar_un_pdf(PDF_DIR, PDF_NAME)
    chunks = dividir_en_fragmentos(documentos)
    embeddings = generar_embeddings()
    indexar_en_qdrant(chunks, embeddings)
