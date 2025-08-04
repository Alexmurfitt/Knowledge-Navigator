# ingest_pdf.py — Versión final optimizada para Knowledge Navigator
# --------------------------------------------------------------
# Carga e indexación de PDFs: extrae páginas → divide → filtra → genera embeddings → indexa en Qdrant
# --------------------------------------------------------------

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings

# =============================================================
# 1. CARGA DE VARIABLES DE ENTORNO Y CONFIGURACIÓN
# =============================================================
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "knowledge-navigator"
PDF_DIR = "data/pdfs"

# Validación básica
if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("❌ Error: Faltan QDRANT_URL o QDRANT_API_KEY en el archivo .env")

# =============================================================
# 2. CARGAR DOCUMENTOS PDF Y EXTRAER TEXTO POR PÁGINA
# =============================================================
def cargar_documentos(pdf_dir):
    documentos = []
    paginas_ignoradas = 0
    print("📥 Cargando documentos PDF...")

    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            path = os.path.join(pdf_dir, filename)
            loader = PyPDFLoader(path)
            docs_por_pagina = loader.load()

            for i, doc in enumerate(docs_por_pagina):
                texto = doc.page_content.strip()
                if not texto:
                    print(f"⚠️ Página vacía ignorada: {filename}, página {i+1}")
                    paginas_ignoradas += 1
                    continue
                doc.metadata["source"] = filename
                doc.metadata["page"] = i + 1
                documentos.append(doc)

    print(f"✅ Se han cargado {len(documentos)} páginas válidas (ignoradas: {paginas_ignoradas})")
    return documentos

# =============================================================
# 3. DIVIDIR DOCUMENTOS EN FRAGMENTOS SEMÁNTICAMENTE COHERENTES
# =============================================================
def dividir_en_fragmentos(documentos):
    print("✂️ Dividiendo en fragmentos...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    todos = splitter.split_documents(documentos)
    chunks_utiles = [c for c in todos if es_fragmento_util(c.page_content)]

    print(f"✅ Fragmentos útiles: {len(chunks_utiles)} (de {len(todos)} totales)")
    return chunks_utiles

# =============================================================
# 4. FILTRO: DESCARTAR FRAGMENTOS POCO INFORMATIVOS
# =============================================================
def es_fragmento_util(texto: str) -> bool:
    texto = texto.strip().lower()
    return (
        len(texto) >= 300 and
        not texto.startswith("índice") and
        "copyright" not in texto and
        not texto.isdigit()
    )

# =============================================================
# 5. GENERAR EMBEDDINGS E INDEXAR EN QDRANT CLOUD
# =============================================================
def indexar_en_qdrant(chunks):
    print("🔗 Generando embeddings y conectando con Qdrant...")
    embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
        force_recreate=True  # 🔄 Evita error si la colección ya existe con distinta dimensión
    )

    print(f"✅ ¡Colección '{COLLECTION_NAME}' indexada correctamente en Qdrant!")

# =============================================================
# 6. EJECUCIÓN DEL FLUJO COMPLETO
# =============================================================
if __name__ == "__main__":
    print("🚀 Iniciando ingestión de documentos...")
    documentos = cargar_documentos(PDF_DIR)
    fragmentos = dividir_en_fragmentos(documentos)
    indexar_en_qdrant(fragmentos)
    print("🎯 Ingestión finalizada con éxito.")
