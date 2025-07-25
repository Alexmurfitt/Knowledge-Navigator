# verificar_loomis_qdrant.py
# ──────────────────────────────────────────────────────
# Objetivo: Verificar que Qdrant devuelva fragmentos sobre Eric Loomis
# con metadatos enriquecidos (score, source, page, contenido)
# ──────────────────────────────────────────────────────

import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# 1. 🔐 Cargar configuración
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "knowledge-navigator"

# 2. 🔗 Conectar a Qdrant
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

vectorstore = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

# 3. 🔍 Consulta a verificar
consulta = "Eric Loomis y el uso de algoritmos COMPAS en decisiones judiciales"
print(f"\n🔎 Consultando Qdrant con: {consulta}\n")

# 4. 🔁 Recuperar documentos relevantes con puntuación
docs_con_score = vectorstore.similarity_search_with_score(consulta, k=5)

for i, (doc, score) in enumerate(docs_con_score, start=1):
    metadata = doc.metadata
    print(f"📄 Fragmento #{i}")
    print("─" * 40)
    print(f"🔢 Score: {score:.4f}")
    print(f"📁 Documento: {metadata.get('source', 'desconocido')}")
    print(f"📄 Página: {metadata.get('page', 'desconocida')}")
    print(f"\n{doc.page_content.strip()}\n")
