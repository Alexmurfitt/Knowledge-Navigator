import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings

# --- Configuración de entorno ---
load_dotenv()
QDRANT_URL = os.getenv("QDRANT-URL")
API_KEY = os.getenv("QDRANT-API-KEY")
COLLECTION_NAME = "knowledge_navigator"
DOCUMENTO_OBJETIVO = "1. Framework for the Ethical Use of Advanced Data Science.pdf"

# --- Conectar con Qdrant ---
client = QdrantClient(url=QDRANT_URL, api_key=API_KEY, prefer_grpc=False)

# --- MODELO DE EMBEDDINGS ---
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

# --- MODO 1: Buscar por metadato ("source") ---
def buscar_por_metadato():
    print(f"\n🔍 Buscando chunks del documento: {DOCUMENTO_OBJETIVO}")
    resultados = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter={
            "must": [
                {"key": "source", "match": {"value": DOCUMENTO_OBJETIVO}}
            ]
        },
        limit=50
    )
    puntos = resultados[0]
    if puntos:
        print(f"\n✅ Se encontraron {len(puntos)} chunks del documento '{DOCUMENTO_OBJETIVO}':\n")
        for i, punto in enumerate(puntos):
            texto = punto.payload.get("text", "")[:200]
            page = punto.payload.get("page", "N/A")
            print(f"[Chunk {i+1} | Página {page}] → {texto}...\n")
    else:
        print("❌ No se encontraron chunks con ese nombre de documento.")

# --- MODO 2: Buscar por pregunta (similaridad semántica) ---
def buscar_por_pregunta(pregunta, k=5):
    print(f"\n🧠 Buscando chunks similares a la pregunta:\n“{pregunta}”\n")
    embedding = embedding_model.embed_query(pregunta)
    resultados = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedding,
        limit=k
    )
    for i, r in enumerate(resultados):
        texto = r.payload.get("text", "")[:200]
        fuente = r.payload.get("source", "N/A")
        pagina = r.payload.get("page", "N/A")
        puntuacion = r.score
        print(f"[Resultado {i+1}] ({fuente} – pág. {pagina}) | Score: {puntuacion:.4f}")
        print(f"{texto}...\n")

# --- EJECUCIÓN INTERACTIVA ---
if __name__ == "__main__":
    print("\n🧪 Verificador de documento en Qdrant")

    modo = input("\nSelecciona modo (1 = por documento, 2 = por pregunta): ").strip()
    if modo == "1":
        buscar_por_metadato()
    elif modo == "2":
        pregunta = input("Introduce la pregunta a evaluar: ")
        buscar_por_pregunta(pregunta)
    else:
        print("❌ Modo inválido.")
