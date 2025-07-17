from pymongo import MongoClient
from langchain_ollama import OllamaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

# --- CONFIGURACIÓN DESDE .env ---
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB_NAME", "knowledge_navigator")
COLLECTION_NAME = "historial"
UMBRAL_SIMILITUD = 0.92

# --- FUNCIONES ---
def obtener_preguntas_historial():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    return [doc["pregunta"] for doc in collection.find({}) if "pregunta" in doc]

def detectar_redundancia(nueva_pregunta):
    preguntas_previas = obtener_preguntas_historial()
    if not preguntas_previas:
        return None

    embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
    vectores_previos = embeddings.embed_documents(preguntas_previas)
    vector_nuevo = embeddings.embed_query(nueva_pregunta)

    similitudes = cosine_similarity([vector_nuevo], vectores_previos)[0]
    similitud_max = max(similitudes)
    indice = similitudes.argmax()

    if similitud_max > UMBRAL_SIMILITUD:
        return f"⚠️ Esta pregunta es similar a una ya realizada:\n➡ {preguntas_previas[indice]}"
    return None

# --- USO DE EJEMPLO ---
if __name__ == "__main__":
    pregunta = input("Introduce tu pregunta: ")
    resultado = detectar_redundancia(pregunta)
    if resultado:
        print(resultado)
    else:
        print("✅ Pregunta nueva. Puedes generar una respuesta.")