from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import OllamaEmbeddings
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Cargar entorno y conectar a Mongo
load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
collection = client[os.getenv("MONGO_DB_NAME")]["historial"]

# Extraer preguntas pasadas
preguntas_previas = [doc["pregunta"] for doc in collection.find({})]

# Vectorizar con Ollama
embedder = OllamaEmbeddings(model="nomic-embed-text")
vectores_historial = embedder.embed_documents(preguntas_previas)

# Nueva pregunta
nueva_pregunta = input("Nueva pregunta: ")
vector_nueva = embedder.embed_query(nueva_pregunta)

# Comparar similitud coseno
similaridades = cosine_similarity([vector_nueva], vectores_historial)[0]
indice = similaridades.argmax()
similitud_max = similaridades[indice]

if similitud_max > 0.92:
    print(f"⚠️ Pregunta muy similar ya fue hecha: '{preguntas_previas[indice]}'")
else:
    print("✅ Pregunta aceptada como nueva.")
