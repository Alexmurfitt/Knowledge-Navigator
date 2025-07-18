import os
import json
from pymongo import MongoClient
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "knowledge_navigator")

# Conectar a MongoDB
client = MongoClient(MONGO_URI)
db = client[MONGO_DB_NAME]
collection = db["historial"]

# Extraer y reconstruir las entradas
historial_valido = []

for doc in collection.find({}):
    if "pregunta" in doc and "respuesta_rag" in doc and "fuentes" in doc:
        pregunta = doc["pregunta"]
        respuesta = doc["respuesta_rag"]

        # Construir contexto textual a partir de "fuentes"
        contexto = []
        for fuente in doc["fuentes"]:
            doc_name = fuente.get("documento", "desconocido")
            page = fuente.get("pagina", "N/A")
            contexto.append(f"Documento: {doc_name}, Página: {page}")

        # Solo añadir si hay contexto no vacío
        if contexto:
            entrada = {
                "pregunta": pregunta,
                "respuesta_rag": respuesta,
                "contexto": contexto
            }
            historial_valido.append(entrada)

# Guardar archivo final compatible con RAGAS
output_path = os.path.join("scripts", "historial_ragas.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(historial_valido, f, ensure_ascii=False, indent=2)

print(f"✅ Historial reconstruido con {len(historial_valido)} entradas válidas.")
print(f"📁 Guardado en: {output_path}")
