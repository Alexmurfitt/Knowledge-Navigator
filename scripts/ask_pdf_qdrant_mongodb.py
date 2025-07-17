import os
import json
import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# ✅ Importar el razonador interno
from razonador_cot import razonamiento_cot

# 🧪 Cargar variables de entorno
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")

if MONGO_DB_NAME is None:
    raise ValueError("❌ Error: MONGO_DB_NAME no está definido. Verifica tu archivo .env")

# 🔐 Inicializar MongoDB
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client[MONGO_DB_NAME]
mongo_collection = mongo_db["historial"]

# 💬 Modelo de lenguaje Ollama
llm = ChatOllama(model="llama3", temperature=0.1)

# 🔍 Embeddings locales
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 🧠 Cliente Qdrant
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# 📦 Conectar con QdrantVectorStore
qdrant = QdrantVectorStore(
    client=client,
    collection_name="knowledge_navigator",
    embedding=embeddings
)
retriever = qdrant.as_retriever()

# 🧠 Memoria conversacional
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# 🔗 Cadena conversacional
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    output_key="answer"
)

# 📁 Guardar historial en JSON
def guardar_en_json(pregunta, razonamiento, respuesta, fuentes):
    entrada = {
        "pregunta": pregunta,
        "razonamiento_interno": razonamiento,
        "respuesta_rag": respuesta,
        "timestamp": datetime.datetime.now().isoformat(),
        "fuentes": [
            {"documento": doc.metadata.get("source", "desconocido"),
             "pagina": doc.metadata.get("page", "N/A")}
            for doc in fuentes
        ]
    }
    with open("historial.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(entrada, ensure_ascii=False) + "\n")

# 🍃 Guardar historial en MongoDB
def guardar_en_mongo(pregunta, razonamiento, respuesta, fuentes):
    entrada = {
        "pregunta": pregunta,
        "razonamiento_interno": razonamiento,
        "respuesta_rag": respuesta,
        "timestamp": datetime.datetime.now(),
        "fuentes": [
            {"documento": doc.metadata.get("source", "desconocido"),
             "pagina": doc.metadata.get("page", "N/A")}
            for doc in fuentes
        ]
    }
    mongo_collection.insert_one(entrada)

# 🤖 Responder con razonamiento interno + RAG
def responder(pregunta):
    print("\n🧠 Razonamiento interno (auto-preguntas):")
    razonamiento = razonamiento_cot(pregunta)
    print(razonamiento)

    print("\n🔍 Recuperando información con RAG...")
    resultado = qa_chain.invoke({"question": pregunta})
    respuesta = resultado["answer"]
    fuentes = resultado["source_documents"]

    print("\n📘 Respuesta basada en documentos:")
    print(respuesta)

    print("\n📚 Documentos fuente:")
    for i, doc in enumerate(fuentes):
        print(f"  {i+1}. {doc.metadata.get('source', 'desconocido')} (p. {doc.metadata.get('page', 'N/A')})")

    guardar_en_json(pregunta, razonamiento, respuesta, fuentes)
    guardar_en_mongo(pregunta, razonamiento, respuesta, fuentes)

# 🏁 Bucle principal
if __name__ == "__main__":
    print("🧪 Knowledge Navigator – Consulta enriquecida con razonamiento + memoria + MongoDB\n")
    while True:
        pregunta = input("🔎 Introduce tu pregunta (o 'salir'): ")
        if pregunta.lower() == "salir":
            break
        responder(pregunta)
