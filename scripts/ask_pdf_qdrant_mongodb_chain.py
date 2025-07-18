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
from razonador_chain import razonador_chain  # Ya adaptado a .invoke()

# 🧪 Cargar entorno
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")

if MONGO_DB_NAME is None:
    raise ValueError("❌ Error: MONGO_DB_NAME no está definido. Verifica tu archivo .env")

# 🔐 MongoDB
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client[MONGO_DB_NAME]
mongo_collection = mongo_db["historial"]

# 💬 Modelo y embeddings
llm = ChatOllama(model="llama3", temperature=0.1)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 🔍 Qdrant
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
qdrant = QdrantVectorStore(client=client, collection_name="knowledge_navigator", embedding=embeddings)
retriever = qdrant.as_retriever()

# 🧠 Memoria conversacional
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# 🔗 Cadena RAG
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    output_key="answer"
)

# 📁 Guardado en JSON + Mongo
def guardar(pregunta, pregunta_refinada, respuesta, adicionales, fuentes):
    entrada = {
        "pregunta": pregunta,
        "pregunta_refinada": pregunta_refinada,
        "respuesta_rag": respuesta,
        "informacion_adicional": adicionales,
        "timestamp": datetime.datetime.now().isoformat(),
        "contexto": [doc.page_content for doc in fuentes],
        "fuentes": [
            {
                "documento": doc.metadata.get("source", "desconocido"),
                "pagina": doc.metadata.get("page", "N/A")
            }
            for doc in fuentes
        ]
    }

    with open("historial.json", "r+", encoding="utf-8") as f:
        try:
            historial = json.load(f)
        except:
            historial = []

        historial.append(entrada)
        f.seek(0)
        json.dump(historial, f, ensure_ascii=False, indent=2)

    mongo_collection.insert_one(entrada)

# 🤖 Función principal
def responder(pregunta):
    # Paso 1: Ejecutar razonador → pregunta refinada + respuestas adicionales
    razonamiento = razonador_chain.invoke({"pregunta_usuario": pregunta})
    pregunta_refinada = razonamiento.get("pregunta_refinada", pregunta)
    adicionales = razonamiento.get("respuestas_adicionales", [])

    # Paso 2: Ejecutar RAG
    resultado = qa_chain.invoke({"question": pregunta_refinada})
    respuesta = resultado["answer"]
    fuentes = resultado.get("source_documents", [])

    # Mostrar solo lo necesario
    print("\n📘 Respuesta:")
    print(respuesta)

    if adicionales:
        print("\n🔸 INFORMACIÓN ADICIONAL:")
        for r in adicionales:
            print(f"• {r}")

    if fuentes:
        print("\n📚 Documentos fuente:")
        for i, doc in enumerate(fuentes):
            nombre = doc.metadata.get("source", "desconocido")
            pagina = doc.metadata.get("page", "N/A")
            print(f"  {i+1}. {nombre} (p. {pagina})")

    # Guardar historial completo
    guardar(pregunta, pregunta_refinada, respuesta, adicionales, fuentes)

# 🏁 Entrada por consola
if __name__ == "__main__":
    print("🧪 Knowledge Navigator – Consulta con razonamiento + memoria + MongoDB\n")
    while True:
        pregunta = input("🔎 Introduce tu pregunta (o 'salir'): ")
        if pregunta.lower() == "salir":
            break
        responder(pregunta)
