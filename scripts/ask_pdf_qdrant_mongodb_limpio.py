import os
import json
import datetime
import re
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Nuevo import para evitar deprecación en historial (requiere `pip install langchain-mongodb`)
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory

# ✅ Razonadores personalizados
from razonador_chain import razonador_chain
from razonador_cot_limpio import razonamiento_cot

# 🧪 Cargar entorno
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")

if not MONGO_DB_NAME:
    raise ValueError("❌ MONGO_DB_NAME no está definido. Revisa el archivo .env")

# 🔐 Conexión MongoDB
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client[MONGO_DB_NAME]
mongo_collection = mongo_db["historial"]

# 💬 LLM + Embeddings
llm = ChatOllama(model="llama3", temperature=0.1)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 🔍 Vector store Qdrant
qdrant = QdrantVectorStore(
    client=QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY),
    collection_name="knowledge_navigator",
    embedding=embeddings
)
retriever = qdrant.as_retriever()

# 🧠 Memoria conversacional
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="output"  # ← obligatorio para evitar KeyError
)

# 🔗 Cadena RAG
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    output_key="output"  # ← clave que devolverá el resultado
)

# 🧹 Utilidad para extraer JSON válido desde texto
def extraer_json_de_texto(salida_cruda):
    if isinstance(salida_cruda, dict):
        return salida_cruda
    match = re.search(r"\{[\s\S]*?\}", str(salida_cruda))
    if not match:
        raise ValueError("❌ No se encontró JSON válido en la salida.")
    bloque = re.sub(r'(?<!")\b([a-zA-Z_][a-zA-Z0-9_]*)\b(?=\s*:)', r'"\1"', match.group())
    return json.loads(bloque)

# 💾 Guardar trazabilidad
def guardar(pregunta, pregunta_refinada, respuesta, adicionales, razonamiento_cot_texto, fuentes):
    entrada = {
        "pregunta": pregunta,
        "pregunta_refinada": pregunta_refinada,
        "respuesta_rag": respuesta,
        "respuestas_adicionales": adicionales,
        "razonamiento_chain_of_thought": razonamiento_cot_texto,
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
        except json.JSONDecodeError:
            historial = []
        historial.append(entrada)
        f.seek(0)
        json.dump(historial, f, ensure_ascii=False, indent=2)

    mongo_collection.insert_one(entrada)

# 🤖 Función principal
def responder(pregunta):
    try:
        # Paso 1: razonamiento estructurado (puede venir con texto extra)
        salida_cruda = razonador_chain.invoke({"pregunta_usuario": pregunta})
        razonamiento = extraer_json_de_texto(salida_cruda)
        pregunta_refinada = razonamiento.get("pregunta_refinada", pregunta)
        adicionales = razonamiento.get("respuestas_adicionales", [])

        # Paso 2: razonamiento tipo Chain of Thought
        razonamiento_cot_texto = razonamiento_cot(pregunta)

        # Paso 3: ejecución de la cadena RAG
        resultado = qa_chain.invoke({"question": pregunta_refinada})
        respuesta = resultado["output"]
        fuentes = resultado.get("source_documents", [])

        # ✅ Mostrar resultados
        print("\n📘 Respuesta:")
        print(respuesta)

        if adicionales:
            print("\n🔸 INFORMACIÓN ADICIONAL:")
            for r in adicionales:
                print(f"• {r}")

        if fuentes:
            print("\n📚 Documentos fuente:")
            for i, doc in enumerate(fuentes):
                print(f"  {i+1}. {doc.metadata.get('source', 'desconocido')} (p. {doc.metadata.get('page', 'N/A')})")

        # 🧾 Guardar
        guardar(pregunta, pregunta_refinada, respuesta, adicionales, razonamiento_cot_texto, fuentes)

    except Exception as e:
        print("\n⚠️ Error durante el procesamiento de la consulta:")
        print(e)

# 🏁 Interfaz
if __name__ == "__main__":
    print("🧪 Knowledge Navigator – Consulta híbrida con razonamiento (Chain + CoT), memoria y MongoDB\n")
    while True:
        pregunta = input("🔎 Introduce tu pregunta (o 'salir'): ")
        if pregunta.strip().lower() == "salir":
            break
        responder(pregunta)
