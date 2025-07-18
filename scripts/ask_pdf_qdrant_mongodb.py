import os
import json
import datetime
import re
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# ✅ Dos razonadores modernos
from razonador_chain import razonador_chain
from razonador_cot import razonamiento_cot

# 🧪 Cargar variables de entorno
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")

if not MONGO_DB_NAME:
    raise ValueError("❌ MONGO_DB_NAME no está definido. Verifica tu archivo .env")

# 🔐 MongoDB
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client[MONGO_DB_NAME]
mongo_collection = mongo_db["historial"]

# 🧠 Modelos
llm = ChatOllama(model="llama3", temperature=0.1)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 🔍 Qdrant vector store
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
qdrant = QdrantVectorStore(client=client, collection_name="knowledge_navigator", embedding=embeddings)
retriever = qdrant.as_retriever()

# 🧠 Memoria conversacional moderna
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

# 💾 Guardado
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

# 🔍 Utilidad para extraer JSON de una salida que incluye texto adicional
# 🔍 Utilidad robusta para extraer y reparar JSON generado por el LLM
def extraer_json_de_texto(salida_cruda):
    import re
    import json

    if isinstance(salida_cruda, dict):
        return salida_cruda  # Ya es un dict válido

    # Buscar bloque de JSON con llaves (incluye saltos de línea y texto extra)
    match = re.search(r"\{[\s\S]*?\}", str(salida_cruda))
    if not match:
        raise ValueError("❌ No se encontró ningún bloque con formato JSON en la salida del razonador_chain.")

    bloque = match.group()

    # Reparar claves sin comillas dobles (clave: → "clave":)
    bloque = re.sub(r'(?<!")\b([a-zA-Z_][a-zA-Z0-9_]*)\b(?=\s*:)', r'"\1"', bloque)

    try:
        return json.loads(bloque)
    except json.JSONDecodeError as e:
        print("\n❌ Error al parsear JSON reparado:")
        print(bloque)
        raise e



# 🤖 Función principal
def responder(pregunta):
    try:
        # Paso 1: razonador estructurado moderno (puede incluir texto fuera del JSON)
        salida_cruda = razonador_chain.invoke({"pregunta_usuario": pregunta})
        razonamiento = extraer_json_de_texto(salida_cruda)
        pregunta_refinada = razonamiento.get("pregunta_refinada", pregunta)
        adicionales = razonamiento.get("respuestas_adicionales", [])

        # Paso 2: razonador CoT para trazabilidad
        razonamiento_cot_texto = razonamiento_cot(pregunta)

        # Paso 3: ejecutar RAG
        resultado = qa_chain.invoke({"question": pregunta_refinada})
        respuesta = resultado["answer"]
        fuentes = resultado.get("source_documents", [])

        # Mostrar en terminal
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

        # Guardar todo
        guardar(pregunta, pregunta_refinada, respuesta, adicionales, razonamiento_cot_texto, fuentes)

    except Exception as e:
        print("\n⚠️ Error durante el procesamiento de la consulta:")
        print(e)

# 🏁 Interfaz por consola
if __name__ == "__main__":
    print("🧪 Knowledge Navigator – Consulta híbrida con razonamiento (Chain + CoT), memoria y MongoDB\n")
    while True:
        pregunta = input("🔎 Introduce tu pregunta (o 'salir'): ")
        if pregunta.lower().strip() == "salir":
            break
        responder(pregunta)
