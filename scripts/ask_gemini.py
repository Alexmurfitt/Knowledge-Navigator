
import os
import json
import datetime
import re
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings

# 🔹 Módulos útiles
from razonador_cot import razonamiento_cot
from buscador_externo import buscar_google

# 🔐 Cargar configuración
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# 🧠 Embeddings y recuperación
embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
qdrant = QdrantVectorStore(
    client=QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY),
    collection_name="Knowledge-Navigator",
    embedding=embeddings
)
retriever = qdrant.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 10, "fetch_k": 20}
)

# 🤖 LLM principal (Gemini)
llm = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    api_key=google_api_key,
    temperature=0
)

# 💬 Memoria (solo en ejecución actual)
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="output"
)

# 🔗 Cadena de recuperación contextual
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    output_key="output"
)

# 🔧 Limpieza de texto
def limpiar_texto(texto):
    try:
        if texto is None:
            return ""
        texto = str(texto)
        texto = texto.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        return ''.join(c for c in texto if c.isprintable()).strip()
    except:
        return "[❌ Error al limpiar texto]"

# 💾 Guardar historial local
def guardar_local(pregunta, respuesta, razonamiento_cot_texto, fuentes):
    entrada = {
        "pregunta": limpiar_texto(pregunta),
        "respuesta": limpiar_texto(respuesta),
        "razonamiento_chain_of_thought": limpiar_texto(razonamiento_cot_texto),
        "timestamp": datetime.datetime.now().isoformat(),
        "contexto": [limpiar_texto(doc.page_content) for doc in fuentes],
        "fuentes": [
            {
                "documento": limpiar_texto(doc.metadata.get("source", "desconocido")),
                "pagina": limpiar_texto(str(doc.metadata.get("page", "N/A")))
            }
            for doc in fuentes
        ]
    }

    if not os.path.exists("historial.json"):
        with open("historial.json", "w", encoding="utf-8") as f:
            json.dump([], f)

    try:
        with open("historial.json", "r+", encoding="utf-8", errors="replace") as f:
            historial = json.load(f)
            historial.append(entrada)
            f.seek(0)
            json.dump(historial, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Error guardando historial: {e}")

# 🚀 Función principal
def responder(pregunta):
    try:
        try:
            razonamiento_cot_texto = razonamiento_cot(pregunta)
            adicionales = []  # ← para evitar errores posteriores

        except Exception as e:
            print("\n❌ Error en razonamiento_cot:")
            print(e)
            return f"❌ Error en razonador: {str(e)}", [], "", []

        docs = retriever.get_relevant_documents(pregunta)
        contexto_local = "\n".join([doc.page_content for doc in docs])

        if not contexto_local.strip():
            print("🌐 No se encontró contexto local. Buscando en la web...")
            snippets = buscar_google(pregunta)
            contexto_web = "\n".join(snippets)
        else:
            contexto_web = ""

        contexto_final = contexto_local + "\n" + contexto_web

        prompt = f"""Usa la siguiente información para responder con claridad, precisión y objetividad.
Pregunta: {pregunta}
Contexto:
{contexto_final}
Tu respuesta debe ser clara, fiable y aportar datos de alto valor si es posible."""

        respuesta_llm = llm.invoke(prompt)
        respuesta = limpiar_texto(respuesta_llm.content.strip())

        print("\n📘 RESPUESTA:")
        print(respuesta)
        print("\n📚 DOCUMENTOS CONSULTADOS:")
        for i, doc in enumerate(docs):
            print(f"  {i+1}. {doc.metadata.get('source', 'desconocido')} (p. {doc.metadata.get('page', 'N/A')})")

        guardar_local(pregunta, respuesta, razonamiento_cot_texto, docs)
        return respuesta, docs, razonamiento_cot_texto, adicionales

    except Exception as e:
        print("\n❌ Error durante la consulta:")
        print(e)
        return f"❌ Error: {str(e)}", [], "", []

# 🔁 Modo consola
if __name__ == "__main__":
    while True:
        pregunta = input("Pregunta: ")
        respuesta, docs, razonamiento, adicionales = responder(pregunta)
        print("\n📘 RESPUESTA:")
        print(respuesta)

        if adicionales:
            print("\n🔸 INFORMACIÓN ADICIONAL:")
            for frase in adicionales:
                print(f"- {frase}")

        print("\n🧠 RAZONAMIENTO COMPLETO:")
        print(razonamiento)

        if docs:
            print("\n📚 DOCUMENTOS CONSULTADOS:")
            for i, doc in enumerate(docs):
                print(f"{i+1}. {doc.metadata.get('source', 'desconocido')} (p. {doc.metadata.get('page', 'N/A')})")
