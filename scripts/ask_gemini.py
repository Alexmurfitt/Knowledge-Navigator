# ask_gemini_definitivo.py — Versión fusionada final
import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from buscador_externo import buscar_web
from razonador_cot import generar_respuesta, generar_respuesta_web

# Configuración
load_dotenv()
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

HISTORIAL_PATH = "historial.json"

_modelo_llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai", temperature=0)
_embeddings = OllamaEmbeddings(model="mxbai-embed-large")
_embedding_memoria = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
_memoria = ConversationBufferMemory(return_messages=True)

client_qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
_vectorstore = QdrantVectorStore(client=client_qdrant, collection_name="knowledge-navigator", embedding=_embeddings)

# Utilidades

def cargar_historial_json():
    if os.path.exists(HISTORIAL_PATH):
        try:
            with open(HISTORIAL_PATH, "r", encoding="utf-8") as f:
                interacciones = json.load(f)
            for item in interacciones:
                _memoria.save_context({"input": item["pregunta"]}, {"output": item["respuesta"]})
        except Exception as e:
            print(f"⚠️ Error al cargar historial: {e}")

def guardar_en_historial(pregunta, respuesta, fuentes, uso_externo):
    entrada = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pregunta": pregunta,
        "respuesta": respuesta,
        "fuentes": fuentes,
        "uso_externo": uso_externo
    }
    try:
        historial = []
        if os.path.exists(HISTORIAL_PATH):
            with open(HISTORIAL_PATH, "r", encoding="utf-8") as f:
                historial = json.load(f)
        historial.append(entrada)
        with open(HISTORIAL_PATH, "w", encoding="utf-8") as f:
            json.dump(historial, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Error al guardar historial: {e}")

def es_pregunta_simple(pregunta):
    clave = ["qué", "cómo", "cuándo", "dónde", "quién", "hola"]
    return len(pregunta.split()) < 8 and any(c in pregunta.lower() for c in clave)

def es_saludo_o_interaccion_social(pregunta):
    saludos = ["hola", "qué tal", "como estas", "buenos días", "buenas tardes", "hey", "saludos"]
    return any(pregunta.lower().startswith(s) for s in saludos)

def recuperar_respuesta_similar(pregunta, umbral=0.75):
    mensajes = _memoria.chat_memory.messages
    pares = [(m.content, mensajes[i+1].content) for i, m in enumerate(mensajes[:-1]) if m.type == "human" and mensajes[i+1].type == "ai"]
    if not pares:
        return None
    preguntas, respuestas = zip(*pares)
    vectores = _embedding_memoria.embed_documents(list(preguntas))
    vector_actual = _embedding_memoria.embed_query(pregunta)
    score = cosine_similarity([vector_actual], vectores)[0]
    idx = int(np.argmax(score))
    if score[idx] >= umbral:
        return preguntas[idx], respuestas[idx]
    return None

# Consulta principal

def consultar_gemini(pregunta: str):
    if es_saludo_o_interaccion_social(pregunta):
        respuesta = "¡Hola! ¿Cómo estás? Estoy aquí para ayudarte a explorar tus documentos o buscar información útil. ¿En qué puedo ayudarte hoy?"
        _memoria.save_context({"input": pregunta}, {"output": respuesta})
        guardar_en_historial(pregunta, respuesta, ["Asistente conversacional"], False)
        return respuesta, "", ["Asistente conversacional"], False

    recuperada = recuperar_respuesta_similar(pregunta)
    if recuperada:
        pregunta_antigua, respuesta_antigua = recuperada
        _memoria.save_context({"input": pregunta}, {"output": respuesta_antigua})
        guardar_en_historial(pregunta, respuesta_antigua, ["Memoria conversacional"], False)
        return respuesta_antigua, f"Respuesta recuperada por similitud: \"{pregunta_antigua}\"", ["Memoria conversacional"], False

    if es_pregunta_simple(pregunta):
        respuesta = _modelo_llm.invoke(pregunta).content
        _memoria.save_context({"input": pregunta}, {"output": respuesta})
        guardar_en_historial(pregunta, respuesta, ["Modelo de lenguaje"], False)
        return respuesta, "", ["Modelo de lenguaje"], False

    try:
        resultados = _vectorstore.similarity_search_with_relevance_scores(pregunta, k=5)
        documentos = [doc for doc, score in resultados if doc.page_content]
    except Exception as e:
        print(f"❌ Error al buscar en Qdrant: {e}")
        documentos = []

    contexto = "\n\n".join(doc.page_content for doc in documentos)
    usar_contexto = bool(contexto.strip())

    if usar_contexto:
        historial = "\n".join([f"Usuario: {m.content}" if m.type == "human" else f"Asistente: {m.content}" for m in _memoria.chat_memory.messages])
        respuesta = generar_respuesta(pregunta, contexto, historial)
        fuentes = [f'Documento: {doc.metadata.get("source", "?")} – Página {doc.metadata.get("page", "?")}' for doc in documentos]
        uso_externo = False
    else:
        snippets, titulo, link = buscar_web(pregunta, k=5)
        contexto_web = "\n\n".join(snippets)
        respuesta = generar_respuesta_web(pregunta, contexto_web)
        fuentes = [f"{titulo} ({link})"] if titulo or link else ["Web externa"]
        uso_externo = True

    partes = respuesta.split("Explicación ampliada:")
    respuesta_principal = partes[0].split(":", 1)[1].strip() if "Definición" in partes[0] else partes[0].strip()
    razonamiento = partes[1].strip() if len(partes) > 1 else ""

    _memoria.save_context({"input": pregunta}, {"output": respuesta_principal})
    guardar_en_historial(pregunta, respuesta_principal, fuentes, uso_externo)

    return respuesta_principal, razonamiento, fuentes, uso_externo

# CLI
if __name__ == "__main__":
    cargar_historial_json()
    pregunta = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Pregunta: ")
    respuesta, razonamiento, fuentes, externo = consultar_gemini(pregunta)

    print("\n📘 Respuesta:")
    print(respuesta or "(No se obtuvo respuesta)")
    print("\n💡 Información adicional:")
    print(razonamiento or "(No se proporcionó explicación adicional)")
    print("\n📚 Fuentes consultadas:")
    for f in fuentes:
        print(f"- {f}")
    print("\n🔍 Nota:", "Se utilizó búsqueda web externa." if externo else "Se respondió con documentos internos.")