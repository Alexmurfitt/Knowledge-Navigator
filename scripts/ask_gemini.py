# =============================================================
# 1. IMPORTACIONES Y CONFIGURACIÓN INICIAL
# =============================================================
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Módulos propios
from razonador_cot import generar_respuesta, generar_respuesta_web
from buscador_externo import buscar_web

# Cargar variables de entorno
load_dotenv()
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# =============================================================
# 2. INSTANCIAS GLOBALES
# =============================================================
_embeddings = None
_vectorstore = None
_embedding_memoria = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
_modelo_evaluador = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai", temperature=0)
_memoria = ConversationBufferMemory(return_messages=True)

# =============================================================
# 3. HISTORIAL CONVERSACIONAL Y CLASIFICACIÓN SEMÁNTICA
# =============================================================
def construir_historial(memoria) -> str:
    if not memoria.buffer:
        return ""
    historial = ""
    for m in memoria.buffer:
        rol = "Usuario" if m.type == "human" else "Asistente"
        historial += f"{rol}: {m.content}\n"
    return historial.strip()

def clasificar_referencia_memoria(pregunta: str, historial: str) -> bool:
    prompt = ChatPromptTemplate.from_template("""
Eres un clasificador semántico. Analiza la siguiente pregunta en función del historial conversacional
y responde si está haciendo referencia a algo que el usuario dijo o preguntó previamente.

Historial:
{historial}

Pregunta:
{pregunta}

Responde solo con "Sí" si la pregunta depende de una interacción anterior, o "No" si es completamente nueva.
""")
    try:
        chain = prompt | _modelo_evaluador
        salida = chain.invoke({"pregunta": pregunta, "historial": historial})
        decision = salida.content.strip().lower()
        print(f"🧐 Clasificador semántico memoria: {decision}")
        return decision.startswith("sí") or decision.startswith("yes")
    except Exception as e:
        print("⚠️ Error al clasificar referencia a memoria:", e)
        return False

def recuperar_respuesta_previa(pregunta_actual: str, memoria=_memoria, umbral: float = 0.75):
    mensajes = memoria.chat_memory.messages
    pares = [(m.content, mensajes[i + 1].content) for i, m in enumerate(mensajes[:-1]) if m.type == "human" and mensajes[i + 1].type == "ai"]
    if not pares:
        return None
    preguntas, respuestas = zip(*pares)
    vectores = _embedding_memoria.embed_documents(list(preguntas))
    vector_actual = _embedding_memoria.embed_query(pregunta_actual)
    similitudes = cosine_similarity([vector_actual], vectores)[0]
    idx_mas_cercano = int(np.argmax(similitudes))
    score = similitudes[idx_mas_cercano]
    if score >= umbral:
        return preguntas[idx_mas_cercano], respuestas[idx_mas_cercano]
    return None

# =============================================================
# 4. EVALUADOR DE CONTEXTO LOCAL
# =============================================================
def evaluar_utilidad_contexto_llm(contexto: str, pregunta: str, historial: str) -> bool:
    contexto_corto = contexto.strip()[:2000]
    prompt = ChatPromptTemplate.from_template("""
Eres un evaluador experto de información. Recibirás:

- Un historial de la conversación
- Una nueva pregunta
- Un contexto extraído de documentos PDF vectorizados

Debes decidir si ese contexto permite responder a la pregunta con:
- Exactitud
- Claridad
- Fundamento verificable

Historial previo:
{historial}

Pregunta:
{pregunta}

Contexto:
{contexto}

¿El contexto permite responder correctamente, sin necesidad de buscar información adicional?

Responde SOLO con "Sí" o "No".
""")
    try:
        chain = prompt | _modelo_evaluador
        salida = chain.invoke({"pregunta": pregunta, "contexto": contexto_corto, "historial": historial})
        decision = salida.content.strip().lower()
        print(f"🔍 Evaluador respondió: {decision}")
        return decision.startswith("sí") or decision.startswith("yes")
    except Exception as e:
        print("⚠️ Error al evaluar contexto con LLM:", e)
        return False

# =============================================================
# 5. CONSULTA INTELIGENTE: MEMORIA + QDRANT + WEB
# =============================================================
def consultar_gemini(pregunta: str, historial_externo: str = ""):
    global _embeddings, _vectorstore

    if _embeddings is None:
        _embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    if _vectorstore is None:
        client_qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
        _vectorstore = QdrantVectorStore(
            client=client_qdrant,
            collection_name="knowledge-navigator",
            embedding=_embeddings
        )

    historial = historial_externo if historial_externo else construir_historial(_memoria)

    if clasificar_referencia_memoria(pregunta, historial):
        resultado = recuperar_respuesta_previa(pregunta)
        if resultado:
            pregunta_antigua, respuesta_antigua = resultado
            razonamiento = f"Recuperada de una interacción anterior.\n\n🗣️ Pregunta original: «{pregunta_antigua}»"
            _memoria.chat_memory.add_user_message(pregunta)
            _memoria.chat_memory.add_ai_message(respuesta_antigua)
            return respuesta_antigua, razonamiento, ["Memoria conversacional"], False

    documentos = []
    try:
        resultados = _vectorstore.similarity_search_with_relevance_scores(pregunta, k=5)
        resultados.sort(key=lambda x: x[1] or 0, reverse=True)
        for doc, score in resultados:
            if doc.page_content:
                documentos.append(doc)
    except Exception as e:
        print(f"❌ Error en búsqueda local: {e}")

    contexto_local = "\n\n".join(doc.page_content for doc in documentos) if documentos else ""
    usar_contexto_local = False
    usar_buscador_externo = False
    contexto_final = ""
    fuentes = []

    if contexto_local.strip():
        utilidad = evaluar_utilidad_contexto_llm(contexto_local, pregunta, historial)
        if utilidad:
            usar_contexto_local = True
            contexto_final = contexto_local
        else:
            usar_buscador_externo = True
    else:
        usar_buscador_externo = True

    if usar_buscador_externo:
        resultados_web = buscar_web(pregunta, k=5)
        if isinstance(resultados_web, tuple):
            snippets, titulo, link = resultados_web
            contexto_final = "\n\n".join(snippets) if isinstance(snippets, list) else ""
            if titulo or link:
                fuentes.append(f"{titulo} ({link})")
            else:
                fuentes.append("Búsqueda web realizada (sin resultados específicos)")
        else:
            print(f"⚠️ Error en búsqueda web: {resultados_web}")

    if not contexto_final.strip():
        return (
            "No se ha encontrado información suficientemente precisa para responder a esta pregunta.",
            "",
            ["Sin fuentes disponibles"],
            usar_buscador_externo
        )

    if usar_buscador_externo:
        respuesta_completa = generar_respuesta_web(pregunta, contexto_final) or ""
        respuesta_completa = respuesta_completa.replace("📘 Respuesta", "").replace("💡 Información adicional", "")
    else:
        respuesta_completa = generar_respuesta(pregunta, contexto_final, historial) or ""

    respuesta_principal = respuesta_completa.strip()
    razonamiento = ""
    if "Explicación ampliada:" in respuesta_completa:
        partes = respuesta_completa.split("Explicación ampliada:")
        respuesta_principal = partes[0].split(":", 1)[1].strip() if "definición textual" in partes[0].lower() else partes[0].strip()
        razonamiento = partes[1].strip()

    if usar_contexto_local:
        for doc in documentos:
            fuente = doc.metadata.get("source", "Documento local")
            pag = doc.metadata.get("page", "?")
            fuentes.append(f'Documento: "{fuente}" – Página {pag}')

    _memoria.chat_memory.add_user_message(pregunta)
    _memoria.chat_memory.add_ai_message(respuesta_principal)

    return respuesta_principal, razonamiento, fuentes, usar_buscador_externo

# =============================================================
# 6. CLI: Ejecutar por consola
# =============================================================
if __name__ == "__main__":
    pregunta = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Ingrese su pregunta: ")

    respuesta, razonamiento, fuentes, uso_externo = consultar_gemini(pregunta)

    print("\n📘 Respuesta:")
    print(respuesta or "(No se obtuvo respuesta)")

    print("\n💡 Información adicional:")
    print(razonamiento or "(No se proporcionó explicación adicional)")

    print("\n📚 Fuentes consultadas:")
    for fuente in fuentes:
        print(f"- {fuente}")

    print("\n🔍 Nota:", end=" ")
    if uso_externo and fuentes and any(f.startswith("Documento") for f in fuentes):
        print("Se utilizó información combinada: documentos PDF + búsqueda web.")
    elif uso_externo:
        print("Se utilizó búsqueda web externa para complementar la respuesta.")
    else:
        print("Se respondió exclusivamente con información de los documentos PDF.")
