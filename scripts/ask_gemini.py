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

# Módulos propios
from razonador_cot import generar_respuesta
from buscador_externo import buscar_web

# Cargar variables de entorno
load_dotenv()
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# =============================================================
# 2. INSTANCIAS GLOBALES
# =============================================================
_embeddings = None
_vectorstore = None
_modelo_evaluador = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai", temperature=0)

# =============================================================
# 3. AGENTE EVALUADOR DE CONTEXTO LOCAL
# =============================================================
def evaluar_utilidad_contexto_llm(contexto: str, pregunta: str) -> bool:
    """
    Evalúa si el contexto recuperado de los documentos PDF es útil, suficiente y preciso
    para responder correctamente a la pregunta sin necesidad de buscar en internet.
    """
    contexto_corto = contexto.strip()[:2000]  # limitar a 2000 caracteres

    prompt = ChatPromptTemplate.from_template("""
Eres un evaluador experto de calidad de información. Recibirás una *pregunta* y un *contexto extraído de documentos PDF vectorizados*.

Debes determinar con máxima objetividad si ese contexto permite responder con:
- Exactitud
- Claridad
- Fundamento verificable

Pregunta:
{pregunta}

Contexto:
{contexto}

¿El contexto proporciona información suficientemente precisa y aplicable para responder la pregunta, sin necesidad de buscar más información?

Responde SOLO con "Sí" o "No", sin explicaciones.
""")

    try:
        chain = prompt | _modelo_evaluador
        salida = chain.invoke({"pregunta": pregunta, "contexto": contexto_corto})
        decision = salida.content.strip().lower()
        print(f"🔍 Evaluador respondió: {decision}")
        return decision.startswith("sí") or decision.startswith("yes")
    except Exception as e:
        print("⚠️ Error al evaluar contexto con LLM:", e)
        return False

# =============================================================
# 4. CONSULTA INTELIGENTE CON GEMINI + QDRANT + BÚSQUEDA WEB
# =============================================================
def consultar_gemini(pregunta: str):
    global _embeddings, _vectorstore

    if _embeddings is None:
        _embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    if _vectorstore is None:
        client_qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
        _vectorstore = QdrantVectorStore(client=client_qdrant, collection_name="knowledge-navigator", embedding=_embeddings)

    # 1. Recuperar contexto desde Qdrant
    documentos = []
    try:
        resultados = _vectorstore.similarity_search_with_relevance_scores(pregunta, k=5)
        resultados.sort(key=lambda x: x[1] or 0, reverse=True)
        for doc, score in resultados:
            if doc.page_content:
                documentos.append(doc)
    except Exception as e:
        print(f"❌ Error en búsqueda local: {e}")
        documentos = []

    contexto_local = "\n\n".join(doc.page_content for doc in documentos) if documentos else ""
    usar_contexto_local = False
    usar_buscador_externo = False
    contexto_final = ""
    fuentes = []

    # 2. Evaluar utilidad del contexto local
    if contexto_local.strip():
        utilidad = evaluar_utilidad_contexto_llm(contexto_local, pregunta)
        if utilidad:
            contexto_final = contexto_local
            usar_contexto_local = True
        else:
            usar_buscador_externo = True
    else:
        usar_buscador_externo = True

    # 3. Buscar en la web si es necesario
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
            contexto_final = ""

    # 4. Verificación final del contexto
    if not contexto_final.strip():
        return (
            "No se ha encontrado información suficientemente precisa para responder a esta pregunta.",
            "",
            ["Sin fuentes disponibles"],
            usar_buscador_externo
        )

    # 5. Generar respuesta con razonador CoT
    respuesta_completa = generar_respuesta(pregunta, contexto_final) or ""
    respuesta_completa = "".join(c for c in respuesta_completa if c.isprintable() or c.isspace())
    respuesta_completa = respuesta_completa.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")

    # 6. Separar respuesta y razonamiento
    respuesta_principal = respuesta_completa.strip()
    razonamiento = ""
    if "Explicación ampliada:" in respuesta_completa:
        partes = respuesta_completa.split("Explicación ampliada:")
        respuesta_principal = partes[0].split(":", 1)[1].strip() if "definición textual" in partes[0].lower() else partes[0].strip()
        razonamiento = partes[1].strip()

    # 7. Fuentes locales si no hubo búsqueda web
    if not usar_buscador_externo:
        for doc in documentos:
            fuente = doc.metadata.get("source", "Documento local")
            pag = doc.metadata.get("page", "?")
            fuentes.append(f'Documento: "{fuente}" – Página {pag}')

    return respuesta_principal, razonamiento, fuentes, usar_buscador_externo

# =============================================================
# 5. PUNTO DE ENTRADA CLI
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
