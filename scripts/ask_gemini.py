import os
import json
import datetime
from dotenv import load_dotenv

# Cargar variables de entorno (API keys, URLs, etc.) una sola vez
load_dotenv()

# Importar modelo de incrustaciones de Ollama y Qdrant vector store
try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    OllamaEmbeddings = None
try:
    from langchain.vectorstores import QdrantVectorStore
except ImportError:
    try:
        from langchain_qdrant import QdrantVectorStore
    except ImportError:
        QdrantVectorStore = None

from razonador_cot import generar_respuesta
from buscador_externo import buscar_web
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

# Clasificador semántico de intención (¿es una pregunta de actualidad o requiere concisión?)
def clasificar_intencion(pregunta: str) -> tuple[bool, bool]:
    """
    Clasifica si la pregunta necesita búsqueda externa y si requiere una respuesta concisa.
    Devuelve dos booleanos: (necesita_externo, respuesta_concisa)
    """
    prompt = ChatPromptTemplate.from_template("""
Eres un clasificador semántico. Analiza la siguiente pregunta y responde en formato JSON.

Tu tarea es determinar:
- "necesita_externo": True si la pregunta requiere información actualizada o reciente que probablemente no esté en los documentos internos.
- "respuesta_concisa": True si la respuesta ideal debe ser breve, precisa y directa, sin explicación larga.

Ejemplos:

Pregunta: "¿Cuáles son las últimas novedades sobre IA responsable?"
Respuesta: {{ "{{\"necesita_externo\": true, \"respuesta_concisa\": false}}" }}

Pregunta: "¿Qué avances recientes hay en la regulación de la IA en Europa?"
Respuesta: {{ "{{\"necesita_externo\": true, \"respuesta_concisa\": false}}" }}

Pregunta: "¿Qué día es hoy?"
Respuesta: {{ "{{\"necesita_externo\": true, \"respuesta_concisa\": true}}" }}

Pregunta: "¿Qué significa el principio de transparencia algorítmica?"
Respuesta: {{ "{{\"necesita_externo\": false, \"respuesta_concisa\": false}}" }}

Pregunta: "{pregunta}"

Devuelve exactamente este formato:
{{ "{{\"necesita_externo\": true/false, \"respuesta_concisa\": true/false}}" }}
""")

    chain = prompt | init_chat_model(
        model="gemini-1.5-flash",
        model_provider="google_genai",
        temperature=0
    ) | RunnableLambda(lambda msg: msg)

    try:
        resultado = chain.invoke({"pregunta": pregunta})
        content = resultado.content.strip() if hasattr(resultado, "content") else str(resultado).strip()

        if not content:
            print("[⚠️ Clasificador vacío] El modelo no devolvió ningún contenido.")
            return False, False

        # ✅ LIMPIEZA ROBUSTA DEL BLOQUE JSON
        if content.startswith("```json") and content.endswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```") and content.endswith("```"):
            content = content.strip("`").strip()

        # ✅ CARGA SEGURA DE JSON
        parsed = json.loads(content)
        print(f"[✔️ Clasificador] Resultado: {parsed}")
        return parsed.get("necesita_externo", False), parsed.get("respuesta_concisa", False)

    except json.JSONDecodeError as e:
        print(f"[⚠️ JSON malformado] Error: {e}\nContenido recibido:\n{content}")
        return False, False

    except Exception as e:
        print(f"[❌ Error inesperado en clasificar_intencion]: {e}")
        return False, False


# Instancias globales para reutilizar (evita recargas repetidas)
_embeddings = None
_vectorstore = None

def consultar_gemini(pregunta: str):
    """
    Procesa la pregunta utilizando la base de conocimiento local y/o búsqueda web,
    y genera una respuesta formateada con la ayuda del modelo Gemini.
    Retorna una tupla: (respuesta_principal, explicacion_adicional, fuentes, uso_busqueda_externa).
    """
    global _embeddings, _vectorstore

    # Inicializar modelo de embeddings Ollama si no está ya disponible
    if OllamaEmbeddings is None:
        raise ImportError("OllamaEmbeddings no está disponible. Instale el paquete 'langchain-ollama'.")
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

    # Conectar (una vez) a Qdrant para buscar contexto relevante
    if _vectorstore is None and QdrantVectorStore:
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        try:
            _vectorstore = QdrantVectorStore.from_existing_collection(
                embedding=_embeddings,
                collection_name="knowledge-navigator",
                url=qdrant_url,
                api_key=qdrant_api_key if qdrant_api_key else None
            )
        except Exception:
            _vectorstore = None  # Si falla (p.ej. colección no existente), no usar vectorstore

    # Recuperar documentos similares si la base de conocimiento está disponible
    documentos = []
    if _vectorstore is not None:
        try:
            resultados_con_score = _vectorstore.similarity_search_with_relevance_scores(pregunta, k=5)
            documentos_filtrados = [
                doc for doc, score in resultados_con_score if score is not None and score >= 0.8
            ]
            documentos = documentos_filtrados
        except Exception:
            documentos = []
        # Determinar si se requiere búsqueda externa (consultas de actualidad/novedades)
    # Clasificación semántica real de la intención
    necesita_externo, respuesta_concisa = clasificar_intencion(pregunta)

    # Determinar la fuente de contexto a usar
    used_external = False
    contexto_texto = ""
    first_title = first_link = ""
    if documentos and len(documentos) > 0:
        # Combinar contenido de documentos internos relevantes
        contexto_texto = "\n\n".join([doc.page_content for doc in documentos if doc.page_content])
        used_external = False
    else:
        # Realizar búsqueda web externa si falta contexto interno o es consulta de actualidad
        used_external = True
        resultado_busqueda = buscar_web(pregunta, k=5)
        if isinstance(resultado_busqueda, str):
            contexto_texto = ""  # error o wrapper no disponible
        else:
            snippets, first_title, first_link = resultado_busqueda
            contexto_texto = "\n\n".join(snippets) if snippets else ""

    # Invocar el modelo Gemini (razonador_cot) con la pregunta y el contexto recopilado
    respuesta_completa = generar_respuesta(pregunta, contexto_texto, modo_conciso=respuesta_concisa)
    if respuesta_completa is None:
        respuesta_completa = ""
    # Limpiar la respuesta eliminando caracteres no imprimibles
    respuesta_completa = "".join(ch for ch in respuesta_completa if ch.isprintable() or ch.isspace())
    respuesta_completa = respuesta_completa.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')

    # Separar la respuesta en las dos secciones esperadas (Definición y Explicación)
    respuesta_principal = respuesta_completa.strip()
    explicacion_adicional = ""

    if not respuesta_concisa and "Explicación ampliada:" in respuesta_completa:
        try:
            partes = respuesta_completa.split("Explicación ampliada:")
            definicion_part = partes[0].strip()
            explicacion_part = partes[1].strip()
            if definicion_part.lower().startswith("definición textual"):
                definicion_content = definicion_part.split(":", 1)[1].strip() if ":" in definicion_part else definicion_part
            else:
                definicion_content = definicion_part
            respuesta_principal = definicion_content
            explicacion_adicional = explicacion_part
        except Exception as e:
            print(f"[⚠️ Error al descomponer respuesta estructurada]: {e}")
            respuesta_principal = respuesta_completa.strip()
            explicacion_adicional = ""
    else:
        respuesta_principal = respuesta_completa.strip()
        explicacion_adicional = ""

    # Preparar la lista de fuentes consultadas para registro y salida
    fuentes = []
    if used_external:
        # Se utilizó búsqueda externa
        if first_title or first_link:
            fuente_str = first_title if first_title else "Fuente externa"
            if first_link:
                fuente_str += f" ({first_link})"
            fuentes.append(fuente_str)
        else:
            fuentes.append("Búsqueda web realizada (sin resultados específicos)")
    else:
        # Solo se usó la base de conocimiento interna
        
        for doc in documentos:
            fuente = doc.metadata.get("source", "Documento desconocido")
            pagina = doc.metadata.get("page", "¿?")
            score = f"{getattr(doc, 'score', 0):.4f}" if hasattr(doc, 'score') else "N/A"
            fuentes.append(f"📁 {fuente} | 📄 Página {pagina} | 🔢 Score: {score}")

    # Guardar trazabilidad en historial.json (registro de la pregunta y respuesta)
    registro = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pregunta": pregunta,
        "respuesta": respuesta_principal,
        "razonamiento": explicacion_adicional,
        "fuentes": fuentes
    }
    try:
        historial = []
        if os.path.exists("historial.json"):
            with open("historial.json", "r", encoding="utf-8") as f:
                historial = json.load(f)
            if not isinstance(historial, list):
                historial = []
        historial.append(registro)
        with open("historial.json", "w", encoding="utf-8") as f:
            json.dump(historial, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Advertencia] No se pudo guardar historial: {e}")

    return respuesta_principal, explicacion_adicional, fuentes, used_external

# Punto de entrada para uso interactivo por consola
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        pregunta_usuario = " ".join(sys.argv[1:])
    else:
        pregunta_usuario = input("Ingrese su pregunta: ")
    resp_principal, resp_expanded, fuentes_consultadas, uso_externo = consultar_gemini(pregunta_usuario)
    # Mostrar resultados en consola
    print("\nRespuesta principal:")
    print(resp_principal if resp_principal else "(No se obtuvo respuesta)")
    print("\nInformación adicional (razonamiento):")
    print(resp_expanded if resp_expanded else "(No se proporcionó explicación adicional)")
    print("\nFuentes consultadas:")
    if fuentes_consultadas:
        for fuente in fuentes_consultadas:
            print(f"- {fuente}")
    else:
        print("(No se consultaron fuentes)")
    if uso_externo:
        print("\n*Nota:* Se utilizó búsqueda web externa para complementar la respuesta.")
