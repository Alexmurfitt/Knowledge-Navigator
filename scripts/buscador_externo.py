# =============================================================
# buscador_externo.py — Versión final con robustez, limpieza y compatibilidad total
# =============================================================

import os
import unicodedata
from dotenv import load_dotenv

# Cargar variables de entorno si no están ya cargadas
if not os.getenv("GOOGLE_API_KEY"):
    load_dotenv()

# Importar wrapper desde distintas ubicaciones posibles
try:
    from langchain_google_community import GoogleSearchAPIWrapper
except ImportError:
    try:
        from langchain.utilities import GoogleSearchAPIWrapper
    except ImportError:
        GoogleSearchAPIWrapper = None

# Instancia del buscador (si está disponible)
search = GoogleSearchAPIWrapper() if GoogleSearchAPIWrapper else None

# =============================================================
# FUNCIÓN PRINCIPAL: búsqueda web controlada
# =============================================================
def buscar_web(query: str, k: int = 5):
    """
    Realiza una búsqueda web usando Google Custom Search API (via LangChain wrapper).
    
    Parámetros:
        - query: pregunta o tema a buscar.
        - k: número máximo de resultados deseados.

    Retorna:
        - snippets: lista de fragmentos limpios.
        - titulo: título del primer resultado.
        - url: enlace del primer resultado.
    """
    if not search:
        return [], "", "⚠️ Error: Wrapper de búsqueda de Google no disponible."

    try:
        resultados = search.results(query, k)
        snippets = [limpiar_snippet(res.get('snippet', '')) for res in resultados if res.get('snippet')]
        titulo = resultados[0].get('title', '') if resultados else ''
        url = resultados[0].get('link', '') if resultados else ''
        return snippets, titulo, url
    except Exception as e:
        return [], "", f"⚠️ Error al realizar búsqueda web: {e}"

# =============================================================
# FUNCIÓN AUXILIAR: limpieza robusta
# =============================================================
def limpiar_snippet(texto: str) -> str:
    """
    Normaliza y limpia texto proveniente de snippets web.
    """
    texto = unicodedata.normalize("NFKD", texto)
    texto = texto.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    return texto.replace("\n", " ").replace("\r", "").strip()
