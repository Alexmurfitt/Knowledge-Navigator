from langchain_google_community import GoogleSearchAPIWrapper
import os
from dotenv import load_dotenv

load_dotenv()

def limpiar_snippet(snippet: str) -> str:
    try:
        return snippet.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    except Exception:
        return "[❌ Error al limpiar resultado de Google]"

def buscar_google(query: str, k: int = 3):
    search = GoogleSearchAPIWrapper(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        google_cse_id=os.getenv("GOOGLE_CSE_ID")
    )
    try:
        resultados = search.results(query, num_results=k)
    except Exception as e:
        return [f"❌ Error en búsqueda externa: {e}"]

    return [limpiar_snippet(r.get("snippet", "")) for r in resultados]
