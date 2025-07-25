import os
from dotenv import load_dotenv

# (Las variables de entorno ya se cargaron desde ask_gemini.py al inicializar la app)
# Cargar claves de API de Google desde .env solo si este módulo se usa independientemente
load_dotenv()

try:
    # Intentar importar el wrapper de búsqueda de Google (comunitario o utilitario)
    from langchain_google_community import GoogleSearchAPIWrapper
except ImportError:
    try:
        from langchain.utilities import GoogleSearchAPIWrapper
    except ImportError:
        GoogleSearchAPIWrapper = None

# Inicializar el cliente de búsqueda si el wrapper se importó correctamente
search = GoogleSearchAPIWrapper() if GoogleSearchAPIWrapper else None

def buscar_web(query: str, k: int = 5):
    """
    Realiza una búsqueda web utilizando la API de Google Custom Search para la consulta proporcionada.
    Retorna una tupla (snippets, titulo_primer_resultado, link_primer_resultado) si tiene éxito.
    - snippets: lista de fragmentos de texto relevantes de los primeros resultados.
    - titulo_primer_resultado: título del primer resultado (o cadena vacía si no disponible).
    - link_primer_resultado: URL del primer resultado (o cadena vacía si no disponible).
    En caso de error, retorna una cadena de error describiendo el problema.
    """
    if search is None:
        return "Error: GoogleSearchAPIWrapper no está disponible. Verifique la instalación y las claves API."
    try:
        # Ejecutar la búsqueda y obtener los resultados
        resultados = search.results(query, k)
    except Exception as e:
        return f"Error en la búsqueda externa: {e}"

    # Extraer los snippets de texto de los resultados (si existen)
    snippets = [res['snippet'].replace("\n", " ").strip() for res in resultados if 'snippet' in res]
    # Preparar título y link del primer resultado (si existen)
    primer_titulo = resultados[0].get('title', '') if resultados else ''
    primer_link = resultados[0].get('link', '') if resultados else ''
    # Devolver lista de snippets (o vacía) junto con info del primer resultado
    return snippets, primer_titulo, primer_link
