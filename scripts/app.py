import streamlit as st
import unicodedata
from ask_gemini import consultar_gemini

# =============================================================
# 1. CONFIGURACIÓN DE LA APP
# =============================================================
st.set_page_config(
    page_title="Knowledge Navigator",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Knowledge Navigator")
st.caption("Explora tus documentos PDF vectorizados con IA. Si la información local no es suficiente, se activa una búsqueda web para enriquecer la respuesta.")

# =============================================================
# 2. FUNCIONES AUXILIARES
# =============================================================
def limpiar_texto(texto):
    """Limpia texto de errores de codificación y normaliza caracteres."""
    if not texto:
        return ""
    if not isinstance(texto, str):
        texto = str(texto)
    texto = unicodedata.normalize("NFKD", texto)
    texto = texto.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    return texto.strip()

# =============================================================
# 3. INTERFAZ DE USUARIO
# =============================================================
pregunta = st.text_input("✏️ Escribe tu pregunta:")

if pregunta:
    with st.spinner("🧐 Procesando con IA..."):
        try:
            respuesta, razonamiento, fuentes, uso_externo = consultar_gemini(pregunta)
        except Exception as e:
            st.error(f"❌ Error al procesar la pregunta: {e}")
            st.stop()

        # ✅ Limpieza de texto
        respuesta_limpia = limpiar_texto(respuesta)
        razonamiento_limpio = limpiar_texto(razonamiento)

        # --- Mostrar respuesta principal
        incluye_encabezado = any(etq in respuesta_limpia.lower() for etq in ["📘 respuesta", "📘 respuesta principal"])
        if not incluye_encabezado:
            st.markdown("## 📘 Respuesta")
        st.markdown(respuesta_limpia or "_No se obtuvo respuesta._")

        # --- Mostrar razonamiento adicional (si no viene ya incluido)
        razonamiento_incluido = "💡 información adicional" in respuesta_limpia.lower() or "💡 información adicional" in razonamiento_limpio.lower()
        if not razonamiento_incluido:
            st.markdown("## 💡 Información adicional")
            st.markdown(razonamiento_limpio or "_No se proporcionó explicación adicional._")

        # --- Mostrar fuentes
        st.markdown("## 📚 Fuentes consultadas")
        if fuentes:
            for fuente in fuentes:
                st.markdown(f"- {limpiar_texto(fuente)}")
        else:
            st.markdown("_No se consultaron fuentes._")

        # --- Mostrar nota sobre el origen de la información
        st.markdown("## 🔍 Nota sobre el origen de la información")
        if uso_externo and any("Documento" in f for f in fuentes):
            st.info("Se utilizó información combinada: documentos PDF + búsqueda web.")
        elif uso_externo:
            st.info("Se utilizó exclusivamente información de búsqueda web externa.")
        else:
            st.success("Se respondió exclusivamente con información de los documentos PDF.")
