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

        # ✅ Respuesta principal
        st.markdown("## 📘 Respuesta")
        st.markdown(limpiar_texto(respuesta) or "_No se obtuvo respuesta._")

        # 💡 Explicación adicional
        st.markdown("## 💡 Información adicional")
        st.markdown(limpiar_texto(razonamiento) or "_No se proporcionó explicación adicional._")

        # 📚 Fuentes consultadas
        st.markdown("## 📚 Fuentes consultadas")
        if fuentes:
            for fuente in fuentes:
                st.markdown(f"- {limpiar_texto(fuente)}")
        else:
            st.markdown("_No se consultaron fuentes._")

        # 🔍 Origen del conocimiento
        st.markdown("## 🔍 Nota sobre el origen de la información")
        if uso_externo and any("Documento" in f for f in fuentes):
            st.info("Se utilizó información combinada: documentos PDF + búsqueda web.")
        elif uso_externo:
            st.info("Se utilizó exclusivamente información de búsqueda web externa.")
        else:
            st.success("Se respondió exclusivamente con información de los documentos PDF.")
