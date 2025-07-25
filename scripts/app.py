import streamlit as st
import unicodedata
from ask_gemini import consultar_gemini

# --- 🔧 Configuración de la app ---
st.set_page_config(page_title="Knowledge Navigator", page_icon="🧠", layout="centered")

# --- 🧠 Encabezado ---
st.title("🧠 Knowledge Navigator")
st.caption("Consulta inteligente de documentos PDF vectorizados con IA. "
           "Si no hay contexto local suficiente, se activa búsqueda web externa para garantizar una respuesta precisa.")

# --- 🔍 Función auxiliar para limpieza de texto ---
def limpiar_texto(texto):
    """Limpia texto eliminando caracteres no imprimibles y normaliza errores de codificación, preservando formato básico."""
    try:
        if not texto:
            return ""
        if not isinstance(texto, str):
            texto = str(texto)
        texto = unicodedata.normalize("NFKD", texto)
        texto = texto.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        # Conservar caracteres imprimibles y espacios (incluyendo saltos de línea)
        texto = ''.join(c for c in texto if c.isprintable() or c.isspace())
        return texto.strip()
    except Exception as e:
        return f"[❌ Error al limpiar texto: {str(e)}]"

# --- 💬 Entrada del usuario ---
with st.form("formulario_pregunta"):
    pregunta = st.text_input(
        label="✏️ Haz tu pregunta:",
        placeholder="Ej. ¿Qué es la transparencia algorítmica?",
        key="input_pregunta"
    )
    submitted = st.form_submit_button("🔍 Consultar")

# --- 🚀 Procesamiento y generación de respuesta ---
if submitted and pregunta.strip():
    with st.spinner("🧐 Procesando con IA..."):
        try:
            respuesta, razonamiento, fuentes, uso_web = consultar_gemini(pregunta.strip())
        except Exception as e:
            st.error(f"❌ Error al procesar la pregunta: {e}")
            st.stop()

    # ✅ Respuesta principal
    st.markdown("## ✅ Respuesta clara y precisa")
    if respuesta:
        # Mostrar error como error, o respuesta normal como éxito
        if respuesta.lower().startswith("error al generar respuesta"):
            st.error(limpiar_texto(respuesta))
        else:
            st.success(limpiar_texto(respuesta))
    else:
        st.warning("⚠️ No se pudo generar una respuesta para esta pregunta.")

    # 💡 Información adicional (si la respuesta NO es concisa)
    if razonamiento and razonamiento.strip():
        if not respuesta.endswith(".") or len(razonamiento.strip()) > 10:
            st.markdown("## 💡 Información adicional")
            with st.expander("🧠 Ver razonamiento ampliado (Chain of Thought)", expanded=True):
                st.markdown(limpiar_texto(razonamiento))


    # 📚 Fuentes consultadas
    st.markdown("## 📚 Fuentes consultadas")
    if fuentes:
        with st.expander("🔍 Ver fuentes utilizadas"):
            for i, fuente in enumerate(fuentes, start=1):
                fuente_limpia = limpiar_texto(fuente)
                st.markdown(f"- {fuente_limpia}")
    else:
        st.info("ℹ️ No se consultaron fuentes específicas para esta respuesta.")

    # 🌐 Aviso si se usó búsqueda externa
    if uso_web:
        st.markdown("---")
        st.info("🌐 *Se utilizó búsqueda externa en internet porque no se encontró información suficiente en los documentos locales.*")

# --- 📝 Pie de página ---
st.markdown("---")
st.caption("© 2025 Knowledge Navigator – Sistema de IA híbrida con recuperación semántica y razonamiento explicativo.")
