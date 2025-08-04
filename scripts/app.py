# app.py — Interfaz Streamlit definitiva para Knowledge Navigator
import streamlit as st
import unicodedata
from ask_gemini import consultar_gemini, cargar_historial_json

st.set_page_config(page_title="Knowledge Navigator", page_icon="🧠", layout="centered")
st.title("🧠 Knowledge Navigator")
st.caption("Explora tus documentos PDF vectorizados con IA. Si no hay información local suficiente, se activa una búsqueda web.")

if "historial_turnos" not in st.session_state:
    st.session_state.historial_turnos = []

cargar_historial_json()

# Limpieza básica de texto

def limpiar(texto):
    if not texto:
        return ""
    texto = unicodedata.normalize("NFKD", str(texto))
    return texto.encode("utf-8", errors="replace").decode("utf-8", errors="replace").strip()

pregunta = st.text_input("✏️ Escribe tu pregunta:")

if pregunta:
    with st.spinner("🧠 Pensando..."):
        try:
            respuesta, razonamiento, fuentes, uso_externo = consultar_gemini(pregunta)
        except Exception as e:
            st.error(f"❌ Error al procesar la pregunta: {e}")
            st.stop()

    st.session_state.historial_turnos.append({"usuario": pregunta, "asistente": respuesta})

    st.markdown("## 📘 Respuesta")
    st.markdown(limpiar(respuesta) or "_No se obtuvo respuesta._")

    if razonamiento:
        st.markdown("## 💡 Información adicional")
        st.markdown(limpiar(razonamiento))

    st.markdown("## 📚 Fuentes consultadas")
    if fuentes:
        for f in fuentes:
            st.markdown(f"- {limpiar(f)}")
    else:
        st.markdown("_No se consultaron fuentes._")

    st.markdown("## 🔍 Nota sobre el origen de la información")
    if uso_externo and any("Documento" in f for f in fuentes):
        st.info("Se utilizó información combinada: documentos PDF + búsqueda web.")
    elif uso_externo:
        st.info("Se utilizó exclusivamente información de búsqueda web externa.")
    else:
        st.success("Se respondió exclusivamente con información de los documentos PDF.")
