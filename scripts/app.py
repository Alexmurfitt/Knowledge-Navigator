# ✅ Knowledge Navigator – Versión mejorada y ordenada

import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv
from ask_gemini import responder  # Ruta al backend limpio
import unicodedata
import os
import sys

# 🔧 Configuración
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
load_dotenv()

# --- 🔍 Limpieza de texto ---
def limpiar_texto(texto):
    try:
        if texto is None:
            return ""
        if not isinstance(texto, str):
            texto = str(texto)
        texto = unicodedata.normalize("NFKD", texto)
        texto = texto.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        texto = ''.join(c for c in texto if c.isprintable())
        return texto.strip()
    except Exception as e:
        return f"[❌ Error al limpiar texto: {str(e)}]"

# --- 💬 Inicializar memoria de conversación ---
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "source_documents" not in st.session_state:
    st.session_state.source_documents = []
if "adicionales" not in st.session_state:
    st.session_state.adicionales = []
if "razonamiento" not in st.session_state:
    st.session_state.razonamiento = ""

# --- 🤖 Función principal de interacción ---
def obtener_respuesta(pregunta_usuario: str):
    try:
        respuesta, fuentes, razonamiento, adicionales = responder(pregunta_usuario)
        st.session_state.source_documents = fuentes
        st.session_state.adicionales = adicionales
        st.session_state.razonamiento = razonamiento

        st.session_state.memory.save_context({"input": pregunta_usuario}, {"output": respuesta})
        return respuesta
    except Exception as e:
        return f"❌ Error durante la generación de la respuesta: {str(e)}"

# --- 🖥️ Interfaz visual ---
st.title("Knowledge Navigator")
st.caption("Sistema con recuperación documental y búsqueda externa enriquecida")

# Mostrar historial previo
for msg in st.session_state.memory.chat_memory.messages:
    rol = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(rol).write(msg.content)

# Entrada del usuario
if prompt := st.chat_input("¿En qué puedo ayudarte hoy?"):
    st.chat_message("user").write(prompt)
    with st.spinner("Generando respuesta..."):
        respuesta_texto = obtener_respuesta(prompt)

    # ✅ 1. Mostrar la respuesta principal clara y precisa
    st.chat_message("assistant").markdown("### ✅ Respuesta clara y precisa")
    st.chat_message("assistant").write(respuesta_texto)

    # 💡 2. Mostrar razonamiento ampliado (solo si existe)
    if st.session_state.razonamiento.strip():
        st.markdown("### 💡 Información adicional para enriquecer la respuesta")
        with st.expander("🧠 Razonamiento ampliado (Chain of Thought)", expanded=True):
            st.markdown(st.session_state.razonamiento)

    # 📚 3. Mostrar fuentes consultadas
    if st.session_state.source_documents:
        st.markdown("### 📚 Fuentes de datos consultadas")
        with st.expander("🔍 Ver fuentes utilizadas"):
            for doc in st.session_state.source_documents:
                if isinstance(doc, str):
                    st.info(limpiar_texto(doc))
                else:
                    nombre = limpiar_texto(doc.metadata.get("source", "desconocido"))
                    pagina = limpiar_texto(str(doc.metadata.get("page", "N/A")))
                    contenido = limpiar_texto(doc.page_content)
                    st.info(f"📄 {nombre} (p. {pagina})\n\n{contenido}")

    # 🟠 4. Mostrar frases clave adicionales (si existen)
    if st.session_state.adicionales:
        st.markdown("### 🟠 Información adicional generada por el sistema")
        with st.expander("🔸 Frases destacadas del razonador"):
            for frase in st.session_state.adicionales:
                st.markdown(f"- {frase}")
