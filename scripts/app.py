# ✅ FUSIÓN COMPLETA – Knowledge Navigator con RAG + Razonamiento + Búsqueda Web

import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain_google_community import GoogleSearchAPIWrapper
from ask_pdf_qdrant_mongodb_limpio import responder  # Sistema RAG + razonamiento híbrido
from detectar_similitud import detectar_redundancia
from dotenv import load_dotenv
import os
import sys

# Permitir imports desde carpetas superiores
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
load_dotenv()

# --- 🔐 Claves de entorno ---
google_api_key = os.getenv("GOOGLE_API_KEY")
google_cse_id = os.getenv("GOOGLE_CSE_ID")

# --- 🌐 Herramienta de búsqueda web ---
search_tool = GoogleSearchAPIWrapper(
    google_api_key=google_api_key,
    google_cse_id=google_cse_id
)

# --- 💾 Memoria de conversación y fuentes ---
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "source_documents" not in st.session_state:
    st.session_state.source_documents = []
if "adicionales" not in st.session_state:
    st.session_state.adicionales = []

# --- 🤖 Función de respuesta principal ---
def obtener_respuesta(pregunta_usuario: str):
    advertencia = detectar_redundancia(pregunta_usuario)

    try:
        # Paso 1: RAG con razonamiento estructurado + CoT
        respuesta, adicionales, fuentes = responder(pregunta_usuario)

        # Paso 2: Si no hay resultados útiles, buscar en la web
        if "No tengo información" in respuesta:
            st.info("No se encontró información en la base interna. Buscando en Internet...")
            resultados_web = search_tool.run(pregunta_usuario)

            # Prompt para Internet
            prompt_internet = f"""
Eres un asistente de IA especializado en ayudar con dudas sobre documentación empresarial.

Basándote en los siguientes resultados de búsqueda en Internet, responde de forma clara, útil y precisa:

Resultados:
\"{resultados_web}\"

Pregunta del usuario:
\"{pregunta_usuario}\"

Respuesta final:
"""
            from langchain_ollama import ChatOllama
            modelo = ChatOllama(model="llama3", temperature=0.2)
            respuesta = modelo.invoke(prompt_internet).content

            st.session_state.source_documents = []
            st.session_state.adicionales = []
        else:
            # Guardar fuentes y adicionales en sesión
            st.session_state.source_documents = fuentes
            st.session_state.adicionales = adicionales

            # 👉 Mostrar en consola
            if fuentes:
                print("\n📚 DOCUMENTOS CONSULTADOS:")
                for i, doc in enumerate(fuentes):
                    nombre = doc.metadata.get("source", "desconocido")
                    pagina = doc.metadata.get("page", "N/A")
                    print(f"  {i+1}. {nombre} (p. {pagina})")

            if adicionales:
                print("\n🔸 INFORMACIÓN ADICIONAL:")
                for frase in adicionales:
                    print(f"• {frase}")

        # Guardar historial conversacional
        st.session_state.memory.save_context({"input": pregunta_usuario}, {"output": respuesta})
        return respuesta, advertencia

    except Exception as e:
        return f"❌ Error durante la generación de la respuesta: {str(e)}", None

# --- 🖥️ Interfaz visual ---
st.title("Knowledge Navigator")
st.caption("Sistema híbrido con recuperación de documentos, razonamiento y búsqueda en Internet")

# Mostrar historial de conversación
for msg in st.session_state.memory.chat_memory.messages:
    rol = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(rol).write(msg.content)

# Captura de nueva pregunta
if prompt := st.chat_input("¿En qué puedo ayudarte hoy?"):
    st.chat_message("user").write(prompt)
    with st.spinner("Generando respuesta..."):
        respuesta_texto, advertencia = obtener_respuesta(prompt)

    if advertencia:
        st.warning(advertencia)

    st.chat_message("assistant").write(respuesta_texto)

    # Mostrar información adicional (frases extra del razonador)
    if st.session_state.adicionales:
        with st.expander("🔸 Información adicional generada por el sistema"):
            for frase in st.session_state.adicionales:
                st.markdown(f"- {frase}")

    # Mostrar fuentes utilizadas
    with st.expander("📚 Fuentes de datos consultadas"):
        if st.session_state.source_documents:
            for doc in st.session_state.source_documents:
                if isinstance(doc, str):
                    st.info(doc)
                else:
                    nombre = doc.metadata.get("source", "desconocido")
                    pagina = doc.metadata.get("page", "N/A")
                    st.info(f"{nombre} (p. {pagina})")
        else:
            st.info("No se utilizaron fuentes documentales internas.")
