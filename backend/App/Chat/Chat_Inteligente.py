import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from langchain_community.utilities.google_search import GoogleSearchAPIWrapper #para buscar por internet
from dotenv import load_dotenv
import os

load_dotenv()

gemini_api_key = os.getenv("GOOGLE-API-KEY")
search_api_key = os.getenv("GOOGLE-SEARCH-API-KEY")
google_cse_id = os.getenv("GOOGLE-SEARCH-ID")
qdrant_url = os.getenv("QDRANT-URL")
qdrant_api_key = os.getenv("QDRANT-API-KEY")
collection_name = os.getenv("COLLECTION-NAME")

Sin_Informacion = "No tengo información sobre eso en mi base de datos"  #Esto es para cuando no encuentre informacion en la base de datos

# --- 2. Inicialización de Componentes ---

@st.cache_resource  #Lo que vimos en clase
def inicializar_componentes():  #Lo voy a meter todo en una funcion
    
    # Modelo a usar
    llm = init_chat_model("gemini-1.5-pro", model_provider="google_genai", api_key=gemini_api_key, temperature = 0.2)
    
    # Ahora los embedding
    embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
    
    # Todo lo de Qdrant
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name,
                                     embedding=embeddings, retrieval_mode=RetrievalMode.DENSE)
    
    # Prompt para la cadena de la base de datos
    rag_prompt_template = f"""
    Basándote únicamente en el siguiente contexto, responde la pregunta del usuario de forma clara y concisa.
    Si la información no está en el contexto, responde EXACTAMENTE: "{Sin_Informacion}". No añadas nada más.

    Contexto: {{context}}
    Pregunta: {{question}}
    Respuesta:
    """

    # rag_prompt_template =f"""
    # Basándote únicamente en el siguiente contexto, responde la pregunta del usuario
    # Eres un agente experto en razonamiento avanzado.

    # Cuando un usuario te hace una pregunta, no debes limitarte a responderla directamente. Antes de dar tu respuesta final, debes:

    # Formularte entre 2 y 4 auto-preguntas que te ayuden a enriquecer la respuesta.
    # Responder esas auto-preguntas de forma clara.
    # Redactar una respuesta principal lo más útil y precisa posible.
    # Añadir por separado las respuestas a las auto-preguntas como información adicional.
    
    # Contexto: {{context}}
    # Pregunta: {{question}}
    
    # Responde con el siguiente formato:
    
    # Respuesta:
    # [respuesta principal]

    # Si la información no está en el contexto, responde EXACTAMENTE: "{Sin_Informacion}". No añadas nada más.

    # Información adicional:
    # [auto-pregunta 1]: [respuesta breve]
    # [auto-pregunta 2]: [respuesta breve]
    # (opcional: más auto-preguntas si es relevante)

    # """

    # rag_prompt_template = f"""
    # Basándote únicamente en el siguiente contexto, responde la pregunta del usuario
                                      
    # Eres un agente especializado en razonamiento avanzado, con la misión de proporcionar respuestas absolutamente precisas, fiables y enriquecidas.

    # Cuando un usuario te haga una pregunta, no debes limitarte a responderla directamente. En su lugar, debes realizar un proceso interno de razonamiento que garantice la máxima calidad informativa. Para ello, sigue estos pasos con rigor:

    # Formula exactamente 2 auto-preguntas estratégicas y altamente relevantes que te ayuden a enriquecer y complementar la respuesta.
    # Responde esas auto-preguntas de forma clara, precisa y basada en el mejor conocimiento disponible.
    # Redacta una respuesta principal que sea lo más útil, completa, rigurosa y fiable posible.
    # Presenta las auto-preguntas y sus respuestas de forma separada, como información adicional complementaria.                               

    # Responde siguiendo estrictamente este formato:

    # ──────────────────────────────────────────────
    # 🔹 RESPUESTA PRINCIPAL:
    # [respuesta directa, clara y completa a la pregunta del usuario]

    # 🔸 INFORMACIÓN ADICIONAL:
    # • [auto-pregunta 1]: [respuesta breve, precisa y relevante]
    # • [auto-pregunta 2]: [respuesta breve, precisa y relevante]
    # (Si lo consideras útil, puedes incluir más auto-preguntas)
    # ──────────────────────────────────────────────
    # Si la información no está en el contexto, responde EXACTAMENTE: "{Sin_Informacion}". No añadas nada más.
    
    # Contexto: {{context}}
    # Pregunta: {{question}}  
    # """

    rag_prompt = PromptTemplate(template=rag_prompt_template, input_variables=["context", "question"]) #, 
    
    # Cadena RetrievalQA (Que es la primera opcion que hara el "agente nuestro" porque no hay agente como tal))
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": rag_prompt}
    )
    
    # Lo que busca en internet
    search_tool = GoogleSearchAPIWrapper(
        google_api_key=search_api_key,
        google_cse_id=google_cse_id
    )
    
    return llm, rag_chain, search_tool  #Devuelvo el modelo a uar, la cadena de la base de datos y la herramietna de internet

# Obtengo cada componente. Se puede hacer sin la funcin pero asi lo tengo todo en 1, tanto el modelo a uar como la busqueda en internet y en la base de datos
llm, rag_chain, search_tool = inicializar_componentes()


if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True) #Vamos guardando la caonversacion
if 'source_documents' not in st.session_state:
    st.session_state.source_documents = []

def generar_respuesta(user_prompt: str): #Esto es para elegir o buscar en la base de datos o buscar en internet
    
    # PRimero hago la funcion de RAG para la base de datos
    rag_result = rag_chain.invoke({"query": user_prompt})
    rag_answer = rag_result['result']
    
    st.session_state.source_documents = rag_result.get('source_documents', [])

    #......................................................................................................................................

    # La logica para decidir si coge RAG o busqueda por internet

    #......................................................................................................................................
    
    #Si no tengo informacion (Sin_Informacion) en el RAG entonces uso la busqueda en internet
    if Sin_Informacion in rag_answer:
        # Si RAG no encontro nada, se procede a buscar en Internet
        st.session_state.source_documents = [] # Limpiamos las fuentes RAG porque no fueron útiles
        
        st.info("No se encontró información en la base de datos. Buscando en Internet...")
        
        search_results = search_tool.run(user_prompt)   #Ejecuto la herramienta de busqueda en internet
        
        # Creo un prompt para internet
        internet_prompt = f"""
        Eres un asistente de IA. Basándote en el historial de la conversación y los siguientes resultados de una búsqueda en Internet, 
        responde a la "Pregunta nueva" del usuario de una forma amable y útil.

        Historial de la conversación:
        {st.session_state.memory.chat_memory}

        Resultados de búsqueda:
        "{search_results}"

        Pregunta nueva: {user_prompt}
        Respuesta final:
        """
        
        # respuesta del modelo con internet
        final_response = llm.invoke(internet_prompt).content
        
    else:   #Si si tengo informacion en la base de datos entonces doy esa respuesta
        final_response = rag_answer 
        
    # Guardamos la interacción en la memoria
    st.session_state.memory.save_context({"input": user_prompt}, {"output": final_response})
    
    return final_response

# DECIRLE A CHATGPT QUE ME PASE EL CHAT A STREAMLIT
# --- 4. Interfaz de Streamlit ---
st.title("Chat con tu secretario profesional 🤖")
st.caption("Conectado a la base de datos interna y a Internet.")

# Muestra el historial del chat
for msg in st.session_state.memory.chat_memory.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)

# Input del usuario
if prompt := st.chat_input("Máquina, ¿qué quieres saber?"):
    st.chat_message("user").write(prompt)
    
    with st.spinner("Pensando..."): #Esto es como esperando a (La respuesta en este caso)
        response_text = generar_respuesta(prompt)
    
    st.chat_message("assistant").write(response_text)

    # El expander muestra las fuentes si la herramienta RAG fue usada con éxito
    with st.expander("**Fuentes de datos (si se usó la base de datos)**"):
        if st.session_state.source_documents:
            for doc in st.session_state.source_documents:
                st.info(f"Extracto: \n> {doc.page_content[:300]}...")
                if doc.metadata:
                    st.caption(f"Metadatos: {doc.metadata}")
        else:
            st.warning("No se consultó la base de datos interna o no se encontró información relevante en ella para esta respuesta.")
