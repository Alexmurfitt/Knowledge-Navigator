import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import init_chat_model
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
from dotenv import load_dotenv
import os

load_dotenv()
gemini_api_key = os.getenv("GOOGLE-API-KEY")
search_api_key = os.getenv("GOOGLE-SEARCH-API-KEY")
google_cse_id = os.getenv("GOOGLE-SEARCH-ID")
qdrant_url = os.getenv("QDRANT-URL")
qdrant_api_key = os.getenv("QDRANT-API-KEY")
collection_name = os.getenv("COLLECTION-NAME")

#......................................................................................................................................

#DEFINO LA CADENA PARA QU HAGA BUSQUEDAS EN LA BASE DE DATOS

#......................................................................................................................................


def crear_cadena_rag(): #Aqui hago la cadena para buscar en la base de datos
    model = init_chat_model("gemini-1.5-pro", model_provider="google_genai", api_key=gemini_api_key, temperature = 0.2)

    embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name,
                                     embedding=embeddings, retrieval_mode=RetrievalMode.DENSE)
    
    prompt_template = """
    Basándote únicamente en el siguiente contexto, responde la pregunta del usuario de forma profesional y amable.
    Si la información no está en el contexto, di "No tengo información sobre eso en mi base de datos".

    Contexto: {context}
    Pregunta: {question}
    Respuesta:
    """
    rag_prompt = PromptTemplate.from_template(prompt_template)  #El from_template hace que no tenga que pasar los input
    
    return RetrievalQA.from_chain_type(
        llm=model,
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": rag_prompt}
    )

#......................................................................................................................................

#PARA STREAMLIT GUARDAMOS EN EL SESSION_STATE

#......................................................................................................................................

if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = crear_cadena_rag()
    st.session_state.search_tool = GoogleSearchAPIWrapper(google_api_key=search_api_key, google_cse_id=google_cse_id)   #Esto es la busqueda de internet
    st.session_state.llm = init_chat_model("gemini-1.5-pro", model_provider="google_genai", api_key=gemini_api_key, temperature = 0.2)
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
    st.session_state.messages = []


st.title("Chat con tu compi virtual ")
st.caption("Conectado a la base de datos. Activa el interruptor para buscar en Internet. Ya no tienes excusas.")

search_internet = st.toggle("Buscar en Internet", key="search_mode")

# Muestra el historial del chat
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)

# Input del usuario
if prompt := st.chat_input("Máquina, ¿qué quieres saber?"):
    st.chat_message("user").write(prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    response_text = ""
    source_documents = []

    if search_internet:
        with st.spinner("Buscando en Internet pesado..."):  
            search_results = st.session_state.search_tool.run(prompt)   #Aqui corre la herramienta de busqueda por internet

            internet_prompt = f"""
            Eres un asistente de investigación profesional. Basándote ÚNICAMENTE en los siguientes resultados de búsqueda, responde a la pregunta del usuario de forma clara y concisa.
            No añadas información que no esté en los resultados. Si los resultados no son suficientes para responder, indícalo amablemente. Añade una pregunta a modo de sugerencia relacionado con la pregunta del usuario y los resultados obtenidos.

            **Resultados de Búsqueda:**
            "{search_results}"

            **Pregunta del Usuario:**
            "{prompt}"

            **Respuesta Profesional:**
            """
            
            llm_response = st.session_state.llm.invoke(internet_prompt)
            response_text = llm_response.content

    else:
        with st.spinner("Buscando en la base de datos asi que tranquilo si tarda..."):
            result = st.session_state.rag_chain.invoke({"query": prompt})
            response_text = result['result']
            source_documents = result.get('source_documents', [])

    st.chat_message("assistant").write(response_text)
    st.session_state.messages.append(AIMessage(content=response_text))

    if not search_internet and source_documents:
        with st.expander("**Fuentes de datos (Base de datos interna)**"):
            for doc in source_documents:
                st.info(f"Extracto: \n> {doc.page_content[:300]}...")
                if doc.metadata:
                    st.caption(f"Metadatos: {doc.metadata}")