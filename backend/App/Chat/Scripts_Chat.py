import streamlit as st
from langchain.chat_models import init_chat_model
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
import os

load_dotenv()
google_api_key = os.getenv("GOOGLE-API-KEY")
qdrant_url = os.getenv("QDRANT-URL")
qdrant_api_key = os.getenv("QDRANT-API-KEY")
collection_name = os.getenv("COLLECTION-NAME")

if 'chain' not in st.session_state: #Para Guardar en memoria de streamlit
    st.session_state.model = init_chat_model("gemini-2.5-flash", model_provider="google_genai", api_key=google_api_key)
    st.session_state.memory = ConversationBufferMemory()
    
    # Vector store
    embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name,
                                     embedding=embeddings, retrieval_mode=RetrievalMode.DENSE)

    # Prompt
    prompt_template = """   
    Eres secretario administrativo profesional de una empresa a nivel global con gran reputación.
    Responderas de manera breve y efectiva y al final de la respuesta sugueriras otras preguntas.

    Tu misión es dar la respuesta más concreta a lo que se te pide de esta forma:
    1) Para poder responder bien a la {question} tendrás que indagar cómo puedes responder más adecuadamente según el {context}
    2) Darás la respuesta breve a lo que se te pide.
    3) Pondrás un ejemplo sencillo para entender mejor el concepto que se te pida (Opcional: Solo si es necesario ya que cuando pida datos de personas de la empresa obviaras el ejemplo.)
    4) Sugerirás, acorde al {context} una pregunta que le podría interesar al usuario 
    Si la información para responder no se encuentra en el contexto proporcionado, debes decir con elegancia que no tienes dicha información en tu base de datos. No inventes nada.

    **Contexto de la información:**
    {context}

    **Pregunta del usuario:**
    {question}

    **Tu Respuesta de secretario profesional:**
    """
    secretario_PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Creación de la cadena
    st.session_state.chain = RetrievalQA.from_chain_type(
        llm=st.session_state.model,
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": secretario_PROMPT}
    )



st.title("Chat con tu secretario")

for msg in st.session_state.memory.chat_memory.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)

# Input del usuario
if prompt := st.chat_input("¿Qué deseas saber?"):
    st.chat_message("user").write(prompt)

    st.session_state.memory.chat_memory.add_user_message(prompt)

    result = st.session_state.chain.invoke({"query": prompt})

    response_text = result['result']
    source_documents = result['source_documents']

    st.session_state.memory.chat_memory.add_ai_message(response_text)
    st.chat_message("assistant").write(response_text)

    with st.expander("**Fuentes de datos (RAG)**"):
        if source_documents:
            for doc in source_documents:
                st.info(f"Extracto: \n> {doc.page_content[:300]}...")
                if doc.metadata:
                    st.caption(f"Metadatos: {doc.metadata}")
        else:
            st.warning("No encontré ninguna página en mi base de datos que hable de eso.")