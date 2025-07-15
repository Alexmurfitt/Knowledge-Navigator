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
google_api_key = os.getenv("GOOGLE-API-KEY")    #Seleecionamos la apikey del modelo

# Modelo y memoria
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai", api_key=google_api_key)  #Elegimos el modelo que utilizaremos
memory = ConversationBufferMemory() #Activamos la memoria para el chat

# Vector store
def init_components():
    embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest") #Sistema de embedding a utilizar
    client = QdrantClient(url=os.getenv("QDRANT-URL"), api_key=os.getenv("QDRANT-API-KEY")) #Nos conectamos a nuestro qdrant (el client es el de la cuenta)
    return QdrantVectorStore(client=client, collection_name="Knowledge-Navigator",
                             embedding=embeddings, retrieval_mode=RetrievalMode.DENSE)  #Aqui ya pasamos todo

vector_store = init_components()    #Aqui ya tenemos todos los vectores guardados

# Vamos ahora a poner un prompt de sistema que tendra el modelo
prompt_template = """   
Eres secretario administrativo profesional de una empresa a nivel global con gran reputación.
Responderas de manera breve y efectiva y al final de la respuesta sugueriras otras preguntas.

Tu misión es dar la respuesta más concreta a lo que se te pide de esta forma:
1) Darás la respuesta breve a lo que se te pide.
2) Pondrás un ejemplo sencillo para entender mejor el concepto que se te pida (Opcional: Solo si es necesario ya que cuando pida datos de personas de la empresa obviaras el ejemplo.)
3) Sugerirás, acorde al {context} una pregunta que le podría interesar al usuario 
Si la información para responder no se encuentra en el contexto proporcionado, debes decir con elegancia que no tienes dicha información en tu base de datos. No inventes nada.

**Contexto de la información:**
{context}

**Pregunta del usuario:**
{question}

**Tu Respuesta de secretario profesional:**
"""
#LAs variables a poner son el contexto (Que sera el RAG) y la pregunta del usuario 
secretario_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)   #Seleccionamos la plantilla que tendrá el chat


# HAcemos la cadena que utilizara
chain = RetrievalQA.from_chain_type(    #En la cadena vamos a pasarle todo junto
    llm=model,
    retriever=vector_store.as_retriever(),  #Para devolver los vectores
    return_source_documents=True,   #Devuelve de donde saca los documentos
    chain_type_kwargs={"prompt": secretario_PROMPT}   #Esto es para pasar el prompt nuestro
)   


st.title("Chat con tu secretario")

#Lo de ahroa es para que se vea bien en streamlit
for msg in memory.chat_memory.messages: #Para cada mensaje guardado en el chat
    role = "user" if isinstance(msg, HumanMessage) else "assistant" #Aqui es si el mensaje es escrito por el humano (HumanMessage) entonces el role es user sino es assistant
    st.chat_message(role).write(msg.content)    #Esto es para mostrar en el streamlit quien escribio y con el content mete el contenido

#Ahora se pondra el input del uduario
if prompt := st.chat_input("Qué deseas máquina?"):   #Esto es la caja de texto para hablar
    st.chat_message("user").write(prompt)   #Aqui pone para escribir el usuario
    memory.chat_memory.add_user_message(prompt) #Añade el prompt del usuario a la memoria

    result = chain.invoke({"query": prompt})    #Aqui utiliza al modelo y el prompt que se le da

    response_text = result['result']
    source_documents = result['source_documents']

    memory.chat_memory.add_ai_message(response_text)
    st.chat_message("assistant").write(response_text)

    # Preguntar a chatgpt como mostrar los archivos de RAG 
    # Muestra las fuentes del RAG que se usaron
    with st.expander("**Fuentes de datos (RAG)**"):
        if source_documents:
            for doc in source_documents:
                st.info(f"Extracto: \n> {doc.page_content[:300]}...")
                if doc.metadata:
                    st.caption(f"Metadatos: {doc.metadata}")
        else:
            # Este mensaje aparecerá si el RAG no encontró ningún documento relevante.
            st.warning("No encontré ninguna página en mi base de datos que hable de eso.")