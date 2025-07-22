from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import List
import tempfile
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient, models  #Añadido por Aaron
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from langchain.chains import RetrievalQA
from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from fastapi.responses import JSONResponse
from fastapi import HTTPException

load_dotenv()

google_api_key=os.getenv("GOOGLE-API-KEY")
url=os.getenv("QDRANT-URL")
api_key=os.getenv("QDRANT-API-KEY")
search_api_key = os.getenv("GOOGLE-SEARCH-API-KEY")
google_cse_id = os.getenv("GOOGLE-SEARCH-ID")
collection_name = os.getenv("COLLECTION-NAME")

Sin_Informacion = "No tengo información sobre eso en mi base de datos"  #Puesto por Aaron


app = FastAPI()

# CORS: Permitir conexiones desde tu HTML
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- Variables globales ---
# vector_store = None   #
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai", api_key=google_api_key)
embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest") #Puesto por Aaron
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qdrant_client = QdrantClient(url=url, api_key=api_key)  #Añadido por Aaron 
vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name,
                                 embedding=embeddings, retrieval_mode=RetrievalMode.DENSE)  #Añadido por Aaron
search_tool = GoogleSearchAPIWrapper(google_api_key=search_api_key, google_cse_id=google_cse_id)    #Añadido por Aaron
suggested_question = None   #Por Aaron


# --- Utilidades ---
def extract_text_from_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    os.remove(tmp_path)
    return documents

def split_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    return splitter.split_documents(docs)

# --- Endpoint: subir PDFs ---



# --- Endpoint: preguntar al bot ---
class ChatRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_bot(req: ChatRequest):
    global suggested_question, memory

    # Lógica para manejar respuestas afirmativas a preguntas sugeridas
    affirmative_responses = ["si", "sí", "yes", "ok", "vale", "dale", "procede", "claro"]
    actual_prompt = req.question
    
    if suggested_question and req.question.lower().strip() in affirmative_responses:
        actual_prompt = suggested_question
        suggested_question = None  # Limpiar la sugerencia después de usarla

    # 1. Ejecutar la cadena RAG
    rag_prompt = PromptTemplate(
        template=f"""Basándote únicamente en el siguiente contexto, responde la pregunta del usuario explicando lo encontrado. Haz una pregunta relacionada con el contexto encontrado para recomendar al usuario.
        Si la información no está en el contexto, responde EXACTAMENTE: "{Sin_Informacion}". No añadas nada más.
        Contexto: {{context}}\nPregunta: {{question}}\nRespuesta:""",
        input_variables=["context", "question"]
    )
    rag_chain = RetrievalQA.from_chain_type(
        llm=model,
        retriever=vector_store.as_retriever(search_kwargs={'k': 6}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": rag_prompt}
    )
    rag_result = rag_chain.invoke({"query": actual_prompt})
    rag_answer = rag_result['result']
    sources = rag_result.get('source_documents', [])

    # Aqui miro si busca o no en internet
    if Sin_Informacion in rag_answer:
        search_results = search_tool.run(actual_prompt)
        internet_prompt = f"""Eres un asistente de IA. Basándote en el historial de la conversación y los siguientes resultados de una búsqueda en Internet, 
        responde a la "Pregunta nueva" del usuario de una forma amable y útil.
        Historial de la conversación: {memory.chat_memory}
        Resultados de búsqueda: "{search_results}"
        Pregunta nueva: {actual_prompt}
        Respuesta final:"""
        final_response = model.invoke(internet_prompt).content
        final_sources = []
    else:
        final_response = rag_answer
        final_sources = [doc.dict() for doc in sources] # Convertir documentos a diccionarios

    # Aqui es para guardar la respuesta
    if '?' in final_response:
        potential_question = final_response.split('?')[-1].strip()
        if len(potential_question) > 5:
            suggested_question = potential_question
    else:
        suggested_question = None
    #Aqui se guarda en memoria
    memory.save_context({"input": actual_prompt}, {"output": final_response})
    
    return {
        "answer": final_response,
        "suggested_question": suggested_question,
        "sources": final_sources
    }

@app.get("/chat-history")
def get_chat_history():
    return {"history": [msg.content for msg in memory.chat_memory.messages]}


@app.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    try:
        print("📥 Archivos recibidos:", [f.filename for f in files])
        all_docs = []
        for f in files:
            if not f.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"El archivo {f.filename} no es un PDF válido.")
            docs = extract_text_from_pdf(f)
            for doc in docs:
                doc.metadata["document_name_id"] = f.filename
            all_docs.extend(docs)


        crear_indice(collection_name=collection_name)
        chunks = split_chunks(all_docs)
        print(f"📄 Total de fragmentos: {len(chunks)}")
#model="mxbai-embed-large:latest"
        embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
        global vector_store
        vector_store = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            url=url,
            api_key=api_key,
            collection_name=collection_name,
            force_recreate=False
        )

        return JSONResponse(
            content={"message": f"{len(chunks)} fragmentos cargados correctamente."},
            status_code=200
        )

    except Exception as e:
        print("❌ Error en /upload:", e)
        raise HTTPException(status_code=500, detail="Error al procesar los archivos.")

def crear_indice(collection_name : str):
    try:
        client = QdrantClient(
            url=url, 
            api_key=api_key
        )
        print("Creando índice para la ruta anidada 'metadata.document_name_id'...")
        
        # Crea el índice en la colección apuntando a la ruta anidada correcta
        client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.document_name_id",  # <-- LA CLAVE ESTÁ AQUÍ
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        print(" ¡Índice para 'metadata.document_name_id' creado con éxito!")

    except Exception as e:
        print(f" No se pudo crear el indice (puede que exista): {e}")





from qdrant_client import QdrantClient, models

@app.delete("/delete")
async def eliminar_pdf_qdrant(collection_name: str, pdf_nombre: str):
    print(f"📥 DELETE recibido para: {pdf_nombre} en colección: {collection_name}")

    # Conexión con el cliente de Qdrant
    client = QdrantClient(
        url=url, 
        api_key=api_key
    )

    filter = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.document_name_id", # El campo de metadatos que creamos
                match=models.MatchValue(value=pdf_nombre),
            )
        ]
    )

    print(filter)

    try:
        # Usamos  delete para borrar los puntos que cumplen con el filtro
        respuesta = client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(filter=filter),
            wait= True
        )
        print(f"SE ha borrado el pdf con nombre: '{pdf_nombre}'.")
        print(f"Respuesta de Qdrant: {respuesta}")
        return JSONResponse(content={"success": True})
    
    except Exception as e:
        print(f"Error al eliminar los datos de Qdrant: {e}")
        return JSONResponse(content={"success": False}, status_code=500)
    








@app.get("/documentos_unicos/{collection_name}")
async def mostrar_documentos_unicos(collection_name: str):
    try:
        client = QdrantClient(
            url=url, 
            api_key=api_key
        )

        scrolled_points, llamada= client.scroll( # La "llamada" es porque client.scroll devuelve (puntos, llamada) y la llamada es para la siguiente llamda (no se necesita para nada), es decir me devuelve una tupla de 2 valores, Puntos y llamada
            collection_name=collection_name,
            limit=10000,    # El limit es la cantidad de endpoint que quiero ver
            with_payload=True   #Awui esta incluyendo los metadatos
        )

        print(f"Lo que devuelve scrolled_points = {scrolled_points[0]}")
        print(50*"-")
        print(f"Lo que devuelve scrolled_points = {scrolled_points[1]}")
        print(f"En este caso, quiero saber la página donde extrajo la información, que es: {scrolled_points[0].payload['metadata']['total_pages']} ")
        print(f"Lo que devuelve _ : {llamada}")

        document_names = set()  #En vez de un diccionario o una lista pongo un set ya que almacena documentos unicos
        for point in scrolled_points:
            if point.payload and "metadata" in point.payload:   #Si hay payload y metadata esta dentro de payload (Lo de metadata es dentro de los metadatos hay un campo llamado metadata y dentro estan el resto de variables)
                # 2. Buscamos 'document_name_id' DENTRO de 'metadata'
                metadata_dict = point.payload["metadata"]
                if "document_name_id" in metadata_dict: #Si dentro de metadata esta document_name_id
                    document_names.add(metadata_dict["document_name_id"])   #Lo añadimos al set

        return sorted(list(document_names))


    except Exception as e:
        print(f"Ha ocurrido un error: {e}")
        return []

