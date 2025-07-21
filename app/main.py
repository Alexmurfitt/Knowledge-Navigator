# main.py (FastAPI backend)

from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import List
import tempfile
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from langchain.chains import RetrievalQA

load_dotenv()

google_api_key=os.getenv("GOOGLE-API-KEY")
url=os.getenv("QDRANT-URL")
api_key=os.getenv("QDRANT-API-KEY")
collection_name="Knowledge-Navigator"

app = FastAPI()

# CORS: Permitir conexiones desde tu HTML
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- Variables globales ---
vector_store = None
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai", api_key=google_api_key)

# Prompt
template = """   
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


secretario_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

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
from fastapi.responses import JSONResponse
from fastapi import HTTPException






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
    



# --- Endpoint: preguntar al bot ---
class ChatRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_bot(req: ChatRequest):
    global vector_store
    if not vector_store:
        return {"error": "No hay documentos cargados."}

    
    chain = RetrievalQA.from_chain_type(
        llm=model,
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": secretario_PROMPT}
    )

    result = chain.invoke({"query": req.question})
    print(result)
    return {"answer": result["result"]}


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
