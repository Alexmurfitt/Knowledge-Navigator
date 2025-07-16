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
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai", api_key=os.getenv("GOOGLE-API-KEY"))

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
            all_docs.extend(extract_text_from_pdf(f))

        chunks = split_chunks(all_docs)
        print(f"📄 Total de fragmentos: {len(chunks)}")

        embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
        global vector_store
        vector_store = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            url=os.getenv("QDRANT-URL"),
            api_key=os.getenv("QDRANT-API-KEY"),
            collection_name="Knowledge-Navigator",
            force_recreate=False
        )

        return JSONResponse(
            content={"message": f"{len(chunks)} fragmentos cargados correctamente."},
            status_code=200
        )

    except Exception as e:
        print("❌ Error en /upload:", e)
        raise HTTPException(status_code=500, detail="Error al procesar los archivos.")

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
    return {"answer": result["result"]}
