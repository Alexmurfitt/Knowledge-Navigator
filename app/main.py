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
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from langchain.docstore.document import Document
import fitz
from fastapi import WebSocket
from typing import Dict
from app.drive_untils import upload_file_to_drive
# Al principio de main.py
import re
from datetime import datetime

# Conexiones WebSocket activas por archivo
active_connections: Dict[str, WebSocket] = {}


load_dotenv()

google_api_key=os.getenv("GOOGLE-API-KEY")
url=os.getenv("QDRANT-URL")
api_key=os.getenv("QDRANT-API-KEY")
search_api_key = os.getenv("GOOGLE-SEARCH-API-KEY")
google_cse_id = os.getenv("GOOGLE-SEARCH-ID")
collection_name = os.getenv("COLLECTION-NAME")

Sin_Informacion = "No tengo información sobre eso en mi base de datos"  #Puesto por Aaron

app = FastAPI()

canceled_uploads = {}

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
def devolver_marcadores(doc):   #Nueva funcion
    """Estrategia 1: Extrae jerarquía desde los marcadores del PDF."""
    toc = doc.get_toc(simple=False)
    page_to_hierarchy = {}
    current_path = {}
    for level, title, page, _ in toc:
        current_path[level] = title
        keys_to_clear = [key for key in current_path if key > level]
        for key in keys_to_clear:
            del current_path[key]
        hierarchy = {f"H{k}": v for k, v in current_path.items()}
        page_to_hierarchy[page] = hierarchy
    
    final_map = {}
    last_known_hierarchy = {}
    for page_num in range(1, doc.page_count + 1):
        if page_num in page_to_hierarchy:
            last_known_hierarchy = page_to_hierarchy[page_num]
        final_map[page_num] = last_known_hierarchy.copy()
    return final_map

def extract_raw_blocks(doc):    #Nueva funcion
    """Estrategia 1: Extrae texto y tablas cuando hay marcadores."""
    blocks = []
    for numero_pagina, page in enumerate(doc):
        tables = page.find_tables()
        for i, tab in enumerate(tables):
            if not tab.to_pandas().empty:
                markdown_table = tab.to_pandas().to_markdown(index=False)
                blocks.append({"page": numero_pagina + 1, "type": "table", "content": markdown_table})

        text_blocks = page.get_text("dict", sort=True)["blocks"]
        for block in text_blocks:
            if "lines" in block and block['lines']:
                block_bbox = fitz.Rect(block['bbox'])
                if any(block_bbox.intersects(tab.bbox) for tab in tables):
                    continue
                block_text = " ".join(span['text'] for line in block['lines'] for span in line['spans']).strip()
                if block_text:
                    blocks.append({"page": numero_pagina + 1, "type": "text", "content": block_text})
    return blocks

def extract_blocks_by_layout(doc):  #Nueva funcion
    """Estrategia 2: Extrae y etiqueta bloques por layout cuando no hay marcadores."""
    font_counts = {}
    for page in doc:
        for block in page.get_text("dict")['blocks']:
            if 'lines' in block:
                for line in block['lines']:
                    if 'spans' in line:
                        for span in line['spans']:
                            size = round(span['size'])
                            font_counts[size] = font_counts.get(size, 0) + 1
    
    sorted_sizes = sorted(font_counts.keys(), reverse=True)
    size_to_type = {}
    if len(sorted_sizes) > 0: size_to_type[sorted_sizes[0]] = 'H1'
    if len(sorted_sizes) > 1: size_to_type[sorted_sizes[1]] = 'H2'
    if len(sorted_sizes) > 2: size_to_type[sorted_sizes[2]] = 'H3'

    blocks = []
    for page_num, page in enumerate(doc):
        tables = page.find_tables()
        for tab in tables:
            if not tab.to_pandas().empty:
                markdown_table = tab.to_pandas().to_markdown(index=False)
                blocks.append({"page": page_num + 1, "type": "table", "content": markdown_table})

        text_blocks = page.get_text("dict", sort=True)["blocks"]
        for block in text_blocks:
            if "lines" in block and block['lines']:
                block_bbox = fitz.Rect(block['bbox'])
                if any(block_bbox.intersects(tab.bbox) for tab in tables):
                    continue

                spans = [span for line in block['lines'] for span in line['spans']]
                if not spans: continue
                
                block_text = " ".join(s['text'] for s in spans).strip()
                if not block_text: continue

                font_size = round(spans[0]['size'])
                is_bold = "bold" in spans[0]['font'].lower()

                block_type = size_to_type.get(font_size, "text")
                if block_type == "text" and is_bold and len(block_text.split()) < 20:
                    block_type = "H4"
                
                blocks.append({"page": page_num + 1, "type": block_type, "content": block_text})
    return blocks

def create_langchain_chunks(structured_blocks, pdf_filename, upload_date_str, bookmark_map=None):    # Nueva funcion
    """Crea los chunks finales para LangChain."""
    final_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    title_hierarchy = {} 
    for block in structured_blocks:
        page_number = block['page']
        current_hierarchy = {}
        if bookmark_map:
            current_hierarchy = bookmark_map.get(page_number, {})
        else:
            block_type = block.get('type', 'text')
            if block_type.startswith('H'):
                level = int(block_type[1])
                title_hierarchy[f'H{level}'] = block['content']
                for i in range(level + 1, 5):
                    title_hierarchy.pop(f'H{i}', None)
            current_hierarchy = title_hierarchy.copy()

        metadata = {
            "document_name_id": pdf_filename,
            "page_number": page_number,
            "upload_date": upload_date_str,
            "content_type": block['type'],
            "title_hierarchy": current_hierarchy
        }
        
        if block['type'] == 'text':
            sub_chunks = text_splitter.split_text(block['content'])
            for sub_chunk in sub_chunks:
                final_chunks.append(Document(page_content=sub_chunk, metadata=metadata))
        else: # H1, H2, H3, H4, table
             final_chunks.append(Document(page_content=block['content'], metadata=metadata))
                
    return final_chunks

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

def is_simple_question(question: str) -> bool:
    question = question.lower()
    keywords = ["quién", "qué", "cuándo", "dónde", "cuánto", "cómo", "hola"]
    exclusion_terms = ["documento", "pdf", "archivo", "tabla", "contenido"]
    return (
        len(question.split()) < 8 and
        any(word in question for word in keywords) and
        not any(term in question for term in exclusion_terms)
    )

def parse_model_response(response_text: str):
    """
    Divide la respuesta del modelo de forma más robusta,
    buscando los separadores con o sin asteriscos de Markdown.
    """
    main_answer = response_text
    additional_info = None
    separator_pattern = re.compile(r'\*\*?Información adicional\*\*?:\s*', re.IGNORECASE)
    match = separator_pattern.search(response_text)

    if match:
        separator = match.group(0)
        parts = response_text.split(separator, 1)
        main_answer_raw = parts[0]
        additional_info = parts[1].strip() if len(parts) > 1 else None
        title_pattern = re.compile(r'\*\*?Respuesta a tu pregunta\*\*?:\s*', re.IGNORECASE)
        title_match = title_pattern.search(main_answer_raw)
        
        if title_match:
            main_answer = main_answer_raw.split(title_match.group(0), 1)[1].strip()
        else:
            main_answer = main_answer_raw.strip()
    
    return main_answer, additional_info

class ChatRequest(BaseModel):
    question: str
    use_internet: bool = False  # valor por defecto si no se marca

from langchain.chains import LLMChain
from fastapi import HTTPException

@app.post("/ask")
async def ask_bot(req: ChatRequest):
    global suggested_question, memory

    actual_prompt = req.question.lower().strip()
    chat_history = memory.chat_memory

    # --- Regla Manual ---
    if any(frase in actual_prompt for frase in ["subir archivo", "subir archivos", "cómo subo", "cómo puedo subir", "dónde subo", "cómo cargar", "adjuntar documento"]):
        respuesta_manual = "Puedes subir los archivos desde la pestaña Documentos, en la parte superior de la aplicación."
        memory.save_context({"input": req.question}, {"output": respuesta_manual})
        return {
            "answer": respuesta_manual,
            "additional_info": None,  # <-- CAMBIO: Añadido para consistencia
            "suggested_question": None,
            "sources": [],
        }

    # --- Lógica principal (simple, RAG, internet) ---
    final_response = ""
    final_sources = []
    
    simple = is_simple_question(actual_prompt)
    if simple and not req.use_internet:
        simple_prompt = f"""Historial de conversación: {chat_history}\nPregunta: {actual_prompt}\nRespuesta:"""
        final_response = model.invoke(simple_prompt).content
        final_sources = []
    else:
        vector_retriever = vector_store.as_retriever(search_kwargs={'k': 6})
        context_docs = vector_retriever.get_relevant_documents(actual_prompt)
        context_text = "\n".join([doc.page_content for doc in context_docs])
        
        rag_prompt = PromptTemplate(
            template="""
            Eres un asistente de IA experto en definiciones normativas y conceptos técnicos.
            Sigue estrictamente el formato indicado a continuación para tu respuesta excepto cuando te hablen con lenguaje natural:

            **Respuesta a tu pregunta**: <una definición concisa, citando normativa textual si es relevante>
            **Información adicional**: <una explicación detallada y pedagógica del concepto, aportando contexto adicional>

            El tono debe ser profesional, claro y pedagógico.
            Además de usar los documentos proporcionados como contexto, debes saber que si no encuentras información suficiente, puedes apoyarte en búsquedas en Internet cuando esté habilitado por el sistema.
            Historial de conversación: {chat_history}
            Contexto: {context}
            Pregunta: {question}
            Respuesta:
            """,
            input_variables=["chat_history", "context", "question"]
        )
        llm_chain = LLMChain(llm=model, prompt=rag_prompt)
        rag_answer = llm_chain.run({"chat_history": chat_history, "context": context_text, "question": actual_prompt})
        sources = [doc.dict() for doc in context_docs]

        if req.use_internet or Sin_Informacion in rag_answer:
            search_results = search_tool.run(actual_prompt)
            internet_prompt = f"""Eres un asistente de IA. Basándote en el historial de la conversación y los siguientes resultados de una búsqueda en Internet, 
            responde a la "Pregunta nueva" del usuario de una forma amable y útil.
            Historial de la conversación: {chat_history}
            Resultados de búsqueda: "{search_results}"
            Pregunta nueva: {actual_prompt}
            Respuesta final:"""
            final_response = model.invoke(internet_prompt).content
            final_sources = []
        else:
            final_response = rag_answer
            final_sources = sources

    # =================================================================
    # CAMBIO CRUCIAL: USAMOS LA FUNCIÓN DE PARSEO AQUÍ
    # =================================================================
    
    # 1. Separamos la respuesta obtenida en dos partes
    parsed_answer, additional_info = parse_model_response(final_response)

    # 2. Guardamos solo la respuesta limpia en la memoria
    memory.save_context({"input": req.question}, {"output": parsed_answer})
    
    # (Opcional) Lógica para sugerir preguntas
    if '?' in parsed_answer:
        potential_question = parsed_answer.split('?')[-1].strip()
        if len(potential_question) > 5:
            suggested_question = potential_question
    else:
        suggested_question = None

    # 3. Devolvemos el JSON con los campos ya separados
    return {
        "answer": parsed_answer,
        "additional_info": additional_info,
        "suggested_question": suggested_question,
        "sources": final_sources,
    }



@app.get("/chat-history")
def get_chat_history():
    return {"history": [msg.content for msg in memory.chat_memory.messages]}

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

from fastapi import UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List

# 🔁 Asegúrate de tener esta variable definida al inicio de tu archivo:
canceled_uploads = {}  # {"nombre.pdf": True}


@app.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """
    Endpoint para subir archivos PDF.
    """
    all_final_chunks = []
    try:
        print("📥 Archivos recibidos:", [f.filename for f in files])

        # NUEVO: Obtenemos la fecha y hora actual UNA SOLA VEZ para todos los archivos de esta subida
        upload_date = datetime.now().strftime('%Y-%m-%d')
        print(f"Fecha de subida para estos archivos: {upload_date}")

        for uploaded_file in files:
            filename = uploaded_file.filename

            if not filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"El archivo {filename} no es un PDF.")

            file_content = await uploaded_file.read()
            print(f"Tamaño del archivo leído: {len(file_content)} bytes")

            # --- SUBIDA A GOOGLE DRIVE (Opcional) ---
            # drive_file_id = upload_file_to_drive(file_content, filename)
            # if not drive_file_id:
            #     print(f"⚠️  Advertencia: No se pudo subir el archivo '{filename}' a Google Drive.")
            
            uploaded_file.file.seek(0)

            structured_blocks = []
            bookmark_map = None

            with fitz.open(stream=file_content, filetype="pdf") as doc:
                if doc.get_toc(simple=False):
                    print(f"📄 Estrategia para '{filename}': Marcadores (alta precisión).")
                    bookmark_map = devolver_marcadores(doc)
                    structured_blocks = extract_raw_blocks(doc)
                else:
                    print(f"📄 Estrategia para '{filename}': Análisis de Layout (fallback).")
                    structured_blocks = extract_blocks_by_layout(doc)

            print(f"-> Se procesaron {len(structured_blocks)} bloques de '{filename}'.")

            # CAMBIO: Pasamos la fecha de subida al crear los chunks
            file_chunks = create_langchain_chunks(structured_blocks, filename, upload_date, bookmark_map)

            all_final_chunks.extend(file_chunks)
            canceled_uploads.pop(filename, None)

        if not all_final_chunks:
            return JSONResponse(content={"message": "No se pudo extraer contenido procesable de los archivos."}, status_code=400)

        print(f"\n📤 Subiendo un total de {len(all_final_chunks)} chunks a Qdrant...")
        crear_indice(collection_name=collection_name)
        
        global vector_store
        if vector_store is None:
            print("📌 Vector store no inicializado, creando uno nuevo...")
            vector_store = crear_vectorstore(collection_name)
        else:
            print("📌 Vector store ya inicializado.")

        # Subimos todos los chunks en un solo lote para mayor eficiencia
        vector_store.add_documents(all_final_chunks)
        print(f"✅ {len(all_final_chunks)} chunks subidos a Qdrant con éxito.")
        
        return JSONResponse(
            content={"message": f"{len(all_final_chunks)} fragmentos de {len(files)} archivo(s) cargados correctamente."},
            status_code=200
        )

    except Exception as e:
        print(f"❌ Error en /upload: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error al procesar los archivos: {e}")


def crear_vectorstore(collection_name: str):
    return QdrantVectorStore(
        client=QdrantClient(url=url, api_key=api_key),
        collection_name=collection_name,
        embedding=embeddings,
        retrieval_mode=RetrievalMode.DENSE
    )

@app.delete("/delete")
async def eliminar_pdf_qdrant(collection_name: str, pdf_nombre: str):
    print(f"📥 DELETE recibido para: {pdf_nombre} en colección: {collection_name}")
    canceled_uploads[pdf_nombre] = True
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
        client = QdrantClient(url=url, api_key=api_key)
        scrolled_points, _ = client.scroll(
            collection_name=collection_name,
            limit=10000,
            with_payload=True
        )

        document_info = {}
        for point in scrolled_points:
            if point.payload and "metadata" in point.payload:
                metadata = point.payload["metadata"]
                doc_name = metadata.get("document_name_id")
                
                if doc_name and doc_name not in document_info:
                    upload_date = metadata.get("upload_date", "Fecha no disponible")
                    document_info[doc_name] = {
                        "name": doc_name,
                        "upload_date": upload_date
                    }

        sorted_documents = sorted(list(document_info.values()), key=lambda x: x['name'])
        return sorted_documents
    except Exception as e:
        print(f"Ha ocurrido un error en /documentos_unicos: {e}")
        return []


@app.websocket("/ws/progreso")
async def websocket_progreso(websocket: WebSocket):
    await websocket.accept()
    try:
        # Espera el nombre del archivo
        filename = await websocket.receive_text()
        print(f"📡 Cliente conectado esperando progreso para: {filename}")
        active_connections[filename] = websocket
    except Exception as e:
        print(f"❌ WebSocket cerrado: {e}")

async def enviar_progreso(filename: str, porcentaje: int):
    ws = active_connections.get(filename)
    if ws:
        try:
            await ws.send_json({"filename": filename, "progress": porcentaje})
        except Exception as e:
            print(f"Error al enviar progreso por WebSocket: {e}")



from app.auth_utils import authenticate_user, register_user
from passlib.context import CryptContext
from app.database import fake_users_db


class LoginData(BaseModel):
    username: str
    password: str

# Seguridad
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


from fastapi.responses import JSONResponse
from fastapi import status  # 👈 importante

@app.post("/login")
async def login(data: LoginData):
    if authenticate_user(data.username, data.password):
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"success": True, "message": f"Bienvenido {data.username}"}
        )
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={"success": False, "message": "Usuario o contraseña incorrectos"}
    )


@app.post("/register")
async def register(data: LoginData):
    if register_user(data.username, data.password):
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={"success": True, "message": "Usuario registrado correctamente"}
        )
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={"success": False, "message": "El usuario ya existe"}
    )

    
def user_exists(username: str):
    return username in fake_users_db

def add_user(username: str, password: str):
    fake_users_db[username] = password    