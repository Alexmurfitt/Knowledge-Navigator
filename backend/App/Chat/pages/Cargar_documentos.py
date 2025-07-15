import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore 
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
import tempfile

load_dotenv()


def extract_text_from_pdf(pdf_file):
    """Extrae el texto de un archivo PDF subido."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        os.remove(tmp_file_path)
        return documents
    except Exception as e:
        st.error(f"Error al leer el PDF: {e}")
        return []
# Aqui es donde separamos el texto, para mas precision deberia bajar los chunck un poco
def split_text_into_chunks(documents):
    """Divide los documentos extraídos en trozos (chunks) más pequeños."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

# --- Interfaz de Usuario de Streamlit ---

st.set_page_config(page_title="Cargar Documentos", page_icon="📄")
st.title("Cargar PDFs a la Base de Conocimiento ")

st.markdown("""
Sube uno o varios archivos PDF. El contenido se procesará y almacenará en Qdrant. Si la colección no existe, se creará automáticamente.
""")

uploaded_files = st.file_uploader(
    "Elige tus archivos PDF",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Procesar y Cargar a Qdrant"):
        with st.spinner("Procesando archivos... Este proceso puede tardar un momento."):
            
            # 1. Extraer y dividir el texto en chunks
            all_docs = []
            for pdf_file in uploaded_files:
                st.write(f"Leyendo `{pdf_file.name}`...")
                docs = extract_text_from_pdf(pdf_file)
                all_docs.extend(docs)
            
            chunks = split_text_into_chunks(all_docs)

            # 2. Usar QdrantVectorStore.from_documents para crear la colección y subir los datos

            try:
                st.write("Conectando con Qdrant y creando vectores...")
                
                # Inicializamos el modelo de embeddings
                embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

                # Usamos .from_documents para crear la colección y añadir los documentos en un solo paso
                QdrantVectorStore.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    url=os.getenv("qdrant_url"),
                    api_key=os.getenv("qdrant_api_key"),
                    collection_name="Prueba_RAG", # Tiene que ser le mismo nombre que tengo en Qdrant
                    force_recreate=False # Poner en False para no borrarla cada vez
                )
                st.success(f"¡Éxito! Se han procesado y añadido {len(chunks)} fragmentos de texto a la colección 'Prueba_RAG'.")

            except Exception as e:
                st.error(f"Ha ocurrido un error al conectar o subir a Qdrant: {e}")