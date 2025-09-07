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

# Al principio de main.py
import re

load_dotenv()

google_api_key=os.getenv("GOOGLE-API-KEY")
url=os.getenv("QDRANT-URL")
api_key=os.getenv("QDRANT-API-KEY")
search_api_key = os.getenv("GOOGLE-SEARCH-API-KEY")
google_cse_id = os.getenv("GOOGLE-SEARCH-ID")
collection_name = os.getenv("COLLECTION-NAME")
def mostrar_documentos_unicos():
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

        document_names = set()  #En vez de un diccionario o una lista pongo un set ya que almacena documentos unicos
        for point in scrolled_points:
            if point.payload and "metadata" in point.payload:   #Si hay payload y metadata esta dentro de payload (Lo de metadata es dentro de los metadatos hay un campo llamado metadata y dentro estan el resto de variables)
                # 2. Buscamos 'document_name_id' DENTRO de 'metadata'
                metadata_dict = point.payload["metadata"]
                if "document_name_id" in metadata_dict: #Si dentro de metadata esta document_name_id
                    document_names.add(metadata_dict["document_name_id"])   #Lo añadimos al set
        print(f"Documentos únicos encontrados: {document_names}")

        return sorted(list(document_names))


    except Exception as e:
        print(f"Ha ocurrido un error: {e}")
        return []

documentos_unicos = mostrar_documentos_unicos()
