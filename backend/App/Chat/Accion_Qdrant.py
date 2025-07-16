import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore 
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models  #PAra poder acceder a la base de datos y eliminar points
import os
import tempfile
import time
load_dotenv()
google_api_key = os.getenv("GOOGLE-API-KEY")    #Seleecionamos la apikey del modelo
qdrant_url = os.getenv("QDRANT-URL")
qdrant_api_key = os.getenv("QDRANT-API-KEY")
collection_name = os.getenv("COLLECTION-NAME")

def mostrar_documentos_unicos(collection_name: str):

    try:
        client = QdrantClient(
            url=qdrant_url, 
            api_key=qdrant_api_key
        )
        '''
        Documentacion de Qdrant para scrollear
        client.scroll(
    collection_name="{collection_name}",
    scroll_filter=models.Filter(
        must=[
            models.FieldCondition(key="color", match=models.MatchValue(value="red")),
        ]
    ),
    limit=1,
    with_payload=True,
    with_vectors=False,
)
        '''

        scrolled_points, llamada= client.scroll( # La "llamada" es porque client.scroll devuelve (puntos, llamada) y la llamada es para la siguiente llamda (no se necesita para nada), es decir me devuelve una tupla de 2 valores, Puntos y llamada
            collection_name=collection_name,
            limit=1000,    # El limit es la cantidad de endpoint que quiero ver
            with_payload=True   #Awui esta incluyendo los metadatos
        )

        '''
        Si solo pongo Scrolle_points sin la ", llamada" entonces el primer elemento (scrolled_points[0]) seria toda la informacion, y el segundo elemento(scrolled_points[1]) seria llamada
        Como solo queremos iterar sobre los points ponemos scrolled_points, llamada y luego haremos un for para cada point 

        '''

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

        # if document_names:
        #     print(f"Numero de documentos importados : {len(document_names)}")
        #     print(f"Documentos unicos en la coleccion '{collection_name}':")
        #     for name in sorted(list(document_names)):
        #         print(f"- {name}")      
        # else:
        #     print(f"No se encontraron documentos con el metadato 'document_name_id' en la colección '{collection_name}'.")

        return sorted(list(document_names))


    except Exception as e:
        print(f"Ha ocurrido un error: {e}")
        return []

# --- Llama a la función con tu colección ---
def eliminar_pdf_qdrant(collection_name: str, pdf_nombre: str):
   
    # Conexión con el cliente de Qdrant
    client = QdrantClient(
        url=qdrant_url, 
        api_key=qdrant_api_key
    )

    '''
    Documentacion de Qdrant para eliminar
    client.delete(
    collection_name="{collection_name}",
    points_selector=models.FilterSelector(
        filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="color",
                    match=models.MatchValue(value="red"),
                ),
            ],
        )
    ),
)
    '''

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
        return True
    
    except Exception as e:
        print(f"Error al eliminar los datos de Qdrant: {e}")
        return False
    
def crear_indice(collection_name : str):
    try:
        client = QdrantClient(
            url=qdrant_url, 
            api_key=qdrant_api_key
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
