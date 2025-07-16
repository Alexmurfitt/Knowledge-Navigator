CÓMO USAR EL CHAT

1) En el .end tenemos que tener todos los mismos nombres de variables:
GOOGLE-API-KEY = ""
LANGSMITH_TRACING =""
LANGSMITH_API_KEY = ""
QDRANT-URL = ""
QDRANT-API-KEY = ""
COLLECTION-NAME = ""

2) Los archivos que subo estan en backend - App - Chat. (As)
    Archivos:

    Scripts_Chat.py
    Accion_Qdrant.py
    requirements_chat.txt
    pages/
         -   Cargar_documentos.py
         -   Eliminar_Documentos.py

3) Lo primero es crear un entonrno

4) instalar el requirements_chat.txt (Cuidado con la libreria pywin32 que es de window)

    pip install -r requirements_chat.txt

5) Ejecutar el streamlit para usarlo (Tener en cuenta en que carpeta estas)

    streamlit run Scripts_Chat.py