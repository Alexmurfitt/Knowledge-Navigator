import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient

# 1. Cargar credenciales desde .env
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "knowledge_navigator"

# 2. Inicializar modelo LLM de Ollama
llm = ChatOllama(model="llama3", temperature=0.3)

# 3. Inicializar modelo de embeddings (debe coincidir con el de indexación)
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

# 4. Conectar con Qdrant Cloud
client = QdrantClient(url=QDRANT_URL, api_key=API_KEY, prefer_grpc=False)
vectorstore = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embedding_model
)

# 5. Crear retriever y cadena RAG con trazabilidad
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 6. Función de respuesta con fuentes
def responder(pregunta):
    print(f"\n❓ Pregunta: {pregunta}")
    respuesta = qa_chain(pregunta)
    print(f"\n🧠 Respuesta:\n{respuesta['result']}")
    print("\n📚 Fuentes:")
    for doc in respuesta["source_documents"]:
        fuente = doc.metadata.get("source", "desconocido")
        pagina = doc.metadata.get("page", "N/A")
        print(f" - {fuente} (página {pagina})")

# 7. Interfaz de usuario
if __name__ == "__main__":
    print("🧪 Knowledge Navigator – Consulta a la base documental")
    while True:
        pregunta = input("\n🔎 Introduce tu pregunta (o 'salir'): ")
        if pregunta.strip().lower() in ["salir", "exit", "q"]:
            break
        responder(pregunta)
