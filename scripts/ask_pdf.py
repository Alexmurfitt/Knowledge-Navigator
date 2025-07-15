import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# Configuración
VECTOR_DB_DIR = "../data/vectorstore"

# Paso 1: Cargar la base de vectores FAISS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(VECTOR_DB_DIR, embeddings, allow_dangerous_deserialization=True)

# Paso 2: Crear LLM local con Ollama
llm = Ollama(model="llama3")

# Paso 3: Crear cadena de QA con recuperación
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# Paso 4: Bucle de preguntas
while True:
    query = input("\n🔍 Tu pregunta (o 'salir'): ")
    if query.lower() == "salir":
        break
    result = qa_chain({"query": query})
    print(f"\n📘 Respuesta: {result['result']}")
    for doc in result["source_documents"]:
        print(f" - Fuente: {doc.metadata.get('source', 'Desconocido')}")

