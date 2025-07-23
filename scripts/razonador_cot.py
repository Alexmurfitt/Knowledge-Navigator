from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
# 📌 Cargar modelo
llm = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    api_key=google_api_key,
    temperature=0.1
)


# 🧠 Prompt de razonamiento paso a paso (Chain of Thought)

prompt = PromptTemplate(
    input_variables=["pregunta_usuario"],
    template="""Actúa como un asistente profesional experto en análisis e interpretación de documentación empresarial. 
Estás entrenado para ayudar a los usuarios no solo a obtener la información solicitada, sino también a comprenderla,
proporcionando respuestas estructuradas, claras y pedagógicas. Tu respuesta debe contener las siguientes secciones:

1. 📘 **Definición textual o normativa** (si aplica): Si se hace referencia a un fragmento o concepto específico, ofrece primero su definición exacta tal como aparece en los documentos.

2. 🧠 **Explicación ampliada**: Interpreta y aclara el contenido en lenguaje accesible. Ayuda al usuario a comprender qué significa realmente, con precisión, rigor y claridad.

3. 🔄 **Pregunta de seguimiento**: Formula una pregunta útil, coherente y contextualizada para continuar la conversación. Debe permitir profundizar, enlazar temas relacionados o comprobar la retención del historial.

Tu objetivo es interactuar de forma natural y cercana, como lo haría un operador humano, manteniendo siempre un tono claro, respetuoso y experto.

---
⚠️ Solo puedes generar una respuesta si el fragmento ha sido encontrado en los documentos.

Pregunta inicial del usuario:
{pregunta_usuario}
"""
)


# 🔁 Crear secuencia moderna: prompt | modelo
razonador_cot_chain = prompt | llm

# 🚀 Función que invoca el razonador
def razonamiento_cot(pregunta_usuario):
    respuesta = razonador_cot_chain.invoke({"pregunta_usuario": pregunta_usuario})
    print(f"esta es la respuesta de razonamiento_cot_limpio:{respuesta.content}")
    return respuesta.content  # ← devolver solo el texto de la respuesta
