from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_ollama import ChatOllama

# 📌 Cargar modelo
llm = ChatOllama(model="llama3", temperature=0.1)

# 🧠 Prompt de razonamiento paso a paso (Chain of Thought)
prompt = PromptTemplate(
    input_variables=["pregunta_usuario"],
    template="""
Eres un asistente de IA experto en ética y transparencia de sistemas inteligentes. Razona paso a paso y genera una respuesta útil.

Pregunta del usuario:
{pregunta_usuario}

Razonamiento detallado:
"""
)

# 🔁 Crear secuencia moderna: prompt | modelo
razonador_cot_chain = prompt | llm

# 🚀 Función que invoca el razonador
def razonamiento_cot(pregunta_usuario):
    respuesta = razonador_cot_chain.invoke({"pregunta_usuario": pregunta_usuario})
    return respuesta.content  # ← devolver solo el texto de la respuesta
