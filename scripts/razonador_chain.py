# scripts/razonador_chain.py
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableSequence

# Modelo LLM
llm = ChatOllama(model="llama3", temperature=0.1)

# Prompt estructurado
prompt = PromptTemplate.from_template("""
Actúa como un asistente experto que mejora preguntas del usuario en temas de ética de la IA.

1. Refina la pregunta para que sea más clara y precisa.
2. Añade entre 1 y 3 frases complementarias que amplíen la perspectiva, sin repetir la pregunta ni formular nuevas.

Devuelve la respuesta en formato JSON con las claves:
- "pregunta_refinada": string
- "respuestas_adicionales": lista de frases

Usuario: {pregunta_usuario}
""")

# Parser JSON (genérico)
parser = JsonOutputParser()

# Cadena completa
razonador_chain = prompt | llm | parser
