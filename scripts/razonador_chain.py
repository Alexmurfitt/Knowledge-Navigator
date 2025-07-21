# scripts/razonador_chain.py
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableSequence

# Parser con instrucciones de formato JSON
parser = JsonOutputParser()

# Prompt actualizado
prompt = PromptTemplate.from_template(f"""
Actúa como un asistente especializado en análisis de documentación empresarial y mejora de consultas de usuario.

Tu tarea es:

1. Reformular la pregunta del usuario para que sea más clara, precisa y adecuada al análisis de documentación.
2. Añadir entre 1 y 3 frases complementarias que amplíen la perspectiva o enfoquen mejor la búsqueda, sin repetir ni reformular la pregunta.

Devuelve la respuesta en formato JSON con las claves:
- "pregunta_refinada": string
- "respuestas_adicionales": lista de frases

{parser.get_format_instructions()}

Usuario: {{pregunta_usuario}}
""")

# Modelo LLM
llm = ChatOllama(model="llama3", temperature=0.1)

# Cadena ensamblada
razonador_chain = prompt | llm | parser
