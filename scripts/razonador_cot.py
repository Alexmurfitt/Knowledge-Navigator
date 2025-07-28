# ======================================
# razonador_cot.py (con soporte web y formato estructurado)
# ======================================
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Función para respuestas normativas o técnicas

def generar_respuesta(pregunta: str, contexto: str = "", temperatura: float = 0.2, modo_conciso: bool = False) -> str:
    if modo_conciso:
        system_prompt = (
            "Eres un asistente de IA que responde de forma breve, precisa y objetiva. "
            "Responde **solo en español** y proporciona una sola frase clara con el dato exacto que responde a la pregunta, "
            "sin explicaciones adicionales ni justificaciones. No uses encabezados ni subtítulos. "
            "Ejemplo de formato: 'Hoy es viernes 25 de julio de 2025.'"
        )
    else:
        system_prompt = (
            "Eres un asistente de IA experto en definiciones normativas y conceptos técnicos. "
            "Responde **solo en español** y sigue el formato indicado a continuación para tu respuesta:\n\n"
            "Definición textual o normativa: <una definición concisa, citando normativa textual si es relevante>\n"
            "Explicación ampliada: <una explicación detallada y pedagógica del concepto, aportando contexto adicional>\n\n"
            "El tono debe ser profesional, claro y pedagógico."
        )

    if contexto and contexto.strip():
        human_prompt = "Contexto relevante:\n{contexto}\n\nPregunta: {pregunta}"
        input_data = {"pregunta": pregunta, "contexto": contexto}
    else:
        human_prompt = "Pregunta: {pregunta}"
        input_data = {"pregunta": pregunta}

    try:
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(human_prompt)
        ])
        modelo = init_chat_model("gemini-2.5-flash", model_provider="google_genai", temperature=temperatura)
        cadena = prompt_template | modelo
        respuesta_modelo = cadena.invoke(input_data)
        texto_respuesta = respuesta_modelo if isinstance(respuesta_modelo, str) else str(respuesta_modelo)
        return texto_respuesta
    except Exception as e:
        return f"Error al generar respuesta con el modelo: {e}"


# Función específica para respuestas a partir de snippets web

def generar_respuesta_web(pregunta: str, contexto: str) -> str:
    prompt_web = ChatPromptTemplate.from_template("""
A continuación se presentan fragmentos de resultados de búsqueda web relacionados con la siguiente pregunta:

Pregunta: {pregunta}

Fragmentos web:
{contexto}

Tu tarea es generar una respuesta clara, objetiva, precisa y actual basada exclusivamente en estos fragmentos. 
No inventes, no generalices, no expliques cómo funciona un modelo de lenguaje. 
Limítate a resumir o sintetizar lo que indican los fragmentos y proporciona información verificable. Si los fragmentos son vagos, dilo.

Incluye:
📘 Respuesta principal
💡 Información adicional (si procede)
""")

    try:
        modelo = init_chat_model("gemini-2.5-flash", model_provider="google_genai", temperature=0)
        cadena = prompt_web | modelo
        resultado = cadena.invoke({"pregunta": pregunta, "contexto": contexto})
        return resultado.content
    except Exception as e:
        return f"Error al generar respuesta web: {e}"