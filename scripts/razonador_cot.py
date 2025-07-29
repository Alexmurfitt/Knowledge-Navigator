# ======================================
# razonador_cot.py (con historial, formato estructurado y soporte web)
# ======================================
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# =============================================================
# FUNCIÓN PRINCIPAL CON SOPORTE DE HISTORIAL
# =============================================================
def generar_respuesta(pregunta: str, contexto: str = "", historial: str = "", temperatura: float = 0.2, modo_conciso: bool = False) -> str:
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
            "El tono debe ser profesional, claro y pedagógico. Si hay historial conversacional, respeta su coherencia."
        )

    # Construir el mensaje humano con historial (si lo hay)
    if historial.strip():
        human_prompt = "Historial previo:\n{historial}\n\nContexto relevante:\n{contexto}\n\nPregunta: {pregunta}"
        input_data = {"pregunta": pregunta, "contexto": contexto, "historial": historial}
    elif contexto.strip():
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
        texto_respuesta = respuesta_modelo.content if hasattr(respuesta_modelo, "content") else str(respuesta_modelo)
        return texto_respuesta
    except Exception as e:
        return f"Error al generar respuesta con el modelo: {e}"


# =============================================================
# FUNCIÓN PARA RESPUESTA A PARTIR DE FRAGMENTOS WEB
# =============================================================
def generar_respuesta_web(pregunta: str, contexto: str) -> str:
    prompt_web = ChatPromptTemplate.from_template("""
A continuación se presentan fragmentos obtenidos en una búsqueda web sobre la siguiente pregunta del usuario:

Pregunta: {pregunta}

Fragmentos extraídos:
{contexto}

Tu tarea es generar una respuesta clara, precisa y basada exclusivamente en estos fragmentos. 
No inventes ni extrapoles información. No expliques cómo funciona un modelo de lenguaje.

Responde directamente en el siguiente formato, **sin duplicar encabezados**:

📘 Respuesta principal
<respuesta directa y verificable basada en los fragmentos>

💡 Información adicional
<añade contexto útil si aporta valor; si no lo hay, indícalo explícitamente>
""")

    try:
        modelo = init_chat_model("gemini-2.5-flash", model_provider="google_genai", temperature=0)
        cadena = prompt_web | modelo
        resultado = cadena.invoke({"pregunta": pregunta, "contexto": contexto})
        return resultado.content
    except Exception as e:
        return f"Error al generar respuesta web: {e}"
