# from langchain import RunnableSequence
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


def generar_respuesta(pregunta: str, contexto: str = "", temperatura: float = 0.2, modo_conciso: bool = False) -> str:

    """
    Genera una respuesta a la pregunta dada, utilizando opcionalmente un contexto proporcionado,
    formateando la salida en dos secciones: "Definición textual o normativa" y "Explicación ampliada".
    Retorna el contenido generado por el modelo como una cadena de texto.
    """
    # Mensaje de sistema con instrucciones sobre el formato de la respuesta
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

    # Mensaje del usuario, incluyendo el contexto si se proporciona
    if contexto and contexto.strip():
        human_prompt = "Contexto relevante:\n{contexto}\n\nPregunta: {pregunta}"
        input_data = {"pregunta": pregunta, "contexto": contexto}
    else:
        human_prompt = "Pregunta: {pregunta}"
        input_data = {"pregunta": pregunta}

    try:
        # Crear la plantilla de prompt de chat con mensajes de sistema y de usuario
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(human_prompt)
        ])
        # Inicializar el modelo de chat Gemini 2.5 Flash (Google GenAI) con la temperatura especificada
        modelo = init_chat_model("gemini-2.5-flash", model_provider="google_genai", temperature=temperatura)
        # Componer la secuencia de ejecución: prompt -> modelo
        cadena = prompt_template | modelo
        respuesta_modelo = cadena.invoke(input_data)
        # Extraer el texto de la respuesta (puede ser str o objeto Message)
        texto_respuesta = respuesta_modelo if isinstance(respuesta_modelo, str) else str(respuesta_modelo)
        return texto_respuesta
    except Exception as e:
        # En caso de error durante la generación, devolver un mensaje de error descriptivo
        return f"Error al generar respuesta con el modelo: {e}"
