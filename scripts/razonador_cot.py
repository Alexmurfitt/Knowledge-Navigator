# =============================================================
# razonador_cot.py — Versión definitiva con historial conversacional y respuesta pedagógica o concisa
# =============================================================

from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# =============================================================
# FUNCIÓN PRINCIPAL: RESPUESTA A PARTIR DE CONTEXTO (local)
# =============================================================
def generar_respuesta(pregunta: str, contexto: str = "", historial: str = "", temperatura: float = 0.2, modo_conciso: bool = False) -> str:
    # SYSTEM PROMPT
    system_prompt = (
        "Eres un asistente conversacional que responde de forma breve, clara y precisa, sin explicaciones." if modo_conciso else
        "Eres un asistente experto que responde con precisión y pedagogía. Usa el siguiente formato:\n\n"
        "Definición textual o normativa: <respuesta principal>\n"
        "Explicación ampliada: <explicación complementaria y contextual>\n\n"
        "Siempre responde en español, con tono profesional y accesible."
    )

    # HUMAN PROMPT
    if historial.strip():
        human_prompt = (
            "Esta es una conversación continua.\n\n"
            "Historial previo:\n{historial}\n\n"
            "Contexto relevante:\n{contexto}\n\n"
            "Pregunta actual:\n{pregunta}"
        )
        input_data = {"pregunta": pregunta, "contexto": contexto, "historial": historial}
    elif contexto.strip():
        human_prompt = "Contexto relevante:\n{contexto}\n\nPregunta: {pregunta}"
        input_data = {"pregunta": pregunta, "contexto": contexto}
    else:
        human_prompt = "Pregunta: {pregunta}"
        input_data = {"pregunta": pregunta}

    try:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(human_prompt)
        ])
        modelo = init_chat_model("gemini-2.5-flash", model_provider="google_genai", temperature=temperatura)
        respuesta = (prompt | modelo).invoke(input_data)
        return respuesta.content if hasattr(respuesta, "content") else str(respuesta)
    except Exception as e:
        return f"❌ Error al generar respuesta: {e}"

# =============================================================
# FUNCIÓN SECUNDARIA: RESPUESTA A PARTIR DE FRAGMENTOS WEB
# =============================================================
def generar_respuesta_web(pregunta: str, contexto: str) -> str:
    prompt_web = ChatPromptTemplate.from_template(
        """
        Has recibido resultados de una búsqueda web sobre la siguiente pregunta:

        🧠 Pregunta del usuario:
        {pregunta}

        🌐 Fragmentos encontrados:
        {contexto}

        Redacta una respuesta clara y precisa basada SOLO en los fragmentos anteriores.
        No inventes información. Usa este formato:

        📘 Respuesta principal
        <respuesta>

        💡 Información adicional
        <si hay algo útil que añadir, escríbelo; si no, indica que no hay información adicional>
        """
    )
    try:
        modelo = init_chat_model("gemini-2.5-flash", model_provider="google_genai", temperature=0)
        respuesta = (prompt_web | modelo).invoke({"pregunta": pregunta, "contexto": contexto})
        return respuesta.content if hasattr(respuesta, "content") else str(respuesta)
    except Exception as e:
        return f"❌ Error al generar respuesta web: {e}"
