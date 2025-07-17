# scripts/razonador_cot.py

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama

# 🧠 Inicializa el modelo local
llm = ChatOllama(model="llama3", temperature=0.1)

# 📄 Prompt con razonamiento paso a paso
prompt = PromptTemplate.from_template("""
Eres un agente especializado en razonamiento avanzado, con la misión de proporcionar respuestas absolutamente precisas, fiables y enriquecidas.

Cuando un usuario te haga una pregunta, no debes limitarte a responderla directamente. En su lugar, debes realizar un proceso interno de razonamiento que garantice la máxima calidad informativa. Para ello, sigue estos pasos con rigor:

1. Formula exactamente 2 auto-preguntas **estratégicas y altamente relevantes** que te ayuden a enriquecer y complementar la respuesta.
2. Responde esas auto-preguntas de forma **clara, precisa y basada en el mejor conocimiento disponible**.
3. Redacta una **respuesta principal** que sea **lo más útil, completa, rigurosa y fiable posible**.
4. Presenta las auto-preguntas y sus respuestas de forma separada, como **información adicional complementaria**.

Responde siguiendo **estrictamente** este formato:

──────────────────────────────────────────────
🔹 RESPUESTA PRINCIPAL:
[respuesta directa, clara y completa a la pregunta del usuario]

🔸 INFORMACIÓN ADICIONAL:
• [auto-pregunta 1]: [respuesta breve, precisa y relevante]  
• [auto-pregunta 2]: [respuesta breve, precisa y relevante]  
(Si lo consideras útil, puedes incluir más auto-preguntas)
──────────────────────────────────────────────

Usuario: {pregunta_usuario}
""")


# 🔗 Cadena LangChain lista para usarse
chain = LLMChain(llm=llm, prompt=prompt)

# ✅ Función exportable para usar en otros scripts
def razonamiento_cot(pregunta_usuario):
    return chain.run(pregunta_usuario=pregunta_usuario)

# 🚀 Interfaz de prueba por consola
if __name__ == "__main__":
    print("\n🤖 AGENTE INTELIGENTE CON AUTO-PREGUNTAS (Chain-of-Thought)\n")
    while True:
        pregunta = input("🔎 Pregunta del usuario (o 'salir'): ")
        if pregunta.lower() in ["salir", "exit", "quit"]:
            print("👋 Hasta pronto.\n")
            break
        try:
            resultado = razonamiento_cot(pregunta)
            print("\n🧠 Respuesta enriquecida:\n")
            print(resultado)
            print("\n" + "=" * 80 + "\n")
        except Exception as e:
            print(f"⚠️ Error al generar la respuesta: {e}")
