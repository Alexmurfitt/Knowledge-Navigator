import json
from ragas import evaluate

# Cargar historial de interacciones desde archivo JSON
with open("scripts/historial.json", "r", encoding="utf-8") as f:
    historial = json.load(f)

# Extraer datos: preguntas, respuestas y contextos
dataset = []

for entry in historial:
    pregunta = entry.get("pregunta")
    respuesta = entry.get("respuesta")
    contexto = entry.get("contexto")  # Asegúrate de guardar contexto como lista de dicts con 'content' y 'source'

    if pregunta and respuesta and contexto:
        dataset.append({
            "question": pregunta,
            "answer": respuesta,
            "context": contexto
        })

# Validación mínima
if not dataset:
    print("❌ No se encontraron datos con pregunta, respuesta y contexto. Verifica historial.json")
    exit()

# Evaluar
print("📊 Ejecutando evaluación automática con RAGAS...")
metrics = evaluate(dataset)

# Mostrar resultados
print("\n✅ Resultados de evaluación:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
