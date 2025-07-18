import os
import json
import pandas as pd
from pprint import pprint
from dotenv import load_dotenv

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_utilization
)

# 📥 Cargar variables de entorno (API keys)
load_dotenv()

# 📍 Ruta del historial JSON reconstruido
ruta = os.path.join(os.path.dirname(__file__), "historial.json")

# 📥 Cargar historial
with open(ruta, "r", encoding="utf-8") as f:
    historial = json.load(f)

# ✅ Filtrar entradas válidas
entradas_validas = [
    item for item in historial
    if all(k in item for k in ("pregunta", "respuesta_rag", "contexto"))
    and isinstance(item["contexto"], list)
    and item["contexto"]
]

if not entradas_validas:
    raise ValueError("❌ No se encontraron entradas válidas para evaluar.")

# 🧪 Preparar dataset para RAGAS
dataset = Dataset.from_list([
    {
        "question": item["pregunta"],
        "answer": item["respuesta_rag"],
        "contexts": item["contexto"]
    }
    for item in entradas_validas
])

# 🧪 Ejecutar evaluación con métricas compatibles (sin ground_truth)
print("🔍 Evaluando respuestas con RAGAS...\n")

resultados = evaluate(
    dataset=dataset,
    metrics=[
        context_utilization,
        faithfulness,
        answer_relevancy
    ]
)

# 📊 Mostrar resultados
print("✅ Métricas de evaluación:")
pprint(resultados)

# 💾 Guardar resultados como CSV
df = pd.DataFrame([resultados])
csv_path = os.path.join(os.path.dirname(__file__), "evaluacion_ragas.csv")
df.to_csv(csv_path, index=False)
print(f"\n📁 Resultados guardados en: {csv_path}")
