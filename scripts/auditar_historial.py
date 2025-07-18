import os
import json
from collections import Counter

ruta = os.path.join(os.path.dirname(__file__), "historial.json")

with open(ruta, "r", encoding="utf-8") as f:
    historial = json.load(f)

faltantes = []
for i, item in enumerate(historial):
    problemas = []
    if "pregunta" not in item:
        problemas.append("pregunta")
    if "respuesta_rag" not in item:
        problemas.append("respuesta_rag")
    if "contexto" not in item or not isinstance(item["contexto"], list) or not item["contexto"]:
        problemas.append("contexto vacío o inválido")
    if problemas:
        faltantes.append((i, problemas))

total = len(historial)
validos = total - len(faltantes)

print(f"🔍 Total de entradas en historial: {total}")
print(f"✅ Entradas válidas: {validos}")
print(f"❌ Entradas con errores: {len(faltantes)}\n")

for i, errores in faltantes:
    print(f" - Entrada #{i+1} → Faltan: {', '.join(errores)}")
