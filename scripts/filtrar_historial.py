import os
import json

# 📁 Ruta original y nueva
BASE_DIR = os.path.dirname(__file__)
ruta_origen = os.path.join(BASE_DIR, "historial.json")
ruta_destino = os.path.join(BASE_DIR, "historial_filtrado.json")

# 📥 Leer historial completo
with open(ruta_origen, "r", encoding="utf-8") as f:
    historial = json.load(f)

# ✅ Filtrar entradas completas
entradas_validas = [
    item for item in historial
    if all(k in item for k in ("pregunta", "respuesta_rag", "contexto"))
       and isinstance(item["contexto"], list)
       and item["contexto"]
]

# 💾 Guardar en nuevo archivo si hay válidas
if entradas_validas:
    with open(ruta_destino, "w", encoding="utf-8") as f:
        json.dump(entradas_validas, f, ensure_ascii=False, indent=2)
    print(f"✅ Se guardaron {len(entradas_validas)} entradas válidas en: {ruta_destino}")
else:
    print("⚠️ No se encontraron entradas válidas para guardar.")
