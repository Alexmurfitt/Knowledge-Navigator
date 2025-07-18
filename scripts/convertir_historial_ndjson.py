import json

archivo_origen = "historial.json"
archivo_destino = "historial_convertido.json"

# Leer línea por línea y cargar cada objeto
with open(archivo_origen, "r", encoding="utf-8") as f:
    lineas = f.readlines()
    datos = [json.loads(line) for line in lineas]

# Guardar como lista JSON válida
with open(archivo_destino, "w", encoding="utf-8") as f:
    json.dump(datos, f, ensure_ascii=False, indent=2)

print(f"✅ Archivo convertido correctamente: {archivo_destino}")
