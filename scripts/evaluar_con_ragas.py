from ragas import evaluate

# Preparar tus datos: preguntas, respuestas y contextos
questions = ["¿Qué problemas plantea la falta de estándares cruzados en el uso de la ciencia de datos en el sector humanitario?"]
answers = ["La falta de estándares cruzados puede generar problemas como la dificultad para comparar datos y la falta de reproducibilidad."]
contexts = [
    [
        {"source": "Documento 1", "content": "La falta de estándares cruzados puede llevar a inconsistencias en la toma de decisiones."}
    ]
]

# Agrupar los datos en un formato adecuado para RAGAS
dataset = [
    {
        "question": q,
        "answer": a,
        "context": c
    }
    for q, a, c in zip(questions, answers, contexts)
]

# Evaluar con RAGAS
metrics = evaluate(dataset)

# Mostrar resultados
print(metrics)
