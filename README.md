# Knowledge Navigator

Asistente inteligente multimodal con IA generativa, razonamiento autónomo y escalabilidad profesional.

# Enlace documento Google Drive:

https://docs.google.com/document/d/1FdvMoNgEYl9LZxdObe8eGxLVGUH_rx3RiR9JgyT_tO8/edit?tab=t.js3ami4q4gx


## 🌟 Descripción del Proyecto

**AI Knowledge Navigator** es un sistema cognitivo avanzado diseñado para interpretar, sintetizar y responder con precisión a consultas complejas sobre documentos estructurados y no estructurados, mediante lenguaje natural. Combina:

* RAG (Retrieval-Augmented Generation)
* Agentes de razonamiento LangChain
* OCR con modelos como Donut y LayoutLM
* Transcripción por voz con WhisperX
* Memoria vectorial personalizada (ChromaDB / MongoDB)

Su misión es transformar información compleja en conocimiento útil y contextualizado, permitiendo una interacción fluida, natural y trazable con grandes volúmenes de información.

---

## 🎮 Demo

![Demo GIF](./assets/demo.gif)

[Ver video de demo](https://youtu.be/tu-enlace)

---

## 🚀 Funcionalidades Principales

* ✅ Entrada por texto y voz (WhisperX)
* 🔍 Búsqueda semántica en documentos (PDF/CSV/escaneados)
* 🧠 Generación de respuestas con GPT-4o (OpenAI API)
* 📅 Memoria vectorial por usuario
* 🕵️ Agentes LangChain para razonamiento multietapas
* 📄 OCR inteligente con Donut/LayoutLM
* 📊 Panel de métricas y actividad
* 🛡️ Seguridad JWT + cifrado AES256

---

## ⚙️ Instalación y Requisitos

git clone https://github.com/tuusuario/smartassistai.git
cd smartassistai
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run frontend/app.py


### Requisitos

* Python 3.10+
* API Key de OpenAI (GPT-4)
* MongoDB (local o Atlas)
* (Opcional) Docker para despliegue

---

## 🧪 Uso

1. Abre la app con `streamlit run frontend/app.py`
2. Sube un documento PDF, CSV o imagen escaneada
3. Escribe o pronuncia una pregunta
4. Obtendras una respuesta contextualizada y trazable

**Ejemplo:**

> "¿Cuál fue el EBITDA de Apple en 2022 según este informe PDF?"

---

## 🛠️ Arquitectura del Sistema

```mermaid
graph TD
    A[Usuario] -->|Texto / Voz / Documento| B[Frontend - Streamlit]
    B --> C[Backend - FastAPI]
    C --> D[OCR Engine (Donut/LayoutLM)]
    C --> E[WhisperX Transcriber]
    C --> F[LangChain RAG + Agents]
    F --> G[Embeddings + ChromaDB]
    F --> H[OpenAI GPT-4o API]
    C --> I[Memoria vectorial por usuario]
    C --> J[MongoDB: historial, resúmenes, alertas]
    B --> K[Panel de métricas / actividad]
```

---

## 🔧 Tecnologías Utilizadas

* 🧠 OpenAI GPT-4o
* 🔍 LangChain + Agents + RAG
* 🎤 WhisperX (entrada por voz)
* 🔢 Donut / LayoutLM (OCR multimodal)
* 🧼 ChromaDB / MongoDB (memoria semántica)
* 🌐 Streamlit (frontend)
* 🚀 FastAPI (backend)
* 🛠️ Docker (despliegue)

---

## 📆 Estado del Proyecto

✅ MVP funcional completo. En fase de optimización y presentación final.

---

## 📂 Estructura del Proyecto

## 📂 Estructura del Proyecto

Knowledge_Navigator/
.
├── backend/                          # (Vacío o reservado) Para lógica de servidor si se despliega como API
├── data/                             # Datos de entrada del sistema
│   ├── pdfs/                         # Documentos fuente utilizados para generar embeddings
│   │   ├── 1. Framework for the Ethical Use of Advanced Data Science.pdf     # Documento base sobre ética en ciencia de datos
│   │   ├── 2. Governing AI: Upholding Human Rights (Data & Society).pdf      # Ética y derechos humanos en IA
│   │   ├── 3. AI and Human Society – Japón\012.pdf                           # Perspectiva japonesa sobre IA y sociedad
│   │   ├── 4. Tech Ethics Best Practices – Markkula Center.pdf               # Buenas prácticas éticas en tecnología
│   │   ├── 5. Gobernanza algorítmica y auditoría de sesgo (Bustelo).pdf     # Control y auditoría de sesgos algorítmicos
│   │   ├── 6. IA y Derecho de Daños (Berenguer et al.).pdf                  # Marco legal de responsabilidad por IA
│   │   ├── 7. IA y Transparencia Algorítmica – G. Vestri.pdf                # Transparencia y trazabilidad en IA
│   │   └── CONTENIDOPDF1_7.md           # Resumen general del contenido de los 7 PDF
│   └── PREGUNTAS.md                    # Conjunto de preguntas de prueba para testear el sistema
├── frontend/                          # Interfaz de usuario (actualmente vacía o mínima)
├── Makefile                           # Comandos automatizados para configurar, ejecutar y limpiar el proyecto
├── README.md                          # Descripción general del proyecto para usuarios y desarrolladores
├── requirements.txt                   # Dependencias Python reales necesarias para ejecutar el sistema
├── setup_env.sh                       # Script para crear entorno virtual e instalar dependencias automáticamente
├── scripts/                           # Scripts funcionales del sistema
│   ├── ask_pdf_qdrant_mongodb.py     # Asistente conversacional principal (RAG + memoria + MongoDB + JSON)
│   ├── detectar_similitud.py         # Compara preguntas nuevas con historial (detección semántica redundante)
│   ├── evaluar_con_ragas.py          # Evaluación automática del sistema usando RAGAS (fidelidad, relevancia, etc.)
│   ├── historial.json                # Registro local en JSON de preguntas, respuestas y fuentes usadas
│   ├── ingest_pdf_qdrant.py          # Carga, divide y sube PDFs a Qdrant generando embeddings
│   ├── razonador_cot.py              # Módulo de razonamiento paso a paso con auto-preguntas (Chain-of-Thought)
│   └── verificar_documento_qdrant.py # Verifica si un documento PDF ya ha sido indexado en Qdrant (por nombre o similitud)


### Equipo

* Alexander Murfitt 
* Aaron
* Eugenio

## ✨ Bonus

* [ ] ⚡ Badges (estado build, versión, licencia)
* [ ] ❓ FAQ
* [ ] 📊 CHANGELOG.md
* [ ] 📅 Roadmap
