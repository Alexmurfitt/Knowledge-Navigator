# 🧠 Knowledge Navigator – Sistema Inteligente de Asistencia Documental (v3)

Knowledge Navigator es una plataforma avanzada de asistencia documental impulsada por modelos de lenguaje generativo (LLMs) y arquitectura RAG (Retrieval-Augmented Generation), diseñada para responder preguntas complejas en lenguaje natural utilizando documentos internos, fuentes web externas y razonamiento contextual estructurado.

---

## 🎯 Objetivo del Sistema

Knowledge Navigator permite a usuarios empresariales, técnicos o investigativos:

* ✅ Consultar documentos extensos o técnicos en lenguaje natural.
* ✅ Obtener respuestas claras, trazables y justificadas.
* ✅ Evaluar automáticamente la calidad del razonamiento.
* ✅ Subir, indexar y eliminar documentos en tiempo real.
* ✅ Complementar el conocimiento interno con fuentes confiables externas.

> Todo ello bajo una arquitectura modular, segura y escalable, ideal para entornos productivos o regulados.

---

## ⚙️ Tecnologías y Herramientas Clave

| Categoría                | Herramientas / Modelos                                  |
| ------------------------ | ------------------------------------------------------- |
| Lenguaje                 | Python 3.10+                                            |
| Framework API            | FastAPI + Uvicorn                                       |
| Almacenamiento Vectorial | Qdrant (API + filtros semánticos + indexado jerárquico) |
| Embeddings               | OllamaEmbeddings con `mxbai-embed-large`                |
| LLM principal            | Gemini 2.5 Flash (vía LangChain)                        |
| Prompting                | LangChain `prompt_templates` + `LLMChain`               |
| Evaluación RAG           | LangChain Evaluation + RAGAS (opcional)                 |
| PDFs                     | PyMuPDF (`fitz`) + LangChain PDFLoader                  |
| Búsqueda web externa     | GoogleSearchAPIWrapper (LangChain Community)            |
| Seguridad                | `passlib[bcrypt]` + autenticación JWT opcional          |
| WebSocket                | Para progreso de carga documental en tiempo real        |
| DevOps                   | `.env`, Docker, estructura modular desacoplada          |

---

## 🧱 Arquitectura General

```text
             🧑 Usuario (cliente web, CLI o app externa)
                        │
                        ▼
              ┌───────────────────────┐
              │ FastAPI (main.py)     │
              └─────────┬─────────────┘
                        ▼
       ┌────────────────────────────────────────────┐
       │                Services/                   │
       │ ┌────────────────────────────────────────┐ │
       │ │ consultas.py        → Lógica principal   │ │
       │ │ razonador.py        → Prompt + modelo   │ │
       │ │ pdf_ingest.py       → Chunking + metadatos││
       │ │ buscador_externo.py → Web CSE wrapper   │ │
       │ │ evaluator.py         → Relevance check  │ │
       │ └────────────────────────────────────────┘ │
       └────────────────────────────────────────────┘
                        │
                        ▼
              Qdrant Vector Store  +  Google CSE API
```

---

## 🧩 Descripción de los Módulos Principales

### `consultas.py`

* Clasifica preguntas simples vs complejas.
* Recupera contexto desde Qdrant.
* Evalúa relevancia del contexto con `load_evaluator`.
* Activa fallback web si es necesario.
* Llama al razonador con el contexto adecuado.

### `razonador.py`

* Prompt estructurado para:

  * Respuestas pedagógicas (modo completo).
  * Respuestas directas (modo conciso).
* Separación clara entre definición y explicación.

### `pdf_ingest.py`

* Procesamiento jerárquico con `fitz.get_toc()`.
* Extracción de tablas y bloques etiquetados (`H1–H4`).
* Chunking semántico con `RecursiveCharacterTextSplitter`.
* Indexado en Qdrant con metadatos enriquecidos.

### `buscador_externo.py`

* Consultas a Google Custom Search API.
* Limpieza, normalización y selección de snippets.
* Devuelve snippets, títulos y enlaces de resultados.

### `evaluator.py`

* Evaluación automática con LangChain Evaluator (`relevance`).
* Extensible a RAGAS (`faithfulness`, `precision`, `recall`).

---

## 🔄 Flujo de Datos (Consulta)

1. 🧠 El usuario realiza una pregunta en lenguaje natural.
2. 🧩 El sistema detecta si es una pregunta simple o compleja.
3. 📚 Recupera contexto desde Qdrant.
4. 🧪 Evalúa la utilidad del contexto recuperado.
5. 🌐 Si no es relevante (o se solicita), activa búsqueda web.
6. 🧠 Construye un prompt con historial, contexto y pregunta.
7. 🤖 Genera respuesta con Gemini 2.5 Flash.
8. 🔎 Separa: respuesta principal, explicación, fuentes.
9. 💾 Guarda interacción en la memoria conversacional.
10. ✅ Devuelve respuesta estructurada al cliente.

---

## 🧪 Flujo de Evaluación (opcional)

1. Evalúa `relevance` del contexto con evaluador LangChain.
2. Si está activado RAGAS:

   * Calcula: `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`.
   * Registra métricas en logs o dashboard.

---

## 📥 Ingesta de PDFs

1. 📤 El usuario sube uno o varios archivos PDF.
2. 🧭 Detecta marcadores (`TOC`) o analiza layout del documento.
3. 🔍 Extrae:

   * Tablas (convertidas a Markdown)
   * Bloques de texto jerarquizados (H1, H2, ...)
4. 🧠 Asigna metadatos enriquecidos: título, jerarquía, página, tipo.
5. ✂️ Fragmenta el contenido en chunks semánticos.
6. 🔗 Genera embeddings.
7. 🚀 Indexa los chunks en Qdrant.

---

## 📦 Entradas y Salidas

| Tipo    | Formato                                          |
| ------- | ------------------------------------------------ |
| Entrada | `str` (pregunta), `UploadFile` (PDFs)            |
| Entrada | WebSocket (`filename`, `progress`)               |
| Entrada | Autenticación (opcional): `username`, `password` |
| Salida  | JSON: `respuesta`, `razonamiento`, `fuentes`     |
| Salida  | Lista de mensajes de historial conversacional    |
| Salida  | Evaluación de respuestas (relevance / RAGAS)     |

---

## 📑 Dependencias Recomendadas (`requirements.txt`)

```txt
fastapi
uvicorn
langchain
langchain-google-genai
langchain-qdrant
langchain-community
qdrant-client
ragas
fitz
pymupdf
python-dotenv
passlib[bcrypt]
```

---

## 🔐 Seguridad (opcional)

* 🔐 Sistema de login/register con `passlib[bcrypt]`
* 🔒 Claves de API protegidas mediante `.env`
* 🧩 Filtros por `document_name_id` en Qdrant para eliminar documentos específicos
* 🛡️ Autenticación JWT opcional para endpoints protegidos

---

## ✅ Conclusión

Knowledge Navigator v3 es un sistema de asistencia documental moderno, modular y potente. Integra:

* 🧠 Razonamiento generativo explicativo
* 📚 Trazabilidad documental con metadatos
* 📥 Ingesta semántica jerárquica
* 🧪 Evaluación de calidad automática (opcional)
* 🌐 Capacidad de ampliación con búsqueda web y seguridad

Es ideal para entornos **corporativos, legales, educativos o regulados**, y puede:

* Desplegarse como backend empresarial
* Integrarse con una interfaz personalizada (SPA, chatbot, etc.)
* O funcionar como motor semántico dentro de sistemas más amplios

> ¡Una solución avanzada para navegar el conocimiento con inteligencia!
