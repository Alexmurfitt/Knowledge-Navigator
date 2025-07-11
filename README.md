# SmartAssistAI Pro

Asistente inteligente multimodal con IA generativa, razonamiento autónomo y escalabilidad profesional.

---

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

```bash
git clone https://github.com/tuusuario/smartassistai.git
cd smartassistai
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run frontend/app.py
```

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

```
AI_Knowledge_Navigator/
├── backend/
│   ├── main.py
│   ├── rag_engine.py
│   ├── agent_executor.py
│   ├── ocr_engine.py
│   ├── whisperx_transcriber.py
│   ├── memory_manager.py
│   ├── summarizer.py
│   └── security.py
├── frontend/
│   ├── app.py
│   └── components/
├── data/
│   ├── uploads/
│   └── vector_db/
├── assets/
├── tests/
├── .env
├── requirements.txt
└── README.md
```

---

## 🤝 Contribución

¡Toda ayuda es bienvenida!

1. Haz un fork del repositorio
2. Crea una rama (`git checkout -b feature/nueva-funcion`)
3. Haz commit de tus cambios
4. Abre un Pull Request

### Equipo

* Alexander Murfitt — Coordinación, arquitectura y desarrollo
* Nombre 2 — Backend e IA
* Nombre 3 — Interfaz y experiencia de usuario

---

## 🛡️ Licencia

Este proyecto está bajo licencia MIT. Consulta el archivo `LICENSE` para más detalles.

---

## 📨 Contacto

* Email: [alexander@email.com](mailto:alexander@email.com)
* Web: [www.smartassistai.com](http://www.smartassistai.com)
* GitHub: [@alexmurfitt](https://github.com/alexmurfitt)

---

## ✨ Bonus

* [ ] ⚡ Badges (estado build, versión, licencia)
* [ ] ❓ FAQ
* [ ] 📊 CHANGELOG.md
* [ ] 📅 Roadmap
