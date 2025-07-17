# Knowledge Navigator

**Knowledge Navigator** es un asistente inteligente diseñado para proporcionar respuestas enriquecidas y contextuales, utilizando **inteligencia artificial** (IA) avanzada, **razonamiento autónomo** (Chain-of-Thought), y **búsqueda semántica** (RAG). Este sistema es ideal para escenarios como ética, derecho, medicina, y más, permitiendo obtener respuestas no solo basadas en datos, sino también mejoradas por razonamiento interno y la integración de información relevante de documentos externos.

---

## **Características Principales**

1. **Razonamiento Autónomo (Chain of Thought)**
   - El sistema genera respuestas a preguntas no solo basadas en los datos existentes, sino también mediante un razonamiento interno. A través de **auto-preguntas**, el sistema se enriquece con contexto adicional y con inferencias antes de proporcionar una respuesta.

2. **Búsqueda Semántica (RAG)**
   - **Retrieval-Augmented Generation** (RAG) permite al sistema recuperar información contextualizada de una base de datos o documentos relevantes (por ejemplo, PDFs), fusionando los resultados obtenidos con la generación de respuestas basada en el modelo de lenguaje.

3. **Memoria Conversacional**
   - El sistema tiene capacidad para almacenar y gestionar el historial de interacciones, utilizando **ConversationBufferMemory**. Esto le permite mantener el contexto a lo largo de las conversaciones, comprendiendo preguntas dependientes y referencias anafóricas.

4. **Detección de Preguntas Redundantes**
   - Utilizando técnicas de **similitud semántica** (embeddings y similitud coseno), el sistema detecta si una pregunta ya ha sido formulada previamente, evitando generar respuestas duplicadas.

5. **Evaluación Automática de Respuestas**
   - El sistema utiliza **RAGAS** para evaluar la calidad de las respuestas generadas, midiendo **precisión contextual**, **fidelidad**, **relevancia** y **cobertura**.

6. **Almacenamiento y Seguimiento de Historial**
   - Las interacciones (preguntas, respuestas, fuentes) se almacenan en **JSON** y en **MongoDB**, lo que permite realizar un seguimiento y análisis detallado de cada interacción.

---

## **Tecnologías Utilizadas**

1. **Ollama (ChatOllama, OllamaEmbeddings)**
   - Utilizado para generar respuestas y razonamiento interno, y para crear **embeddings** semánticos de preguntas y documentos.

2. **LangChain**
   - Gestiona el flujo del sistema, conectando el razonamiento interno, la búsqueda semántica y la generación de respuestas utilizando **ConversationalRetrievalChain** y **LLMChain**.

3. **Qdrant**
   - Base de datos **vectorial** para almacenar y recuperar documentos semánticamente, optimizando la búsqueda de información relevante mediante similitudes de vectores.

4. **MongoDB**
   - Base de datos utilizada para almacenar el historial de interacciones del sistema, permitiendo consultas rápidas y la trazabilidad de las interacciones.

5. **RAGAS**
   - **RAGAS** se utiliza para evaluar la calidad de las respuestas generadas por el sistema en términos de **precisión**, **fidelidad**, **relevancia** y **cobertura**.

6. **Python y Bibliotecas Relacionadas**
   - El proyecto está desarrollado en **Python** con bibliotecas como **Sklearn** (para la similitud coseno), **FastAPI** o **Streamlit** (para interfaces web), entre otras herramientas.

---

## **Estructura del Proyecto**

```bash
Knowledge_Navigator/
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
│   ├── __pycache__
│   │   └── razonador_cot.cpython-312.pyc
│   ├── razonador_cot.py              # Módulo de razonamiento paso a paso con auto-preguntas (Chain-of-Thought)
│   └── verificar_documento_qdrant.py # Verifica si un documento PDF ya ha sido indexado en Qdrant (por nombre o similitud)
└── setup_env.sh

Instalación y Requisitos
Clonar el repositorio

bash
Copiar
git clone https://github.com/tu-usuario/knowledge-navigator.git
cd knowledge-navigator
Instalar dependencias

bash
Copiar
pip install -r requirements.txt
Configurar las variables de entorno:
Asegúrate de tener el archivo .env configurado con las siguientes variables:

MONGO_URI: URI de conexión a MongoDB.

MONGO_DB_NAME: Nombre de la base de datos en MongoDB.

QDRANT_URL: URL de conexión a Qdrant.

QDRANT_API_KEY: Clave de API de Qdrant.

Flujo de Trabajo
Entrada del Usuario: El usuario realiza una pregunta.

Análisis Semántico: Se analiza la intención del usuario mediante embeddings generados por OllamaEmbeddings.

Generación de Auto-Preguntas (Chain of Thought): El sistema se formula entre 2 y 4 auto-preguntas para enriquecer la respuesta.

Búsqueda en Documentos (RAG): Si es necesario, el sistema busca en la base de datos utilizando Qdrant.

Generación de Respuesta: Se genera una respuesta final, que se enriquece con la información derivada de las auto-preguntas.

Formato de Salida: Se presenta la respuesta junto con información adicional de las auto-preguntas.

Almacenamiento de Historial: La interacción (pregunta, respuesta, fuentes) se guarda en MongoDB y en un archivo JSON.

Próximos Pasos
Implementar Detección de Redundancias:
Se avanzará en la comparación semántica de preguntas utilizando embeddings y similitud coseno, lo cual evitará respuestas redundantes.

Evaluación Automática del Sistema:
Usaremos RAGAS para medir la calidad de las respuestas generadas.


### **Cambios Principales en el `README.md`:**

1. **Estructura del Proyecto**: Se actualizó la estructura de carpetas para reflejar la organización actual del proyecto.
2. **Tecnologías Utilizadas**: Se destacaron las principales tecnologías que hacen funcionar el sistema, como **Ollama**, **LangChain**, **Qdrant**, **MongoDB** y **RAGAS**.
3. **Flujo de Trabajo**: Se describió el proceso desde la entrada del usuario hasta el almacenamiento de respuestas y metadatos.
4. **Próximos Pasos**: Se mencionaron las próximas acciones que se implementarán para mejorar el sistema, como la **detección de redundancias** y la **evaluación automática**.




