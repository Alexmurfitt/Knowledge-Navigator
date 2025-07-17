.PHONY: setup run rag limpia_reqs

setup:
	@echo "🔧 Configurando entorno virtual..."
	@bash setup_env.sh

run:
	@echo "🚀 Ejecutando sistema de consulta RAG..."
	@python scripts/ask_pdf_qdrant_mongodb.py

rag:
	@echo "🧪 Ejecutando evaluación automática con RAGAS..."
	@python scripts/evaluar_con_ragas.py

limpia_reqs:
	@echo "📦 Limpiando dependencias no usadas..."
	@python scripts/clean_requirements.py
