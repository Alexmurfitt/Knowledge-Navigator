# Atajos de gestión del proyecto

.PHONY: setup clean run

setup:
	@echo "🔧 Configurando entorno virtual..."
	@bash setup_env.sh

clean:
	@echo "🧹 Limpiando proyecto..."
	@bash clean_project.sh

run:
	@echo "🚀 Ejecutando aplicación principal..."
	@source my_env/bin/activate && python scripts/main.py
