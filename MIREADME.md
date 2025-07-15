# 📘 README.md

# Knowledge Navigator 🧠

Asistente inteligente para razonamiento sobre información compleja usando IA generativa, LangChain, NLP y análisis vectorial.

## 🚀 Instalación

make setup
▶️ Ejecución

make run
🧹 Limpieza

make clean
📦 Exportar para entrega

make zip
📦 Contenido
data/: PDFs, vectores y recursos.

scripts/: Código fuente principal.

requirements.txt: Dependencias reales del proyecto.

---

## 🛠 `Makefile`

setup:
	@bash setup_env.sh

run:
	source my_env/bin/activate && python scripts/main.py

clean:
	@bash clean_project.sh

cleanreqs:
	pipreqs . --force --encoding=utf-8 --ignore-errors

zip:
	@bash zip_project.sh
⚙️ setup_env.sh

#!/bin/bash
echo "⚙️ Creando entorno virtual..."
python3 -m venv my_env
source my_env/bin/activate
echo "📦 Instalando dependencias..."
pip install -r requirements.txt
echo "✅ Entorno listo."
🧹 clean_project.sh

#!/bin/bash
echo "🧹 Limpiando archivos innecesarios..."
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -r {} +
find . -name ".ipynb_checkpoints" -type d -exec rm -r {} +
rm -rf my_env
echo "✅ Proyecto limpio."
📦 zip_project.sh

#!/bin/bash
echo "📦 Generando .zip final limpio..."
zip -r Knowledge-Navigator.zip . -x "my_env/*" "*.pyc" "__pycache__/*" "*.ipynb_checkpoints/*"
echo "✅ Archivo Knowledge-Navigator.zip creado."

📄 .gitignore
my_env/
__pycache__/
*.pyc
.ipynb_checkpoints/
*.zip
✅ ¿Qué debes hacer ahora?
Copia y guarda todos los archivos anteriores.

Hazlos ejecutables:

chmod +x *.sh
Ejecuta:

make setup
make run
make zip
Y tendrás tu proyecto listo para entrega profesional o subida a GitHub.

