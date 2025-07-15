#!/bin/bash

echo "⚙️ Creando nuevo entorno virtual 'my_env'..."
python3 -m venv my_env

echo "📦 Activando entorno..."
source my_env/bin/activate

echo "📦 Instalando dependencias desde requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Entorno recreado con éxito. Ya puedes continuar trabajando."
