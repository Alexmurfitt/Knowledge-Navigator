import importlib

modules = [
    "langchain_ollama",
    "langchain_qdrant",
    "langchain_community.document_loaders",
    "qdrant_client"
]

for module in modules:
    try:
        importlib.import_module(module)
        print(f"✅ {module} encontrado")
    except ImportError:
        print(f"❌ {module} no está instalado")
