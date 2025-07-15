from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)  # ajusta host/puerto según tu setup

collections = client.get_collections()
print(collections)
