import os
from langchain_google_community import GoogleSearchAPIWrapper
from dotenv import load_dotenv

load_dotenv()

search = GoogleSearchAPIWrapper(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    google_cse_id=os.getenv("GOOGLE_CSE_ID")
)

print("\n🔍 Resultado de prueba:")
print(search.run("¿Qué es la inteligencia artificial humanitaria?"))
