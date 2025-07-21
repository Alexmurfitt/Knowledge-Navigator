from langchain_google_community import GoogleSearchAPIWrapper
from dotenv import load_dotenv
import os

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
google_cse_id = os.getenv("GOOGLE_CSE_ID")

search = GoogleSearchAPIWrapper(
    google_api_key=google_api_key,
    google_cse_id=google_cse_id
)

query = "¿Qué es el humanitarismo digital?"
results = search.run(query)
print(results)
