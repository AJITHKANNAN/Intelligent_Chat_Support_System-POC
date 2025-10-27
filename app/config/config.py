import os
from dotenv import load_dotenv

load_dotenv()

DB_FAISS_PATH = os.path.join(os.getcwd(), "db", "vectorstore_faiss")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
