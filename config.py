# config.py
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL         = "openai/gpt-oss-120b"      

EMBEDDING_MODEL = "all-MiniLM-L6-v2"       

CHROMA_DIR      = "./chroma_db"            
COLLECTION_NAME = "course_materials"

CHUNK_SIZE    = 512                        
CHUNK_OVERLAP = 64                         

TOP_K = 5                                  

ANALYTICS_DB = "sqlite:///analytics.db"

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
